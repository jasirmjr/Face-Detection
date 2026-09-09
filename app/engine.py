import os
import cv2
import hashlib
import numpy as np
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
from insightface.app import FaceAnalysis
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct, 
    Filter, 
    FieldCondition, 
    MatchValue,
    PayloadSchemaType
)

class FaceEngine:
    def __init__(self, collection_name="event_faces"):
        self.app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # PERSISTENT STORAGE: Saves vectors to a folder so server restarts won't delete them
        db_path = os.path.abspath("qdrant_db")
        os.makedirs(db_path, exist_ok=True)
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="event_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )

    def _resize_if_needed(self, img, max_dim=1920):
        """Downscale DSLR images so InsightFace doesn't miss faces due to ultra-high res."""
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    def _read_image(self, image_path: str):
        # 1. Fast OpenCV read
        img = cv2.imread(image_path)
        if img is not None:
            return img
        # 2. Fallback via PIL (supports HEIC/HEIF, non-ASCII paths, and various formats)
        try:
            pil_img = Image.open(image_path).convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[ERROR] Failed to read image '{image_path}': {e}")
            return None

    def extract_faces_from_path(self, image_path: str):
        img = self._read_image(image_path)
        if img is None:
            return []
        img = self._resize_if_needed(img)
        return self.app.get(img)

    def index_event_image(self, event_id: str, image_filename: str, full_path: str):
        faces = self.extract_faces_from_path(full_path)
        points = []

        for idx, face in enumerate(faces):
            # Deterministic unique ID using MD5 hash of event, filename, and face index
            point_id = int(hashlib.md5(f"{event_id}_{image_filename}_{idx}".encode()).hexdigest()[:15], 16)
            
            # Normalize embedding
            embedding = face.normed_embedding.tolist()
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "event_id": event_id,
                        "file_name": image_filename,
                        "relative_url": f"/photos/{event_id}/{image_filename}",
                        "bbox": [int(x) for x in face.bbox]
                    }
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        return len(points)

    def clear_event(self, event_id: str):
        """Clears all vectors for an event before re-indexing to avoid duplicates or orphaned entries."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="event_id",
                            match=MatchValue(value=event_id)
                        )
                    ]
                )
            )
        except Exception as e:
            print(f"[DEBUG] Failed to clear event {event_id}: {e}")

    def search_by_selfie(self, selfie_path: str, event_id: str, similarity_threshold: float = 0.45):
        faces = self.extract_faces_from_path(selfie_path)
        if not faces:
            print(f"[DEBUG] No face detected in uploaded selfie: {selfie_path}")
            return []

        # Use largest face in selfie
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]), reverse=True)
        query_vector = faces[0].normed_embedding.tolist()

        try:
            q_filter = Filter(
                must=[
                    FieldCondition(
                        key="event_id",
                        match=MatchValue(value=event_id)
                    )
                ]
            )
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=q_filter,
                    limit=100,
                    score_threshold=similarity_threshold
                )
                search_result = response.points
            else:
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    query_filter=q_filter,
                    limit=100,
                    score_threshold=similarity_threshold
                )
            print(f"[DEBUG] Hits found in Qdrant: {len(search_result)}")
            for hit in search_result:
                print(f"  -> Match: {hit.payload['file_name']} (Score: {hit.score})")
        except Exception as e:
            print(f"[DEBUG] Search failed: {e}")
            return []

        seen = set()
        matched_photos = []
        for hit in search_result:
            file_name = hit.payload.get("file_name")
            if file_name and file_name not in seen:
                seen.add(file_name)
                matched_photos.append({
                    "file_name": file_name,
                    "url": hit.payload.get("relative_url"),
                    "similarity_score": round(hit.score, 3)
                })

        return matched_photos

    def close(self):
        """Release Qdrant database lock and close connections."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass