import os
import cv2
import json
import hashlib
import numpy as np
from PIL import Image
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass
import onnxruntime as ort
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
        # Limit OpenCV threads to prevent thread pool memory explosion on cloud servers
        cv2.setNumThreads(1)

        # Configure ONNX Runtime to use minimal memory (<100MB) without pre-allocating arenas
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_cpu_mem_arena = False

        # Model selection: Defaults to lightweight 'buffalo_s' (<100MB RAM) for cloud free tiers (e.g. Render 512MB)
        model_name = os.getenv("FACE_MODEL", "buffalo_s")
        self.app = FaceAnalysis(
            name=model_name, 
            allowed_modules=['detection', 'recognition'], 
            providers=['CPUExecutionProvider'],
            session_options=sess_options
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # PERSISTENT STORAGE: Uses cloud Qdrant if credentials provided, else local folder
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
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
        try:
            from PIL import Image, ImageOps
            pil_img = Image.open(image_path)
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception as e:
            img = cv2.imread(image_path)
            if img is not None:
                return img
            print(f"[ERROR] Failed to read image '{image_path}': {e}")
            return None

    def extract_faces_from_path(self, image_path: str):
        img = self._read_image(image_path)
        if img is None:
            return []
        img = self._resize_if_needed(img)
        return self.app.get(img)

    def index_event_image(self, event_id: str, image_filename: str, full_path: str, download_url: str | None = None):
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
                        "preview_url": f"/photos/{event_id}/{image_filename}",
                        "download_url": download_url or f"/photos/{event_id}/{image_filename}",
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

    def get_indexed_count(self, event_id: str) -> int:
        """Fast query returning number of face vectors indexed for this event."""
        try:
            return self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="event_id",
                            match=MatchValue(value=event_id)
                        )
                    ]
                )
            ).count
        except Exception:
            return 0

    def get_indexed_filenames(self, event_id: str) -> set[str]:
        """Returns a set of all file names already indexed in Qdrant for this event."""
        try:
            indexed_files = set()
            offset = None
            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="event_id",
                                match=MatchValue(value=event_id)
                            )
                        ]
                    ),
                    with_payload=["file_name"],
                    limit=250,
                    offset=offset
                )
                for record in records:
                    fn = record.payload.get("file_name")
                    if fn:
                        indexed_files.add(fn)
                if offset is None:
                    break
            return indexed_files
        except Exception as e:
            print(f"[DEBUG] Failed to fetch indexed filenames: {e}")
            return set()

    def convert_heic_if_needed(self, file_path: str) -> tuple[str, str]:
        """Converts HEIC/HEIF file to JPG if needed and returns (actual_file_path, filename)."""
        filename = os.path.basename(file_path)
        if filename.lower().endswith(('.heic', '.heif')):
            base_name, _ = os.path.splitext(filename)
            jpg_filename = f"{base_name}.jpg"
            jpg_path = os.path.join(os.path.dirname(file_path), jpg_filename)
            try:
                from PIL import Image, ImageOps
                import pillow_heif
                pillow_heif.register_heif_opener()
                with Image.open(file_path) as im:
                    im = ImageOps.exif_transpose(im)
                    im.convert("RGB").save(jpg_path, "JPEG", quality=95)
                return jpg_path, jpg_filename
            except Exception as e:
                print(f"[ERROR] Failed to convert {filename} to JPG: {e}")
        return file_path, filename

    def optimize_to_web_preview(self, file_path: str, max_dim: int = 900, quality: int = 75) -> str:
        """
        Compresses an image in-place to a lightweight web preview (approx 50-80 KB),
        reducing storage by ~95-98% while keeping crisp visual quality for screens.
        Applies EXIF transpose so photos are always physically upright.
        """
        try:
            from PIL import Image, ImageOps
            # Don't re-compress if already small (under 120 KB) and not rotated
            with Image.open(file_path) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                im.save(file_path, "JPEG", quality=quality, optimize=True)
            return file_path
        except Exception as e:
            print(f"[ERROR] Failed to optimize web preview for {file_path}: {e}")
            return file_path

    def index_single_image(self, event_id: str, file_path: str, download_url: str | None = None) -> int:
        """Helper to convert, extract face vectors from full-res image, then optimize to lightweight preview."""
        actual_path, filename = self.convert_heic_if_needed(file_path)
        # 1. Extract faces and vectors from full-res image for 100% precision
        faces_found = self.index_event_image(event_id, filename, actual_path, download_url=download_url)
        # 2. Immediately shrink local file to lightweight ~60 KB preview to save 97% storage
        self.optimize_to_web_preview(actual_path)
        return faces_found

    def index_event_folder(self, event_id: str, event_folder: str, force_reindex: bool = False, drive_links: dict | None = None) -> dict:
        """Scans and indexes an event directory. Performs incremental indexing unless force_reindex is True."""
        if not os.path.exists(event_folder):
            raise FileNotFoundError(f"Folder for event '{event_id}' not found.")

        valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')
        if force_reindex:
            self.clear_event(event_id)
            already_indexed = set()
        else:
            already_indexed = self.get_indexed_filenames(event_id)

        processed = 0
        newly_indexed = 0
        faces_count = 0
        if not drive_links:
            links_file = os.path.join(event_folder, "gdrive_links.json")
            if os.path.exists(links_file):
                try:
                    with open(links_file, "r", encoding="utf-8") as lf:
                        drive_links = json.load(lf)
                except Exception as e:
                    print(f"[WARN] Failed to read gdrive_links.json: {e}")
                    drive_links = {}
                drive_links = {}

        # Sync download URLs for any existing photos in this event
        if drive_links:
            self.update_event_download_urls(event_id, drive_links)

        for file_name in os.listdir(event_folder):
            if file_name.lower().endswith(valid_extensions):
                processed += 1
                base_name, ext = os.path.splitext(file_name)
                check_name = f"{base_name}.jpg" if ext.lower() in ('.heic', '.heif') else file_name

                if not force_reindex and check_name in already_indexed:
                    continue

                full_path = os.path.join(event_folder, file_name)
                dl_url = drive_links.get(file_name) or drive_links.get(check_name)
                faces_found = self.index_single_image(event_id, full_path, download_url=dl_url)
                newly_indexed += 1
                faces_count += faces_found

        total_indexed_files = len(self.get_indexed_filenames(event_id))
        return {
            "status": "success",
            "event_id": event_id,
            "images_in_folder": processed,
            "new_images_indexed": newly_indexed,
            "new_faces_indexed": faces_count,
            "total_indexed_images": total_indexed_files
        }

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

    def update_event_download_urls(self, event_id: str, drive_links: dict):
        """Updates the download_url in Qdrant for existing photos without needing re-indexing."""
        if not drive_links:
            return 0
        updated = 0
        for fname, dl_url in drive_links.items():
            if not dl_url:
                continue
            try:
                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload={"download_url": dl_url},
                    points=Filter(
                        must=[
                            FieldCondition(key="event_id", match=MatchValue(value=event_id)),
                            FieldCondition(key="file_name", match=MatchValue(value=fname)),
                        ]
                    )
                )
                updated += 1
            except Exception as e:
                print(f"[WARN] Failed to update download_url for {fname}: {e}")
        return updated

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
                preview = hit.payload.get("preview_url") or hit.payload.get("relative_url")
                download = hit.payload.get("download_url") or hit.payload.get("relative_url")
                matched_photos.append({
                    "file_name": file_name,
                    "url": preview,
                    "preview_url": preview,
                    "download_url": download,
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