import os
import cv2
from app.engine import FaceEngine

engine = FaceEngine()

# 1. Check event folder
event_folder = "storage/events/annual-meet"
files = [f for f in os.listdir(event_folder) if not f.startswith('.')] if os.path.exists(event_folder) else []
print(f"\n[1] Files in '{event_folder}': {files}")

# 2. Test reading & face detection on the event photos
for f in files:
    path = os.path.join(event_folder, f)
    img = cv2.imread(path)
    if img is None:
        print(f"  [ERROR] OpenCV failed to read: {f} (Path might be wrong or file corrupted)")
        continue
    faces = engine.extract_faces_from_path(path)
    print(f"  -> File '{f}': Found {len(faces)} face(s)")

# 3. Test Qdrant Collection Count
count = engine.client.count(collection_name=engine.collection_name)
print(f"\n[2] Total vectors stored inside Qdrant: {count.count}")

# 4. Force index all photos now
print("\n[3] Re-indexing event images directly...")
indexed_total = 0
for f in files:
    path = os.path.join(event_folder, f)
    num = engine.index_event_image("annual-meet", f, path)
    indexed_total += num
print(f"Direct indexing complete. Faces indexed: {indexed_total}")

# 5. Check if search works on the exact same image
if files:
    test_photo = os.path.join(event_folder, files[0])
    print(f"\n[4] Testing self-search using '{files[0]}' against the database...")
    matches = engine.search_by_selfie(test_photo, "annual-meet", similarity_threshold=0.1)
    print(f"Matches found: {len(matches)}")
    for m in matches:
        print(f"  Match: {m['file_name']} with Score: {m['similarity_score']}")

engine.close()