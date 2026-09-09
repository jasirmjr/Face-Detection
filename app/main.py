import os
import shutil
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.engine import FaceEngine

STORAGE_DIR = "storage/events"
TEMP_DIR = "storage/temp_selfies"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

engine: Optional[FaceEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = FaceEngine()
    try:
        yield
    finally:
        if engine is not None:
            engine.close()

app = FastAPI(title="Event Face Finder API", lifespan=lifespan)

app.mount("/photos", StaticFiles(directory=STORAGE_DIR), name="photos")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/api/events/{event_id}/index")
def index_event(event_id: str):
    if engine is None:
        raise HTTPException(status_code=503, detail="Face engine is not initialized.")

    event_folder = os.path.join(STORAGE_DIR, event_id)
    if not os.path.exists(event_folder):
        raise HTTPException(status_code=404, detail=f"Directory for event '{event_id}' not found.")

    # Clear previously indexed vectors for this event to prevent duplicates
    engine.clear_event(event_id)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')
    indexed_files = 0
    total_faces = 0

    for file_name in os.listdir(event_folder):
        if file_name.lower().endswith(valid_extensions):
            file_path = os.path.join(event_folder, file_name)

            # Convert HEIC/HEIF to JPG so browsers can render thumbnails
            if file_name.lower().endswith(('.heic', '.heif')):
                base_name, _ = os.path.splitext(file_name)
                jpg_filename = f"{base_name}.jpg"
                jpg_path = os.path.join(event_folder, jpg_filename)
                try:
                    from PIL import Image
                    import pillow_heif
                    pillow_heif.register_heif_opener()
                    with Image.open(file_path) as im:
                        im.convert("RGB").save(jpg_path, "JPEG", quality=95)
                    file_name = jpg_filename
                    file_path = jpg_path
                except Exception as e:
                    print(f"[ERROR] Failed to convert {file_name} to JPG: {e}")

            faces_found = engine.index_event_image(event_id, file_name, file_path)
            indexed_files += 1
            total_faces += faces_found

    return {
        "status": "success",
        "event_id": event_id,
        "images_processed": indexed_files,
        "faces_indexed": total_faces
    }

@app.post("/api/events/{event_id}/search")
async def search_faces(
    event_id: str, 
    selfie: UploadFile = File(...),
    threshold: float = Form(0.45)
):
    if engine is None:
        raise HTTPException(status_code=503, detail="Face engine is not initialized.")

    temp_path = os.path.join(TEMP_DIR, selfie.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    try:
        matches = engine.search_by_selfie(temp_path, event_id=event_id, similarity_threshold=threshold)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "event_id": event_id,
        "total_matches": len(matches),
        "results": matches
    }