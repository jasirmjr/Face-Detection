import os
import json
import shutil
import threading
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.engine import FaceEngine
from app.watcher import FolderWatcher

STORAGE_DIR = "storage/events"
TEMP_DIR = "storage/temp_selfies"
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

engine: Optional[FaceEngine] = None
watcher: Optional[FolderWatcher] = None
import_progress: dict[str, dict] = {}
_engine_lock = threading.Lock()

def get_engine() -> Optional[FaceEngine]:
    global engine, watcher
    if engine is None:
        with _engine_lock:
            if engine is None:
                try:
                    print("[INFO] Starting FaceEngine initialization in background...")
                    inst = FaceEngine()
                    engine = inst
                    if watcher is None:
                        watcher = FolderWatcher(engine=inst, watch_dir=STORAGE_DIR)
                        watcher.start()
                    print("[INFO] FaceEngine and FolderWatcher are ready!")
                except Exception as e:
                    print(f"[ERROR] Failed to initialize FaceEngine: {e}")
    return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    global watcher, engine
    # Initialize engine in background thread so port binds immediately in <0.2s!
    # Render's port detection passes immediately without timeout or startup OOM.
    threading.Thread(target=get_engine, daemon=True).start()
    try:
        yield
    finally:
        if watcher is not None:
            watcher.stop()
        if engine is not None:
            engine.close()

app = FastAPI(title="Event Face Finder API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health")
def health():
    return {"status": "ok", "engine_ready": engine is not None}

@app.get("/photos/{event_id}/{filename:path}")
async def get_photo(event_id: str, filename: str):
    """Safely serves event photos with automatic subfolder and cache resolution."""
    filename = os.path.normpath(filename).lstrip("\\/.")
    event_folder = os.path.join(STORAGE_DIR, event_id)
    direct_path = os.path.join(event_folder, filename)
    if os.path.exists(direct_path) and os.path.isfile(direct_path):
        return FileResponse(direct_path, headers={"Cache-Control": "public, max-age=86400"})

    # Fallback: search recursively inside event_folder in case photo is in a subfolder
    base_name = os.path.basename(filename)
    if os.path.exists(event_folder):
        for root, _, files in os.walk(event_folder):
            if base_name in files:
                found_path = os.path.join(root, base_name)
                if os.path.isfile(found_path):
                    return FileResponse(found_path, headers={"Cache-Control": "public, max-age=86400"})

    raise HTTPException(status_code=404, detail=f"Photo '{filename}' not found for event '{event_id}'")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/api/events/{event_id}/status")
async def event_status(event_id: str):
    event_folder = os.path.join(STORAGE_DIR, event_id)
    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')
    folder_files_count = len([f for f in os.listdir(event_folder) if f.lower().endswith(valid_exts)]) if os.path.exists(event_folder) else 0
    eng = get_engine()
    faces_count = eng.get_indexed_count(event_id) if eng else 0
    return {
        "event_id": event_id,
        "folder_exists": os.path.exists(event_folder),
        "total_photos_in_folder": folder_files_count,
        "total_faces_indexed": faces_count,
        "engine_ready": eng is not None,
        "watcher": watcher.get_status() if watcher else {"active": False}
    }

@app.post("/api/events/{event_id}/index")
def index_event(event_id: str, force: bool = Query(False)):
    eng = get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Face engine is warming up. Please try again in a few seconds.")

    event_folder = os.path.join(STORAGE_DIR, event_id)
    if not os.path.exists(event_folder):
        raise HTTPException(status_code=404, detail=f"Directory for event '{event_id}' not found.")

    try:
        result = eng.index_event_folder(event_id, event_folder, force_reindex=force)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/events/{event_id}/upload")
async def upload_photos(event_id: str, files: list[UploadFile] = File(...)):
    """Allows camera crew to upload photos directly from a phone or laptop browser."""
    event_folder = os.path.join(STORAGE_DIR, event_id)
    os.makedirs(event_folder, exist_ok=True)
    saved_files = []

    for file in files:
        save_path = os.path.join(event_folder, file.filename)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)

    return {
        "status": "success",
        "event_id": event_id,
        "uploaded_count": len(saved_files),
        "message": "Photos received. Background watcher is indexing them automatically."
    }

def run_gdrive_import_task(event_id: str, clean_url: str):
    global import_progress
    event_folder = os.path.join(STORAGE_DIR, event_id)
    os.makedirs(event_folder, exist_ok=True)

    import_progress[event_id] = {
        "status": "in_progress",
        "percent": 5,
        "message": "Connecting to Google Drive and scanning files...",
        "current_file": "",
        "downloaded_count": 0,
        "total_count": 0,
        "faces_indexed": 0,
        "error": None
    }

    try:
        import gdown
        # 1. Scan folder metadata first with skip_download=True
        folder_items = gdown.download_folder(
            url=clean_url,
            output=event_folder,
            quiet=True,
            use_cookies=False,
            skip_download=True
        )

        if not folder_items:
            raise ValueError("Could not access files in Google Drive. Ensure the folder is shared with 'Anyone with the link can view'.")

        valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')
        valid_items = []
        for item in folder_items:
            fname = os.path.basename(item.path)
            # Skip hidden / macOS resource fork / temporary files
            if fname.startswith(('.', '._', '~', '~$')) or fname.endswith(('.tmp', '.part', '.crdownload')):
                continue
            if fname.lower().endswith(valid_exts):
                valid_items.append(item)

        total_files = len(valid_items)
        if total_files == 0:
            raise ValueError("No valid image files (.jpg, .jpeg, .png, .webp, .heic) found in this Drive folder.")

        # Save/update gdrive_links.json for persistence
        drive_links = {}
        for item in valid_items:
            fname = os.path.basename(item.path)
            direct_url = f"https://drive.google.com/uc?export=download&id={item.id}"
            drive_links[fname] = direct_url
            base, ext = os.path.splitext(fname)
            if ext.lower() in ('.heic', '.heif'):
                drive_links[f"{base}.jpg"] = direct_url

        links_file = os.path.join(event_folder, "gdrive_links.json")
        saved_links = {}
        if os.path.exists(links_file):
            try:
                with open(links_file, "r", encoding="utf-8") as lf:
                    saved_links = json.load(lf)
            except Exception:
                pass
        saved_links.update(drive_links)
        try:
            with open(links_file, "w", encoding="utf-8") as lf:
                json.dump(saved_links, lf, indent=2)
        except Exception as e:
            print(f"[WARN] Could not write gdrive_links.json: {e}")

        import_progress[event_id].update({
            "total_count": total_files,
            "percent": 10,
            "message": f"Found {total_files} photos. Starting download..."
        })

        # 2. Download each photo one by one directly into event_folder (NO SUBFOLDERS!)
        eng = get_engine()
        already_indexed = eng.get_indexed_filenames(event_id) if eng else set()
        total_faces = 0

        for idx, item in enumerate(valid_items):
            fname = os.path.basename(item.path)
            base_name, ext = os.path.splitext(fname)
            check_name = f"{base_name}.jpg" if ext.lower() in ('.heic', '.heif') else fname

            # Target path is directly in event_folder (never nested)
            target_path = os.path.join(event_folder, fname)

            current_pct = int(10 + ((idx) / total_files) * 85)
            import_progress[event_id].update({
                "current_file": fname,
                "downloaded_count": idx,
                "percent": current_pct,
                "message": f"Downloading photo {idx + 1} of {total_files}: {fname}"
            })

            # Download single file if not already on disk
            if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                try:
                    gdown.download(id=item.id, output=target_path, quiet=True)
                except Exception as dl_err:
                    print(f"[WARN] Failed to download {fname}: {dl_err}")
                    continue

            # Index into Qdrant & optimize to web preview
            if eng and check_name not in already_indexed:
                try:
                    dl_url = saved_links.get(fname) or saved_links.get(check_name)
                    faces_found = eng.index_single_image(event_id, target_path, download_url=dl_url)
                    total_faces += faces_found
                    already_indexed.add(check_name)
                    import_progress[event_id]["faces_indexed"] = total_faces
                except Exception as idx_err:
                    print(f"[WARN] Failed to index {fname}: {idx_err}")
            elif eng:
                eng.optimize_to_web_preview(target_path)

        # 3. Clean up any accidental subfolders or macOS metadata files
        for root, dirs, files in os.walk(event_folder, topdown=False):
            if root != event_folder:
                for file in files:
                    if file.startswith('._'):
                        try:
                            os.remove(os.path.join(root, file))
                        except Exception:
                            pass
                    else:
                        src = os.path.join(root, file)
                        dst = os.path.join(event_folder, file)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                try:
                    os.rmdir(root)
                except Exception:
                    pass

        # Update Qdrant download URLs for all photos
        if eng and saved_links:
            eng.update_event_download_urls(event_id, saved_links)

        import_progress[event_id].update({
            "status": "completed",
            "percent": 100,
            "downloaded_count": total_files,
            "faces_indexed": total_faces,
            "message": f"Success! Downloaded {total_files} photos and indexed {total_faces} faces."
        })

    except Exception as e:
        print(f"[ERROR] Import task failed for {event_id}: {e}")
        import_progress[event_id].update({
            "status": "error",
            "error": str(e),
            "message": f"Import error: {str(e)}"
        })

@app.post("/api/events/{event_id}/import-gdrive")
async def import_from_gdrive(
    event_id: str, 
    drive_url: str = Form(...)
):
    """Starts background download and indexing of all photos from Google Drive."""
    eng = get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Face engine is warming up. Please try again in a few seconds.")

    clean_url = drive_url.strip()
    if not clean_url:
        raise HTTPException(status_code=400, detail="Google Drive URL cannot be empty.")

    current = import_progress.get(event_id, {})
    if current.get("status") == "in_progress":
        return {
            "status": "already_running",
            "event_id": event_id,
            "message": "An import task is already running for this event."
        }

    thread = threading.Thread(
        target=run_gdrive_import_task,
        args=(event_id, clean_url),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "event_id": event_id,
        "message": "Google Drive import started in background."
    }

@app.get("/api/events/{event_id}/import-progress")
async def get_import_progress(event_id: str):
    """Returns real-time percentage and status of the Google Drive import."""
    return import_progress.get(event_id, {
        "status": "idle",
        "percent": 0,
        "message": "No active import."
    })

@app.post("/api/events/{event_id}/search")
async def search_faces(
    event_id: str, 
    selfie: UploadFile = File(...),
    threshold: float = Form(0.45)
):
    eng = get_engine()
    if eng is None:
        raise HTTPException(status_code=503, detail="Face engine is warming up. Please try again in a few seconds.")

    temp_path = os.path.join(TEMP_DIR, selfie.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(selfie.file, buffer)

    try:
        matches = eng.search_by_selfie(temp_path, event_id=event_id, similarity_threshold=threshold)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {
        "event_id": event_id,
        "total_matches": len(matches),
        "results": matches
    }