import os
import time
import json
import queue
import threading
from typing import Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class LivePhotoEventHandler(FileSystemEventHandler):
    def __init__(self, watch_queue: queue.Queue, base_dir: str):
        super().__init__()
        self.watch_queue = watch_queue
        self.base_dir = os.path.abspath(base_dir)
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.heic', '.heif')

    def _process_candidate(self, file_path: str):
        # Ignore temporary files created by Google Drive / Windows / Office
        filename = os.path.basename(file_path)
        if filename.startswith(('.', '~', '~$')) or filename.endswith(('.tmp', '.part', '.crdownload')):
            return

        if not filename.lower().endswith(self.valid_exts):
            return

        # Determine event_id from folder path
        rel = os.path.relpath(file_path, self.base_dir)
        parts = rel.split(os.sep)
        if len(parts) >= 2:
            event_id = parts[0]
            self.watch_queue.put((event_id, os.path.abspath(file_path)))

    def on_created(self, event):
        if not event.is_directory:
            self._process_candidate(event.src_path)

    def on_moved(self, event):
        # Google Drive often writes a .tmp file and renames/moves it to final .jpg
        if not event.is_directory:
            self._process_candidate(event.dest_path)


class FolderWatcher:
    def __init__(self, engine, watch_dir: str = "storage/events"):
        self.engine = engine
        self.watch_dir = os.path.abspath(watch_dir)
        os.makedirs(self.watch_dir, exist_ok=True)
        
        self.queue = queue.Queue()
        self.observer: Optional[Observer] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.processed_events_count = 0

    def _wait_for_file_ready(self, file_path: str, timeout: float = 12.0) -> bool:
        """Waits until Google Drive or file transfer finishes writing the file."""
        start_time = time.time()
        last_size = -1

        while time.time() - start_time < timeout:
            if not os.path.exists(file_path):
                return False
            try:
                current_size = os.path.getsize(file_path)
                if current_size > 0 and current_size == last_size:
                    # Test opening file exclusively to confirm write lock is released
                    with open(file_path, "rb"):
                        return True
                last_size = current_size
            except (PermissionError, OSError):
                # File is still locked by Google Drive sync process
                pass
            time.sleep(0.6)

        return os.path.exists(file_path) and os.path.getsize(file_path) > 0

    def _worker_loop(self):
        while self.is_running:
            try:
                event_id, file_path = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Wait for Google Drive sync to finish writing bytes
                if not self._wait_for_file_ready(file_path):
                    continue

                filename = os.path.basename(file_path)
                base_name, ext = os.path.splitext(filename)
                target_name = f"{base_name}.jpg" if ext.lower() in ('.heic', '.heif') else filename

                # Check if already indexed
                already_indexed = self.engine.get_indexed_filenames(event_id)
                if target_name in already_indexed:
                    continue

                print(f"[WATCHER] New photo detected from live sync: {filename} in event '{event_id}'")
                
                # Check if gdrive_links.json has a direct download URL
                download_url = None
                event_folder = os.path.join(self.watch_dir, event_id)
                links_file = os.path.join(event_folder, "gdrive_links.json")
                if os.path.exists(links_file):
                    try:
                        with open(links_file, "r", encoding="utf-8") as lf:
                            glinks = json.load(lf)
                            download_url = glinks.get(filename) or glinks.get(target_name)
                    except Exception:
                        pass

                faces_found = self.engine.index_single_image(event_id, file_path, download_url=download_url)
                self.processed_events_count += 1
                print(f"[WATCHER] Successfully indexed {target_name}: Found {faces_found} face(s). (High-Res link: {'Yes' if download_url else 'Local'})")
            except Exception as e:
                print(f"[WATCHER ERROR] Failed to process {file_path}: {e}")
            finally:
                self.queue.task_done()

    def start(self):
        if self.is_running:
            return

        self.is_running = True
        self.observer = Observer()
        handler = LivePhotoEventHandler(self.queue, self.watch_dir)
        self.observer.schedule(handler, path=self.watch_dir, recursive=True)
        self.observer.start()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print(f"[WATCHER] Live Google Drive / Event Watcher active on: {self.watch_dir}")

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False
        if self.observer:
            try:
                self.observer.stop()
                self.observer.join(timeout=2.0)
            except Exception:
                pass
        print("[WATCHER] Live Event Watcher stopped.")

    def get_status(self) -> dict:
        return {
            "active": self.is_running,
            "monitored_directory": self.watch_dir,
            "queue_size": self.queue.qsize(),
            "photos_processed_since_startup": self.processed_events_count
        }
