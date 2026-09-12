import uvicorn

if __name__ == "__main__":
    print("[INFO] Starting Event Face Finder Server...")
    print("[INFO] Web Interface: http://127.0.0.1:8000")
    print("[INFO] Admin Portal:  http://127.0.0.1:8000/admin")
    print("[INFO] API Docs:      http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

