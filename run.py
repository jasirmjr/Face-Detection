import uvicorn

if __name__ == "__main__":
    print("🚀 Starting Event Face Finder Server...")
    print("📍 Web Interface: http://127.0.0.1:8000")
    print("📖 API Documentation: http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
