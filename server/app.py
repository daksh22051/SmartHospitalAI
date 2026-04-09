from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# 1. Static files (JS/CSS) serve karne ke liye
# Check kar ki tera build folder 'dist' hai ya 'build'
dist_path = os.path.join(os.getcwd(), "dist")

if os.path.exists(dist_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

# 2. CATCH-ALL ROUTE: Ye sabse important hai
@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Agar file exist karti hai (jaise logo.png), toh wo dikhao
    file_path = os.path.join(dist_path, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Nahi toh hamesha index.html bhejo (React Router handle kar lega)
    return FileResponse(os.path.join(dist_path, "index.html"))