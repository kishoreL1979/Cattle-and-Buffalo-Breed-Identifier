from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes.predict import router as predict_router

app = FastAPI(
    title="AI Cattle & Buffalo Breed Identification System API",
    description="CNN-based deep learning backend API for classifying 9 cattle and buffalo breeds.",
    version="1.0.0"
)

# Enable CORS for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api", tags=["Prediction"])

@app.get("/")
def root():
    return {
        "status": "online",
        "system": "AI Cattle & Buffalo Breed Identification System API",
        "version": "1.0.0"
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
