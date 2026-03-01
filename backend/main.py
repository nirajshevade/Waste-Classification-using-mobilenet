"""
FastAPI Backend for Waste Classification API
Production-ready with authentication, logging, and monitoring
"""
import os
import io
import time
import uuid
import base64
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from functools import lru_cache

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf

# Configure logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / 'api.log')
    ]
)
logger = logging.getLogger(__name__)


# Configuration
class Settings:
    BASE_DIR = Path(__file__).parent.parent
    MODEL_PATH: str = str(BASE_DIR / "models" / "best_model.keras")
    TFLITE_PATH: str = str(BASE_DIR / "models" / "waste_classifier_fp16.tflite")
    API_KEY: str = os.getenv("API_KEY", "waste-classifier-api-key-2024")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".webp"]
    CLASSES: List[str] = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
    IMG_SIZE: tuple = (224, 224)


settings = Settings()


# Initialize FastAPI app
app = FastAPI(
    title="Waste Classification API",
    description="Real-time waste classification using MobileNetV2",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting storage (in production, use Redis)
request_counts: Dict[str, List[float]] = {}
RATE_LIMIT = 100  # requests per minute


# Pydantic models
class PredictionResponse(BaseModel):
    request_id: str
    predictions: Dict[str, float]
    top_prediction: str
    confidence: float
    inference_time_ms: float
    timestamp: str
    disposal_guideline: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    classes: List[str]
    input_shape: tuple
    framework: str


class Base64ImageRequest(BaseModel):
    image: str


# Disposal guidelines
DISPOSAL_GUIDELINES = {
    "battery": "🔋 Take to a battery recycling drop-off point. Never throw in regular trash!",
    "biological": "🌱 Compost bin or organic waste container. Great for composting!",
    "cardboard": "♻️ Flatten and place in paper/cardboard recycling bin. Keep dry.",
    "clothes": "👕 Donate if wearable, or take to a textile recycling bin.",
    "glass": "♻️ Rinse and place in glass recycling bin. Remove caps and lids.",
    "metal": "♻️ Rinse cans, crush if possible, place in metal recycling bin.",
    "paper": "♻️ Keep dry, flatten, and place in paper recycling bin.",
    "plastic": "♻️ Check recycling number, rinse, and place in plastic recycling.",
    "shoes": "👟 Donate if wearable, or take to a textile/shoe recycling point.",
    "trash": "🗑️ General waste bin. Consider if items can be reused or recycled first."
}

# Global model variable
model = None
interpreter = None
start_time = time.time()


def load_model():
    """Load the classification model"""
    global model, interpreter
    try:
        # Try TFLite first for faster inference
        if os.path.exists(settings.TFLITE_PATH):
            interpreter = tf.lite.Interpreter(model_path=settings.TFLITE_PATH)
            interpreter.allocate_tensors()
            logger.info(f"TFLite model loaded from {settings.TFLITE_PATH}")
            return True
        # Fall back to Keras model
        elif os.path.exists(settings.MODEL_PATH):
            model = tf.keras.models.load_model(settings.MODEL_PATH)
            logger.info(f"Keras model loaded from {settings.MODEL_PATH}")
            return True
        else:
            logger.warning("No model file found. Running in demo mode.")
            return False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def verify_api_key(x_api_key: str = Header(default=None)):
    """Verify API key (optional for development)"""
    if x_api_key is None:
        return None  # Allow requests without API key in development
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def rate_limit(request: Request):
    """Simple rate limiting"""
    client_ip = request.client.host
    current_time = time.time()
    
    if client_ip not in request_counts:
        request_counts[client_ip] = []
    
    # Remove old requests
    request_counts[client_ip] = [t for t in request_counts[client_ip] if current_time - t < 60]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_counts[client_ip].append(current_time)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model inference"""
    image = image.convert("RGB")
    image = image.resize(settings.IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)


def predict(img_array: np.ndarray) -> np.ndarray:
    """Run inference"""
    global model, interpreter
    
    if interpreter is not None:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]["index"])
    elif model is not None:
        return model.predict(img_array, verbose=0)
    else:
        # Demo mode - return random predictions
        logger.warning("Running in demo mode - no model loaded")
        predictions = np.random.dirichlet(np.ones(len(settings.CLASSES)))
        return np.array([predictions])


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Waste Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None or interpreter is not None,
        uptime_seconds=time.time() - start_time,
        version="1.0.0"
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    return ModelInfoResponse(
        name="WasteClassifier_MobileNetV2",
        version="1.0.0",
        classes=settings.CLASSES,
        input_shape=(224, 224, 3),
        framework="TensorFlow/Keras"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    """Classify waste image"""
    rate_limit(request)
    request_id = str(uuid.uuid4())[:8]
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )
    
    try:
        # Read and preprocess image
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        image = Image.open(io.BytesIO(contents))
        img_array = preprocess_image(image)
        
        # Run inference
        inference_start = time.perf_counter()
        predictions = predict(img_array)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Process results
        pred_dict = {cls: float(prob) for cls, prob in zip(settings.CLASSES, predictions[0])}
        top_class = max(pred_dict, key=pred_dict.get)
        confidence = pred_dict[top_class]
        
        logger.info(f"Request {request_id}: {top_class} ({confidence:.2%}) in {inference_time:.2f}ms")
        
        return PredictionResponse(
            request_id=request_id,
            predictions=pred_dict,
            top_prediction=top_class,
            confidence=confidence,
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.utcnow().isoformat(),
            disposal_guideline=DISPOSAL_GUIDELINES.get(top_class, "")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/base64", response_model=PredictionResponse)
async def predict_base64(
    request: Request,
    image_data: Base64ImageRequest,
    api_key: str = Depends(verify_api_key)
):
    """Classify waste image from base64 string"""
    rate_limit(request)
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # Decode base64 image
        base64_string = image_data.image
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        image_bytes = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_bytes))
        img_array = preprocess_image(image)
        
        # Run inference
        inference_start = time.perf_counter()
        predictions = predict(img_array)
        inference_time = (time.perf_counter() - inference_start) * 1000
        
        # Process results
        pred_dict = {cls: float(prob) for cls, prob in zip(settings.CLASSES, predictions[0])}
        top_class = max(pred_dict, key=pred_dict.get)
        confidence = pred_dict[top_class]
        
        return PredictionResponse(
            request_id=request_id,
            predictions=pred_dict,
            top_prediction=top_class,
            confidence=confidence,
            inference_time_ms=round(inference_time, 2),
            timestamp=datetime.utcnow().isoformat(),
            disposal_guideline=DISPOSAL_GUIDELINES.get(top_class, "")
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
