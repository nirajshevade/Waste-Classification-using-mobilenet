"""
Streamlit Frontend for Waste Classification
Real-time waste classification with webcam and file upload support
Supports: API backend, local TFLite model, or local Keras model
"""
import streamlit as st
import numpy as np
import requests
import base64
from PIL import Image
import io
import time
from datetime import datetime
import json
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .prediction-card h1 {
        font-size: 2.5rem;
        margin: 0;
        text-transform: uppercase;
    }
    .prediction-card h2 {
        font-size: 1.5rem;
        margin: 10px 0;
        opacity: 0.9;
    }
    .stats-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #e9ecef;
    }
    .disposal-guide {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid #4CAF50;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
        font-size: 1.1rem;
    }
    .confidence-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
    }
    .stProgress > div > div {
        height: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "waste-classifier-api-key-2024")

# Model paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
TFLITE_MODEL_PATH = MODEL_DIR / "waste_classifier_fp16.tflite"
TFLITE_DYNAMIC_PATH = MODEL_DIR / "waste_classifier_dynamic.tflite"
KERAS_MODEL_PATH = MODEL_DIR / "best_model.keras"

CLASSES = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]
CLASS_COLORS = {
    "battery": "#FF5722", "biological": "#4CAF50", "cardboard": "#8D6E63",
    "clothes": "#9C27B0", "glass": "#00BCD4", "metal": "#9E9E9E",
    "paper": "#FF9800", "plastic": "#F44336", "shoes": "#3F51B5",
    "trash": "#795548"
}
CLASS_EMOJIS = {
    "battery": "🔋", "biological": "🌱", "cardboard": "📦",
    "clothes": "👕", "glass": "🍾", "metal": "🥫",
    "paper": "📄", "plastic": "🥤", "shoes": "👟",
    "trash": "🗑️"
}

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


# Session state initialization
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_classified' not in st.session_state:
    st.session_state.total_classified = 0
if 'session_start' not in st.session_state:
    st.session_state.session_start = datetime.now()


def classify_image_api(image: Image.Image) -> dict:
    """Send image to API for classification"""
    try:
        # Convert to bytes
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG", quality=95)
        img_bytes = img_buffer.getvalue()
        
        # Send to API
        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        headers = {"X-API-Key": API_KEY}
        
        response = requests.post(
            f"{API_URL}/predict",
            files=files,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code} - {response.text}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server. Please ensure the backend is running."}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


def classify_local_demo(image: Image.Image) -> dict:
    """Local classification using TFLite or Keras model, with random fallback"""
    
    # Try loading model (cached)
    model, model_type = load_local_model()
    
    if model is not None:
        return _run_local_inference(model, model_type, image)
    
    # Final fallback: random demo mode
    import random
    pred_class = random.choice(CLASSES)
    confidence = random.uniform(0.6, 0.99)
    
    predictions = {}
    remaining = 1.0 - confidence
    for c in CLASSES:
        if c == pred_class:
            predictions[c] = confidence
        else:
            share = random.uniform(0, remaining / (len(CLASSES) - 1))
            predictions[c] = share
            remaining -= share
    
    return {
        "request_id": f"demo_{random.randint(1000, 9999)}",
        "top_prediction": pred_class,
        "confidence": confidence,
        "predictions": predictions,
        "inference_time_ms": random.uniform(10, 50),
        "disposal_guideline": DISPOSAL_GUIDELINES[pred_class],
        "timestamp": datetime.utcnow().isoformat(),
        "mode": "demo"
    }


@st.cache_resource
def load_local_model():
    """Load a local model (TFLite preferred, then Keras). Cached across reruns."""
    
    # Try TFLite models first (lightweight, fast)
    for tflite_path in [TFLITE_MODEL_PATH, TFLITE_DYNAMIC_PATH]:
        if tflite_path.exists():
            try:
                # Try lightweight runtimes first, fallback to full tensorflow
                interpreter = None
                try:
                    from tflite_runtime.interpreter import Interpreter
                    interpreter = Interpreter(model_path=str(tflite_path))
                except ImportError:
                    pass
                
                if interpreter is None:
                    try:
                        from ai_edge_litert.interpreter import Interpreter
                        interpreter = Interpreter(model_path=str(tflite_path))
                    except ImportError:
                        pass
                
                if interpreter is None:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                
                interpreter.allocate_tensors()
                return interpreter, "tflite"
            except Exception as e:
                st.warning(f"Failed to load TFLite model: {e}")
    
    # Try Keras model
    if KERAS_MODEL_PATH.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(KERAS_MODEL_PATH))
            return model, "keras"
        except Exception as e:
            st.warning(f"Failed to load Keras model: {e}")
    
    return None, None


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for model input (224x224, normalized)"""
    img = image.convert("RGB").resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def _run_local_inference(model, model_type: str, image: Image.Image) -> dict:
    """Run inference using a local TFLite or Keras model"""
    import random
    
    start_time = time.time()
    img_input = _preprocess_image(image)
    
    try:
        if model_type == "tflite":
            input_details = model.get_input_details()
            output_details = model.get_output_details()
            
            # Ensure input dtype matches model expectation
            input_dtype = input_details[0]['dtype']
            img_input = img_input.astype(input_dtype)
            
            model.set_tensor(input_details[0]['index'], img_input)
            model.invoke()
            output = model.get_tensor(output_details[0]['index'])[0]
        else:  # keras
            output = model.predict(img_input, verbose=0)[0]
        
        inference_time = (time.time() - start_time) * 1000
        
        # Map predictions to class names
        predictions = {}
        for i, cls in enumerate(CLASSES):
            if i < len(output):
                predictions[cls] = float(output[i])
            else:
                predictions[cls] = 0.0
        
        # Normalize predictions to sum to 1
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v / total for k, v in predictions.items()}
        
        pred_class = max(predictions, key=predictions.get)
        confidence = predictions[pred_class]
        
        return {
            "request_id": f"local_{random.randint(1000, 9999)}",
            "top_prediction": pred_class,
            "confidence": confidence,
            "predictions": predictions,
            "inference_time_ms": inference_time,
            "disposal_guideline": DISPOSAL_GUIDELINES.get(pred_class, "🗑️ Check local guidelines."),
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "local_model"
        }
    except Exception as e:
        return {"error": f"Model inference failed: {str(e)}"}


def display_prediction(result: dict, image: Image.Image):
    """Display prediction results with visualization"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="📸 Analyzed Image", use_container_width=True)
    
    with col2:
        # Check for errors
        if "error" in result:
            st.info("🔄 API unavailable — using local inference...")
            result = classify_local_demo(image)
        
        # Show inference mode badge
        mode = result.get("mode", "api")
        if mode == "demo":
            st.caption("⚡ Demo mode — train & add a model for real predictions")
        elif mode == "local_model":
            st.caption("🧠 Running on local model")
        
        pred_class = result.get("top_prediction", "unknown")
        confidence = result.get("confidence", 0)
        inference_time = result.get("inference_time_ms", 0)
        
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card">
            <h1>{CLASS_EMOJIS.get(pred_class, '🗑️')} {pred_class}</h1>
            <h2>{confidence*100:.1f}% Confidence</h2>
            <p style="opacity: 0.8;">⏱️ {inference_time:.1f}ms inference time</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Disposal guideline
        st.markdown(f"""
        <div class="disposal-guide">
            <strong>📋 Disposal Guideline:</strong><br>
            {DISPOSAL_GUIDELINES.get(pred_class, 'N/A')}
        </div>
        """, unsafe_allow_html=True)
    
    # All predictions visualization
    st.subheader("📊 Confidence Scores")
    predictions = result.get("predictions", {})
    
    # Sort by confidence
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    for cls, prob in sorted_preds:
        col_emoji, col_name, col_bar, col_val = st.columns([0.5, 1.5, 5, 1])
        with col_emoji:
            st.write(CLASS_EMOJIS.get(cls, ''))
        with col_name:
            st.write(cls.capitalize())
        with col_bar:
            st.progress(prob)
        with col_val:
            st.write(f"{prob*100:.1f}%")


def main():
    # Header
    st.markdown('<h1 class="main-header">♻️ Waste Classification AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Smart Waste Sorting Powered by MobileNetV2 Deep Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        input_method = st.radio(
            "📥 Input Method",
            ["📷 Camera Capture", "📁 Upload Image"],
            index=1
        )
        
        st.divider()
        
        confidence_threshold = st.slider(
            "🎯 Confidence Threshold",
            0.0, 1.0, 0.5, 0.05,
            help="Minimum confidence level to accept a prediction"
        )
        
        st.divider()
        
        # API Status & Inference Mode
        st.header("🔌 API Status")
        api_available = False
        try:
            response = requests.get(f"{API_URL}/health", timeout=3)
            if response.status_code == 200:
                health = response.json()
                st.success("✅ Connected")
                st.caption(f"Uptime: {health['uptime_seconds']:.0f}s")
                api_available = True
            else:
                st.error("❌ API Error")
        except:
            # Check for local model
            model, model_type = load_local_model()
            if model is not None:
                st.info(f"🧠 Using local {model_type.upper()} model")
                st.caption("Running predictions locally")
            else:
                st.warning("⚠️ Demo mode")
                st.caption("No API or model available")
        
        st.divider()
        
        # Statistics
        st.header("📈 Session Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Classified", st.session_state.total_classified)
        with col2:
            duration = (datetime.now() - st.session_state.session_start).seconds
            st.metric("Duration", f"{duration//60}m {duration%60}s")
        
        # History
        if st.session_state.history:
            st.header("📜 Recent History")
            for item in st.session_state.history[-5:][::-1]:
                emoji = CLASS_EMOJIS.get(item['class'], '🗑️')
                st.write(f"{emoji} **{item['class']}** - {item['confidence']:.0%}")
        
        st.divider()
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.total_classified = 0
            st.rerun()
    
    # Main content area
    st.divider()
    
    if "📷 Camera" in input_method:
        st.subheader("📷 Camera Capture")
        st.info("📌 Position the waste item clearly in the camera frame and capture")
        
        camera_image = st.camera_input(
            "Take a photo of the waste item",
            help="Click to capture an image"
        )
        
        if camera_image:
            image = Image.open(camera_image)
            
            with st.spinner("🔄 Analyzing image..."):
                result = classify_image_api(image)
            
            display_prediction(result, image)
            
            # Update session state
            if "error" not in result:
                st.session_state.total_classified += 1
                st.session_state.history.append({
                    'class': result.get('top_prediction', 'unknown'),
                    'confidence': result.get('confidence', 0),
                    'timestamp': datetime.now().isoformat()
                })
    
    else:
        st.subheader("📁 Upload Image")
        st.info("📌 Upload an image of a waste item to classify it")
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, JPEG, PNG, WebP"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            with st.spinner("🔄 Analyzing image..."):
                result = classify_image_api(image)
            
            display_prediction(result, image)
            
            # Update session state
            if "error" not in result:
                st.session_state.total_classified += 1
                st.session_state.history.append({
                    'class': result.get('top_prediction', 'unknown'),
                    'confidence': result.get('confidence', 0),
                    'timestamp': datetime.now().isoformat()
                })
    
    # Information section
    st.divider()
    
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### 🗑️ Waste Classification AI
        
        This application uses a **MobileNetV2** deep learning model to classify waste items into 7 categories:
        
        | Category | Description | Emoji |
        |----------|-------------|-------|
        | Glass | Bottles, jars, containers | 🍾 |
        | Metal | Cans, foil, metal containers | 🥫 |
        | Organic | Food waste, yard waste | 🌱 |
        | Paper | Newspapers, cardboard, paper | 📄 |
        | Plastic | Bottles, containers, packaging | 🥤 |
        | Recyclable | Mixed recyclable materials | ♻️ |
        | Non-recyclable | General waste | 🗑️ |
        
        ### 🎯 Features
        - Real-time classification (<100ms inference)
        - Camera and file upload support
        - Confidence scores for all categories
        - Disposal guidelines for proper sorting
        
        ### 🌍 Help Save the Planet!
        Proper waste sorting is crucial for:
        - Reducing landfill waste
        - Increasing recycling rates
        - Conserving natural resources
        - Protecting the environment
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🌍 <strong>Help save the planet by sorting waste correctly!</strong></p>
        <p style="font-size: 0.9rem;">Built with ❤️ using TensorFlow, FastAPI, and Streamlit</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">© 2024 Waste Classification AI | Version 1.0.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
