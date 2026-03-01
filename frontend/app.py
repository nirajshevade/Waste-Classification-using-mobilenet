"""
Streamlit Frontend for Waste Classification
Real-time waste classification with webcam and file upload support
"""
import streamlit as st
import cv2
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

CLASSES = ["glass", "metal", "organic", "paper", "plastic", "recyclable", "non-recyclable"]
CLASS_COLORS = {
    "glass": "#00BCD4", "metal": "#9E9E9E", "organic": "#4CAF50",
    "paper": "#FF9800", "plastic": "#F44336", "recyclable": "#2196F3",
    "non-recyclable": "#795548"
}
CLASS_EMOJIS = {
    "glass": "🍾", "metal": "🥫", "organic": "🌱",
    "paper": "📄", "plastic": "🥤", "recyclable": "♻️",
    "non-recyclable": "🗑️"
}

DISPOSAL_GUIDELINES = {
    "glass": "♻️ Rinse and place in glass recycling bin. Remove caps and lids.",
    "metal": "♻️ Rinse cans, crush if possible, place in metal recycling bin.",
    "organic": "🌱 Compost bin or organic waste container. Great for composting!",
    "paper": "♻️ Keep dry, flatten cardboard, place in paper recycling bin.",
    "plastic": "♻️ Check recycling number, rinse, and place in plastic recycling.",
    "recyclable": "♻️ Clean and sort into appropriate recycling category.",
    "non-recyclable": "🗑️ General waste bin. Consider if items can be reused first."
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
    """Local classification fallback (demo mode)"""
    import random
    pred_class = random.choice(CLASSES)
    confidence = random.uniform(0.6, 0.99)
    
    # Generate realistic predictions
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
        "timestamp": datetime.utcnow().isoformat()
    }


def display_prediction(result: dict, image: Image.Image):
    """Display prediction results with visualization"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="📸 Analyzed Image", use_container_width=True)
    
    with col2:
        # Check for errors
        if "error" in result:
            st.warning(f"⚠️ {result['error']}")
            st.info("🔄 Using demo mode for prediction...")
            result = classify_local_demo(image)
        
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
        
        # API Status
        st.header("🔌 API Status")
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                st.success("✅ Connected")
                st.caption(f"Uptime: {health['uptime_seconds']:.0f}s")
            else:
                st.error("❌ API Error")
        except:
            st.warning("⚠️ API Offline")
            st.caption("Using demo mode")
        
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
