# 🗑️ Waste Classification AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Production-ready real-time waste classification system using MobileNetV2 deep learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-endpoints) • [Model](#-model-architecture)

</div>

---

## 🌟 Overview

A comprehensive AI-powered waste classification system designed for smart city and sustainability applications. The system captures live camera input or uploaded images, performs real-time preprocessing, and classifies waste into 10 categories with confidence scores.

### 🎯 Waste Categories

| Category | Emoji | Description |
|----------|-------|-------------|
| Battery | 🔋 | Batteries and electronic waste |
| Biological | 🌱 | Food waste, yard waste, compostables |
| Cardboard | 📦 | Cardboard boxes, packaging |
| Clothes | 👕 | Clothing, fabric, textiles |
| Glass | 🍾 | Bottles, jars, glass containers |
| Metal | 🥫 | Cans, foil, metal containers |
| Paper | 📄 | Newspapers, paper products |
| Plastic | 🥤 | Bottles, containers, plastic packaging |
| Shoes | 👟 | Footwear of all types |
| Trash | 🗑️ | General waste, non-recyclable items |

---

## ✨ Features

### 🤖 Machine Learning
- **MobileNetV2 Transfer Learning** - Lightweight yet powerful CNN architecture
- **Data Augmentation** - Rotation, flipping, brightness, contrast adjustments
- **Class Imbalance Handling** - Class weights and oversampling techniques
- **Model Optimization** - Pruning and quantization for fast inference
- **Real-time Inference** - <100ms latency with TFLite optimization

### 🔧 Backend (FastAPI)
- RESTful API with OpenAPI documentation
- Image upload and base64 encoding support
- API key authentication
- Rate limiting protection
- CORS middleware for cross-origin requests
- Health checks and model versioning

### 🎨 Frontend (Streamlit)
- Modern, responsive UI with custom CSS
- Live camera capture support
- File upload for images
- Real-time confidence visualization
- Disposal guidelines for each category
- Session history tracking

### 📊 MLOps & Monitoring
- Structured logging with Python logging
- Performance metrics tracking
- Model versioning system
- Inference time monitoring
- Error handling and reporting

---

## 📁 Project Structure

```
Waste-Classification/
├── 📂 backend/
│   ├── __init__.py
│   └── main.py                 # FastAPI REST API server
├── 📂 frontend/
│   └── app.py                  # Streamlit web application
├── 📂 config/
│   ├── __init__.py
│   └── settings.py             # Configuration management
├── 📂 utils/
│   ├── __init__.py
│   ├── preprocessing.py        # Image preprocessing pipelines
│   ├── logging_utils.py        # Logging & monitoring utilities
│   └── dataset.py              # Dataset management
├── 📂 notebooks/
│   └── waste_classification_training.ipynb  # ML training pipeline
├── 📂 models/                  # Trained models (generated)
├── 📂 logs/                    # Application logs
├── 📂 data/
│   ├── raw/                    # Raw training data
│   └── processed/              # Processed data
├── 📄 requirements.txt         # Python dependencies
├── 📄 .env.template            # Environment variables template
├── 📄 .gitignore
├── 📄 run_backend.bat          # Backend launcher (Windows)
├── 📄 run_frontend.bat         # Frontend launcher (Windows)
└── 📄 run_all.bat              # Full system launcher (Windows)
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone the Repository
```bash
git clone https://github.com/nirajshevade/Waste-Classification-using-mobilenet.git
cd Waste-Classification
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment
```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings (optional)
```

---

## 💻 Usage

### Option 1: Run Complete System (Recommended)
```bash
# Windows
run_all.bat

# Linux/Mac
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
streamlit run frontend/app.py --server.port 8501
```

### Option 2: Run Components Separately

**Backend API:**
```bash
# Windows
run_backend.bat

# Or manually
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend UI:**
```bash
# Windows
run_frontend.bat

# Or manually
streamlit run frontend/app.py --server.port 8501
```

### Access Points
| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| ReDoc | http://localhost:8000/redoc |

---

## 📓 Training the Model

1. **Prepare Dataset**: Place your waste images in `data/raw/` organized by category:
   ```
   data/raw/
   ├── battery/
   ├── biological/
   ├── cardboard/
   ├── clothes/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   ├── shoes/
   └── trash/
   ```

2. **Run Training Notebook**: Open and execute `notebooks/waste_classification_training.ipynb`

3. **Models Generated**:
   - `models/best_model.keras` - Full Keras model
   - `models/waste_classifier_fp16.tflite` - Optimized TFLite (Float16)
   - `models/waste_classifier_dynamic.tflite` - Dynamic quantization

---

## 🔌 API Endpoints

### Health Check
```http
GET /health
```
Returns API health status and uptime.

### Model Information
```http
GET /model-info
```
Returns model metadata (name, version, classes).

### Classify Image (File Upload)
```http
POST /predict
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: <image_file>
```

### Classify Image (Base64)
```http
POST /predict/base64
Content-Type: application/json
X-API-Key: your-api-key

{
  "image": "base64_encoded_image_string"
}
```

### Response Format
```json
{
  "request_id": "abc123",
  "predictions": {
    "glass": 0.05,
    "metal": 0.02,
    "biological": 0.85,
    "paper": 0.03,
    "plastic": 0.02,
    "battery": 0.01,
    "cardboard": 0.01,
    "clothes": 0.01,
    "shoes": 0.01,
    "trash": 0.01
  },
  "top_prediction": "biological",
  "confidence": 0.85,
  "inference_time_ms": 45.2,
  "timestamp": "2024-01-15T10:30:00Z",
  "disposal_guideline": "🌱 Compost bin or organic waste container."
}
```

---

## 🧠 Model Architecture

### Base Model: MobileNetV2
- Pre-trained on ImageNet (1.4M+ images)
- Lightweight architecture optimized for mobile/edge deployment
- 3.4M parameters with inverted residual blocks

### Custom Classification Head
```
MobileNetV2 (frozen/fine-tuned)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense(10, Softmax) → Output Classes
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.001 (with decay) |
| Epochs | 15 (early stopping, patience=5) |
| Loss Function | Categorical Crossentropy |

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | ~90%+ |
| Inference Time (Keras) | ~50-80ms |
| Inference Time (TFLite) | ~15-40ms |
| Model Size (Keras) | ~15MB |
| Model Size (TFLite FP16) | ~7MB |

*Note: Actual metrics depend on your dataset and hardware.*

---

## 🛠️ Configuration

### Environment Variables (.env)
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-api-key

# Model Paths
MODEL_PATH=models/best_model.keras
TFLITE_PATH=models/waste_classifier_fp16.tflite

# Frontend
STREAMLIT_PORT=8501
API_URL=http://localhost:8000

# Logging
LOG_LEVEL=INFO
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [TensorFlow](https://tensorflow.org/) - ML framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Streamlit](https://streamlit.io/) - ML app framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) - Base architecture

---

<div align="center">

**🌍 Help save the planet by sorting waste correctly!**

Made with ❤️ for a sustainable future

</div>
