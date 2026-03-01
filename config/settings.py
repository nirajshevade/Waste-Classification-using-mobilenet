"""
Configuration settings for Waste Classification System
"""
import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

@dataclass
class ModelConfig:
    """Model configuration settings"""
    name: str = "WasteClassifier_MobileNetV2"
    version: str = "1.0.0"
    input_shape: tuple = (224, 224, 3)
    num_classes: int = 7
    classes: List[str] = field(default_factory=lambda: [
        "glass", "metal", "organic", "paper", "plastic", "recyclable", "non-recyclable"
    ])
    backbone: str = "MobileNetV2"
    weights: str = "imagenet"
    trainable_layers: int = 30
    dropout_rate: float = 0.3
    l2_regularization: float = 0.01

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    min_learning_rate: float = 1e-7
    patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Data augmentation settings
    rotation_range: int = 40
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    vertical_flip: bool = False
    brightness_range: tuple = (0.8, 1.2)
    
    # Class weights for imbalance handling
    use_class_weights: bool = True
    use_oversampling: bool = True

@dataclass
class OptimizationConfig:
    """Model optimization settings"""
    enable_quantization: bool = True
    quantization_type: str = "float16"  # float16, int8, dynamic
    enable_pruning: bool = True
    pruning_schedule: str = "polynomial"
    initial_sparsity: float = 0.0
    final_sparsity: float = 0.5
    pruning_frequency: int = 100

@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    api_version: str = "v1"
    api_prefix: str = "/api/v1"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 100  # requests per minute
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    
    # Security
    api_key_header: str = "X-API-Key"
    enable_api_key: bool = True
    jwt_secret: str = os.getenv("JWT_SECRET", "waste-classifier-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    token_expiry_hours: int = 24

@dataclass
class MonitoringConfig:
    """Performance monitoring settings"""
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    inference_timeout_ms: int = 100
    health_check_interval: int = 30

@dataclass
class FrontendConfig:
    """Frontend configuration settings"""
    streamlit_port: int = 8501
    camera_resolution: tuple = (640, 480)
    frame_rate: int = 30
    confidence_threshold: float = 0.5
    display_top_k: int = 3

# Global configuration instances
model_config = ModelConfig()
training_config = TrainingConfig()
optimization_config = OptimizationConfig()
api_config = APIConfig()
monitoring_config = MonitoringConfig()
frontend_config = FrontendConfig()

# Class color mapping for UI
CLASS_COLORS: Dict[str, str] = {
    "glass": "#00BCD4",
    "metal": "#9E9E9E",
    "organic": "#4CAF50",
    "paper": "#FF9800",
    "plastic": "#F44336",
    "recyclable": "#2196F3",
    "non-recyclable": "#795548"
}

# Disposal guidelines
DISPOSAL_GUIDELINES: Dict[str, str] = {
    "glass": "♻️ Rinse and place in glass recycling bin. Remove caps and lids.",
    "metal": "♻️ Rinse cans, crush if possible, place in metal recycling bin.",
    "organic": "🌱 Compost bin or organic waste container. Great for composting!",
    "paper": "♻️ Keep dry, flatten cardboard, place in paper recycling bin.",
    "plastic": "♻️ Check recycling number, rinse, and place in plastic recycling.",
    "recyclable": "♻️ Clean and sort into appropriate recycling category.",
    "non-recyclable": "🗑️ General waste bin. Consider if items can be reused first."
}
