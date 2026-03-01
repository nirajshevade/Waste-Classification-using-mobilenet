"""
Image preprocessing utilities for Waste Classification
"""
import cv2
import numpy as np
from PIL import Image
import io
import base64
from typing import Union, Tuple, Optional
import tensorflow as tf

class ImagePreprocessor:
    """Production-ready image preprocessing pipeline"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, bytes, str]) -> np.ndarray:
        """
        Preprocess image for model inference
        
        Args:
            image: Input image (numpy array, PIL Image, bytes, or base64 string)
            
        Returns:
            Preprocessed numpy array ready for model inference
        """
        # Convert to numpy array if needed
        if isinstance(image, bytes):
            image = self._bytes_to_array(image)
        elif isinstance(image, str):
            image = self._base64_to_array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3 and self._is_bgr(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize using MobileNetV2 preprocessing
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image.astype(np.float32))
        
        return image
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """Preprocess a batch of images"""
        processed = [self.preprocess_image(img) for img in images]
        return np.array(processed)
    
    def _bytes_to_array(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to numpy array"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def _base64_to_array(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to numpy array"""
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_bytes = base64.b64decode(base64_string)
        return self._bytes_to_array(image_bytes)
    
    def _is_bgr(self, image: np.ndarray) -> bool:
        """Heuristic to detect if image is BGR (from OpenCV)"""
        # This is a simple heuristic, may not always be accurate
        return False  # Assume RGB by default
    
    def array_to_base64(self, image: np.ndarray, format: str = "JPEG") -> str:
        """Convert numpy array to base64 string"""
        pil_image = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        return base64.b64encode(buffer.getvalue()).decode()
    
    @staticmethod
    def validate_image(image_bytes: bytes, max_size: int = 10 * 1024 * 1024) -> Tuple[bool, Optional[str]]:
        """
        Validate image bytes
        
        Args:
            image_bytes: Raw image bytes
            max_size: Maximum allowed file size in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(image_bytes) > max_size:
            return False, f"Image size exceeds maximum allowed size of {max_size // (1024*1024)}MB"
        
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                return False, "Invalid image format or corrupted file"
            return True, None
        except Exception as e:
            return False, f"Error validating image: {str(e)}"


class RealTimePreprocessor:
    """Optimized preprocessor for real-time camera feed"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self._resize_cache = {}
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Fast preprocessing for real-time camera frames
        Optimized for <10ms processing time
        """
        # Convert BGR to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Fast resize using INTER_LINEAR for speed
        frame_resized = cv2.resize(frame_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # MobileNetV2 preprocessing
        frame_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(
            frame_resized.astype(np.float32)
        )
        
        return frame_normalized
    
    def process_frame_batch(self, frames: list) -> np.ndarray:
        """Process multiple frames efficiently"""
        return np.array([self.process_frame(f) for f in frames])


def apply_data_augmentation(image: np.ndarray, training: bool = True) -> np.ndarray:
    """
    Apply data augmentation for training
    
    Args:
        image: Input image (H, W, C)
        training: Whether to apply augmentation
        
    Returns:
        Augmented image
    """
    if not training:
        return image
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        image = np.fliplr(image)
    
    # Random rotation (-20 to 20 degrees)
    angle = np.random.uniform(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h))
    
    # Random brightness adjustment
    brightness = np.random.uniform(0.8, 1.2)
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    # Random contrast adjustment
    contrast = np.random.uniform(0.8, 1.2)
    mean = np.mean(image)
    image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
    
    return image
