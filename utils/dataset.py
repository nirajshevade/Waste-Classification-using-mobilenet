"""
Dataset management utilities for Waste Classification
"""
import os
import shutil
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime

class DatasetManager:
    """Manage datasets for waste classification training"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_file = self.data_dir / "dataset_metadata.json"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dataset metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "created_at": datetime.utcnow().isoformat(),
                "total_images": 0,
                "class_distribution": {},
                "splits": {},
                "version": "1.0.0"
            }
    
    def _save_metadata(self):
        """Save dataset metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def scan_dataset(self, classes: List[str]) -> Dict[str, int]:
        """
        Scan dataset directory and count images per class
        
        Args:
            classes: List of class names (folder names)
            
        Returns:
            Dictionary mapping class names to image counts
        """
        distribution = {}
        total = 0
        
        for class_name in classes:
            class_dir = self.raw_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + \
                        list(class_dir.glob("*.jpeg")) + \
                        list(class_dir.glob("*.png"))
                distribution[class_name] = len(images)
                total += len(images)
            else:
                distribution[class_name] = 0
        
        self.metadata["class_distribution"] = distribution
        self.metadata["total_images"] = total
        self.metadata["last_scan"] = datetime.utcnow().isoformat()
        self._save_metadata()
        
        return distribution
    
    def get_class_weights(self, classes: List[str]) -> Dict[int, float]:
        """
        Calculate class weights for handling imbalanced data
        
        Args:
            classes: List of class names
            
        Returns:
            Dictionary mapping class indices to weights
        """
        distribution = self.metadata.get("class_distribution", {})
        if not distribution:
            distribution = self.scan_dataset(classes)
        
        total_samples = sum(distribution.values())
        n_classes = len(classes)
        
        weights = {}
        for idx, class_name in enumerate(classes):
            count = distribution.get(class_name, 1)
            # Balanced class weight formula
            weights[idx] = total_samples / (n_classes * count) if count > 0 else 1.0
        
        return weights
    
    def create_train_val_test_split(self, 
                                     classes: List[str],
                                     val_ratio: float = 0.15,
                                     test_ratio: float = 0.15,
                                     random_state: int = 42) -> Dict[str, List[str]]:
        """
        Create train/validation/test splits
        
        Args:
            classes: List of class names
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, test file paths
        """
        all_files = []
        all_labels = []
        
        for idx, class_name in enumerate(classes):
            class_dir = self.raw_dir / class_name
            if class_dir.exists():
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for img_path in class_dir.glob(ext):
                        all_files.append(str(img_path))
                        all_labels.append(idx)
        
        # First split: train + temp
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            all_files, all_labels,
            test_size=val_ratio + test_ratio,
            stratify=all_labels,
            random_state=random_state
        )
        
        # Second split: val + test
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels,
            test_size=relative_test_ratio,
            stratify=temp_labels,
            random_state=random_state
        )
        
        splits = {
            "train": {"files": train_files, "labels": train_labels},
            "val": {"files": val_files, "labels": val_labels},
            "test": {"files": test_files, "labels": test_labels}
        }
        
        self.metadata["splits"] = {
            "train_count": len(train_files),
            "val_count": len(val_files),
            "test_count": len(test_files),
            "created_at": datetime.utcnow().isoformat()
        }
        self._save_metadata()
        
        return splits
    
    def create_tf_dataset(self,
                          file_paths: List[str],
                          labels: List[int],
                          batch_size: int = 32,
                          image_size: Tuple[int, int] = (224, 224),
                          augment: bool = False,
                          shuffle: bool = True) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from file paths
        
        Args:
            file_paths: List of image file paths
            labels: List of corresponding labels
            batch_size: Batch size
            image_size: Target image size
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle the dataset
            
        Returns:
            TensorFlow Dataset
        """
        def load_and_preprocess(file_path, label):
            # Read image
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, image_size)
            
            # Preprocess for MobileNetV2
            img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
            
            return img, label
        
        def augment_image(img, label):
            # Random flip
            img = tf.image.random_flip_left_right(img)
            # Random brightness
            img = tf.image.random_brightness(img, 0.2)
            # Random contrast
            img = tf.image.random_contrast(img, 0.8, 1.2)
            # Random saturation
            img = tf.image.random_saturation(img, 0.8, 1.2)
            
            return img, label
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(file_paths))
        
        # Parallel loading
        dataset = dataset.map(
            load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if augment:
            dataset = dataset.map(
                augment_image,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_sample_images(self, classes: List[str], n_per_class: int = 5) -> Dict[str, List[str]]:
        """Get sample images from each class"""
        samples = {}
        for class_name in classes:
            class_dir = self.raw_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))[:n_per_class]
                samples[class_name] = [str(p) for p in images]
            else:
                samples[class_name] = []
        return samples


class OverSampler:
    """Handle class imbalance through oversampling"""
    
    @staticmethod
    def oversample_minority_classes(file_paths: List[str], 
                                    labels: List[int],
                                    strategy: str = "balance") -> Tuple[List[str], List[int]]:
        """
        Oversample minority classes to balance the dataset
        
        Args:
            file_paths: List of image file paths
            labels: List of corresponding labels
            strategy: 'balance' for equal distribution, 'moderate' for partial balancing
            
        Returns:
            Tuple of (oversampled_files, oversampled_labels)
        """
        label_counts = Counter(labels)
        max_count = max(label_counts.values())
        
        if strategy == "moderate":
            target_count = int(max_count * 0.75)
        else:
            target_count = max_count
        
        oversampled_files = []
        oversampled_labels = []
        
        for label in set(labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            current_count = len(indices)
            
            # Add all original samples
            for idx in indices:
                oversampled_files.append(file_paths[idx])
                oversampled_labels.append(labels[idx])
            
            # Oversample if needed
            if current_count < target_count:
                additional_needed = target_count - current_count
                additional_indices = np.random.choice(indices, size=additional_needed, replace=True)
                
                for idx in additional_indices:
                    oversampled_files.append(file_paths[idx])
                    oversampled_labels.append(labels[idx])
        
        return oversampled_files, oversampled_labels


def compute_dataset_hash(data_dir: str) -> str:
    """Compute hash of dataset for versioning"""
    hasher = hashlib.md5()
    
    data_path = Path(data_dir)
    for file_path in sorted(data_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            hasher.update(str(file_path).encode())
            hasher.update(str(file_path.stat().st_size).encode())
    
    return hasher.hexdigest()[:12]
