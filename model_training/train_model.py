"""
Train Edge Sentinel Model for Candlestick Pattern and Trend Detection
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from PIL import Image
import json
from typing import Tuple, List

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def load_labels(labels_path: str) -> List[str]:
    """Load labels from labels.txt file"""
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                labels.append(parts[1])
    return labels

def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load images and labels from directory structure
    
    Directory structure:
    training_data/
        pattern_trend/
            image1.jpg
            image2.jpg
            ...
    """
    images = []
    labels = []
    class_names = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d != '__pycache__']
    class_dirs.sort()
    
    print(f"📁 Found {len(class_dirs)} classes")
    
    for class_idx, class_dir in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_dir)
        class_names.append(class_dir)
        
        # Load images from this class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  📊 {class_dir}: {len(image_files)} images")
        
        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"    ⚠️ Error loading {img_file}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\n✅ Loaded {len(images)} images, {len(class_names)} classes")
    
    return images, labels, class_names

def create_model(num_classes: int) -> keras.Model:
    """
    Create CNN model for pattern and trend classification
    
    Architecture:
    - Input: 224x224x3 RGB images
    - Convolutional layers with batch normalization
    - Max pooling
    - Dropout for regularization
    - Dense layers for classification
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    
    return model

def train_model(
    data_dir: str,
    output_dir: str = 'models',
    validation_split: float = 0.2,
    test_split: float = 0.1
):
    """
    Train the Edge Sentinel model
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save trained model
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
    """
    print("🚀 Starting Edge Sentinel Model Training")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    images, labels, class_names = load_dataset(data_dir)
    
    num_classes = len(class_names)
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Classes: {', '.join(class_names)}")
    
    # Split dataset
    print("\n✂️ Splitting dataset...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=42, stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=validation_split / (1 - test_split), 
        random_state=42, stratify=y_temp
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Data augmentation
    print("\n🔄 Setting up data augmentation...")
    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip candlestick charts
        fill_mode='nearest',
        brightness_range=[0.9, 1.1],
        channel_shift_range=0.1
    )
    
    val_datagen = ImageDataGenerator()  # No augmentation for validation
    
    # Create model
    print("\n🏗️ Creating model architecture...")
    model = create_model(num_classes)
    
    # Print model summary
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    print(f"\n📊 Total parameters: {total_params:,}")
    
    # Callbacks
    checkpoint_path = os.path.join(output_dir, 'best_model.keras')
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🎓 Starting training...")
    print("=" * 60)
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE),
        validation_steps=len(X_val) // BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_loss, test_accuracy, test_top3 = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Top-3 Accuracy: {test_top3:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'edge_sentinel_model.keras')
    model.save(final_model_path)
    print(f"\n💾 Saved final model: {final_model_path}")
    
    # Convert to TensorFlow Lite
    print("\n🔄 Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = os.path.join(output_dir, 'model_unquant.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Saved TFLite model: {tflite_path}")
    print(f"   Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    # Save labels
    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{i} {class_name}\n")
    
    print(f"✅ Saved labels: {labels_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    print(f"✅ Saved training history: {history_path}")
    
    # Save model info
    info_path = os.path.join(output_dir, 'model_info.json')
    model_info = {
        'num_classes': num_classes,
        'classes': class_names,
        'img_size': IMG_SIZE,
        'test_accuracy': float(test_accuracy),
        'test_top3_accuracy': float(test_top3),
        'total_params': total_params,
        'model_size_mb': len(tflite_model) / 1024 / 1024
    }
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Saved model info: {info_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"📁 Models saved to: {output_dir}")
    print(f"📱 TFLite model ready for React Native: {tflite_path}")
    print("=" * 60)
    
    return model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Edge Sentinel Model')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    EPOCHS = args.epochs
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

