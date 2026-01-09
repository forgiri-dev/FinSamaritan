"""
Simple Model Training Script
Uses Keras (simple TensorFlow wrapper) - lightweight and error-free
Generates model files that can be converted to .tflite format
"""
import os
import sys
import json
import numpy as np
from PIL import Image
import keras
from keras import layers
from sklearn.model_selection import train_test_split

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass  # If it fails, continue without emoji support

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50  # Increased for better accuracy
LEARNING_RATE = 0.001
RANDOM_STATE = 42

def load_dataset(data_dir: str):
    """Load images and labels from directory structure"""
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
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"    ⚠️ Error loading {img_file}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\n✅ Loaded {len(images)} images, {len(class_names)} classes")
    
    return images, labels, class_names


def train_model(data_dir: str, output_dir: str = 'models', epochs: int = EPOCHS):
    """
    Train a simple Keras CNN model
    """
    print("🚀 Starting Simple Model Training (Keras)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    images, labels, class_names = load_dataset(data_dir)
    
    num_classes = len(class_names)
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Classes: {', '.join(class_names)}")
    
    # Split dataset (80% train, 20% test)
    print("\n✂️ Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Create improved CNN model using Keras Sequential API
    print("\n🏗️ Creating model (Improved Keras CNN)...")
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Data augmentation (only during training)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Convolutional block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Convolutional block 4
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
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
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with learning rate schedule
    initial_learning_rate = LEARNING_RATE
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"📊 Total parameters: {model.count_params():,}")
    
    # Setup callbacks for better training
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n🎓 Training model...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best model if early stopping was used
    best_model_path = os.path.join(output_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        print("\n📥 Loading best model from training...")
        model = keras.models.load_model(best_model_path)
    
    # Evaluate
    print("\n📊 Evaluating model...")
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Save Keras model
    print("\n💾 Saving model...")
    model_path = os.path.join(output_dir, 'model.keras')
    model.save(model_path)
    print(f"✅ Saved Keras model: {model_path}")
    
    # Save model metadata
    metadata = {
        'num_classes': num_classes,
        'classes': class_names,
        'img_size': IMG_SIZE,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'model_type': 'Keras Sequential CNN'
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {metadata_path}")
    
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
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"📦 Model format: Keras (.keras)")
    print(f"💡 Run convert_to_tflite.py to convert to .tflite format")
    print("=" * 60)
    
    return model, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Simple Model')
    parser.add_argument('--data-dir', type=str, default='training_data',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of training epochs (default: 50)')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
