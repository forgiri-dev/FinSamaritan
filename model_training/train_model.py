"""
Train Edge Sentinel Model for Candlestick Pattern and Trend Detection
TensorFlow implementation (no Keras)
"""
import os
import numpy as np
import json
import tensorflow as tf
from typing import Tuple, List
from PIL import Image

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Check TensorFlow installation and get core APIs
def get_tf_variable():
    """Get tf.Variable, trying multiple locations"""
    if hasattr(tf, 'Variable'):
        return tf.Variable
    elif hasattr(tf, 'compat') and hasattr(tf.compat, 'v1') and hasattr(tf.compat.v1, 'Variable'):
        return tf.compat.v1.Variable
    else:
        raise AttributeError(
            "TensorFlow Variable not found. Your TensorFlow installation appears to be broken.\n"
            "Please reinstall TensorFlow:\n"
            "  pip uninstall tensorflow tensorflow-cpu tensorflow-gpu\n"
            "  pip install tensorflow==2.15.1"
        )

def get_tf_zeros():
    """Get tf.zeros function"""
    if hasattr(tf, 'zeros'):
        return tf.zeros
    elif hasattr(tf, 'compat') and hasattr(tf.compat, 'v1') and hasattr(tf.compat.v1, 'zeros'):
        return tf.compat.v1.zeros
    else:
        raise AttributeError("TensorFlow zeros function not found. Please reinstall TensorFlow.")

def get_tf_constant():
    """Get tf.constant function"""
    if hasattr(tf, 'constant'):
        return tf.constant
    elif hasattr(tf, 'compat') and hasattr(tf.compat, 'v1') and hasattr(tf.compat.v1, 'constant'):
        return tf.compat.v1.constant
    else:
        raise AttributeError("TensorFlow constant function not found. Please reinstall TensorFlow.")

# Get TensorFlow core functions
try:
    TF_Variable = get_tf_variable()
    TF_zeros = get_tf_zeros()
    TF_constant = get_tf_constant()
except AttributeError as e:
    print(f"❌ ERROR: {e}")
    print(f"\nTensorFlow version: {tf.__version__ if hasattr(tf, '__version__') else 'Unknown'}")
    print(f"TensorFlow location: {tf.__file__ if hasattr(tf, '__file__') else 'Unknown'}")
    print("\nPlease check your TensorFlow installation.")
    raise

# Get TensorFlow dtypes safely (fallback to string if attributes don't exist)
try:
    TF_FLOAT32 = tf.float32
except AttributeError:
    try:
        TF_FLOAT32 = tf.dtypes.float32
    except AttributeError:
        TF_FLOAT32 = 'float32'  # Use string dtype

try:
    TF_INT32 = tf.int32
except AttributeError:
    try:
        TF_INT32 = tf.dtypes.int32
    except AttributeError:
        TF_INT32 = 'int32'  # Use string dtype

# Set random seeds for reproducibility
np.random.seed(42)
# Set TensorFlow random seed if available
if hasattr(tf, 'random') and hasattr(tf.random, 'set_seed'):
    tf.random.set_seed(42)
elif hasattr(tf, 'set_random_seed'):
    # Fallback for older TensorFlow versions
    tf.set_random_seed(42)


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int = 42,
):
    """
    Simple replacement for sklearn.model_selection.train_test_split with stratification.

    Args:
        X: Data array (num_samples, ...)
        y: Integer labels (num_samples,)
        test_size: Fraction of samples to put in the test set (0 < test_size < 1)
        random_state: Seed for reproducibility

    Returns:
        X_train, X_test, y_train, y_test
    """
    assert 0.0 < test_size < 1.0, "test_size must be between 0 and 1"

    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    X = np.asarray(X)

    train_indices = []
    test_indices = []

    classes = np.unique(y)
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)

        n_test = max(1, int(len(cls_indices) * test_size)) if len(cls_indices) > 1 else 0
        if n_test == 0:
            # If there is only a single sample for this class, keep it in train
            train_indices.extend(cls_indices.tolist())
        else:
            test_indices.extend(cls_indices[:n_test].tolist())
            train_indices.extend(cls_indices[n_test:].tolist())

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)

    # Shuffle final indices to remove class ordering
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices],
    )


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
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
                
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"    ⚠️ Error loading {img_file}: {e}")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"\n✅ Loaded {len(images)} images, {len(class_names)} classes")
    
    return images, labels, class_names


# ============================================================================
# TensorFlow CNN Model (no Keras)
# ============================================================================

class EdgeSentinelModel:
    """
    Simple CNN implemented with TensorFlow (no Keras).
    """

    def __init__(self, num_classes: int, name: str = "edge_sentinel_model"):
        self.name = name
        self.num_classes = num_classes
        
        # Initialize weights using He initialization
        # Check if tf.initializers exists and has HeNormal
        if hasattr(tf, 'initializers') and hasattr(tf.initializers, 'HeNormal'):
            initializer_fn = tf.initializers.HeNormal(seed=42)
        elif hasattr(tf, 'keras') and hasattr(tf.keras, 'initializers') and hasattr(tf.keras.initializers, 'HeNormal'):
            # Fallback to Keras initializers if available
            initializer_fn = tf.keras.initializers.HeNormal(seed=42)
        else:
            # Fallback to glorot_uniform if HeNormal not available
            if hasattr(tf, 'initializers') and hasattr(tf.initializers, 'glorot_uniform'):
                initializer_fn = tf.initializers.glorot_uniform(seed=42)
            else:
                # Last resort: use numpy random with appropriate scaling (He initialization)
                class HeNormalInitializer:
                    def __init__(self, seed=42):
                        self.seed = seed
                        np.random.seed(seed)
                    def __call__(self, shape, dtype=TF_FLOAT32):
                        # Use numpy random since tf.random may not be available
                        fan_in = np.prod(shape[:-1]) if len(shape) > 1 else shape[0]
                        std = np.sqrt(2.0 / fan_in)
                        values = np.random.randn(*shape).astype(np.float32) * std
                        return TF_constant(values, dtype=dtype)
                initializer_fn = HeNormalInitializer(seed=42)
        
        # Block 1: Conv layers
        self.w1 = TF_Variable(
            initializer_fn(shape=(3, 3, 3, 32), dtype=TF_FLOAT32),
            name='conv1_weights'
        )
        self.b1 = TF_Variable(TF_zeros(32, dtype=TF_FLOAT32), name='conv1_bias')
        
        self.w2 = TF_Variable(
            initializer_fn(shape=(3, 3, 32, 32), dtype=TF_FLOAT32),
            name='conv2_weights'
        )
        self.b2 = TF_Variable(TF_zeros(32, dtype=TF_FLOAT32), name='conv2_bias')
        
        # Block 2: Conv layers
        self.w3 = TF_Variable(
            initializer_fn(shape=(3, 3, 32, 64), dtype=TF_FLOAT32),
            name='conv3_weights'
        )
        self.b3 = TF_Variable(TF_zeros(64, dtype=TF_FLOAT32), name='conv3_bias')
        
        self.w4 = TF_Variable(
            initializer_fn(shape=(3, 3, 64, 64), dtype=TF_FLOAT32),
            name='conv4_weights'
        )
        self.b4 = TF_Variable(TF_zeros(64, dtype=TF_FLOAT32), name='conv4_bias')
        
        # Dense layers
        # 224x224 -> 112x112 (pool) -> 56x56 (pool) -> flatten
        flattened_dim = 56 * 56 * 64
        self.w_fc1 = TF_Variable(
            initializer_fn(shape=(flattened_dim, 256), dtype=TF_FLOAT32),
            name='fc1_weights'
        )
        self.b_fc1 = TF_Variable(TF_zeros(256, dtype=TF_FLOAT32), name='fc1_bias')
        
        self.w_fc2 = TF_Variable(
            initializer_fn(shape=(256, num_classes), dtype=TF_FLOAT32),
            name='fc2_weights'
        )
        self.b_fc2 = TF_Variable(TF_zeros(num_classes, dtype=TF_FLOAT32), name='fc2_bias')
        
        # Keep a list of trainable variables
        self.trainable_variables = [
            self.w1, self.b1,
            self.w2, self.b2,
            self.w3, self.b3,
            self.w4, self.b4,
            self.w_fc1, self.b_fc1,
            self.w_fc2, self.b_fc2,
        ]

    def forward(self, x, training: bool = True):
        """Forward pass"""
        # Block 1
        x = tf.nn.conv2d(x, self.w1, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.b1)
        x = tf.nn.relu(x)
        
        x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.b2)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Block 2
        x = tf.nn.conv2d(x, self.w3, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.b3)
        x = tf.nn.relu(x)
        
        x = tf.nn.conv2d(x, self.w4, strides=[1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(x, self.b4)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Flatten
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1])
        
        # Dense layers
        x = tf.matmul(x, self.w_fc1) + self.b_fc1
        x = tf.nn.relu(x)
        
        logits = tf.matmul(x, self.w_fc2) + self.b_fc2
        return logits

    def predict(self, x):
        """Predict class probabilities"""
        logits = self.forward(x, training=False)
        return tf.nn.softmax(logits)


def compute_loss(logits, labels):
    """Compute cross-entropy loss"""
    labels_one_hot = tf.one_hot(labels, depth=tf.shape(logits)[1])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels_one_hot, logits=logits)
    return tf.reduce_mean(loss)


def train_model(
    data_dir: str,
    output_dir: str = 'models',
    validation_split: float = 0.2,
    test_split: float = 0.1
):
    """
    Train the Edge Sentinel model using TensorFlow
    
    Args:
        data_dir: Directory containing training data
        output_dir: Directory to save trained model
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
    """
    print("🚀 Starting Edge Sentinel Model Training (TensorFlow)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print("\n📂 Loading dataset...")
    images, labels, class_names = load_dataset(data_dir)
    
    num_classes = len(class_names)
    print(f"📊 Number of classes: {num_classes}")
    print(f"📊 Classes: {', '.join(class_names)}")
    
    # Split dataset (train/validation/test)
    print("\n✂️ Splitting dataset...")
    X_temp, X_test, y_temp, y_test = stratified_split(
        images, labels, test_size=test_split, random_state=42
    )
    
    X_train, X_val, y_train, y_val = stratified_split(
        X_temp,
        y_temp,
        test_size=validation_split / (1 - test_split),
        random_state=42,
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")

    # Helper to iterate over numpy arrays in mini-batches
    def iterate_minibatches(X, y, batch_size):
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]

    # Create model
    print("\n🏗️ Creating model architecture (TensorFlow)...")
    model = EdgeSentinelModel(num_classes=num_classes)
    
    # Count parameters
    total_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"📊 Total parameters: {total_params:,}")

    # Optimizer
    # Check if tf.optimizers exists
    if hasattr(tf, 'optimizers') and hasattr(tf.optimizers, 'Adam'):
        optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    elif hasattr(tf, 'train') and hasattr(tf.train, 'AdamOptimizer'):
        # Fallback to TensorFlow 1.x style optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    else:
        raise AttributeError("Cannot find Adam optimizer in TensorFlow. Please check your TensorFlow installation.")

    # Metrics history
    history_accuracy = []
    history_val_accuracy = []
    history_loss = []
    history_val_loss = []

    print("\n🎓 Starting training...")
    print("=" * 60)

    for epoch in range(EPOCHS):
        # Training
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0

        for batch_x, batch_y in iterate_minibatches(X_train, y_train, BATCH_SIZE):
            batch_x_tf = TF_constant(batch_x, dtype=TF_FLOAT32)
            batch_y_tf = TF_constant(batch_y, dtype=TF_INT32)
            
            with tf.GradientTape() as tape:
                logits = model.forward(batch_x_tf, training=True)
                loss = compute_loss(logits, batch_y_tf)
            
            # Compute gradients and update weights
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_losses.append(loss.numpy())
            
            # Predictions
            preds = tf.argmax(logits, axis=1)
            epoch_correct += tf.reduce_sum(tf.cast(preds == batch_y_tf, TF_INT32)).numpy()
            epoch_total += len(batch_y)

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0

        # Validation
        val_losses = []
        val_correct = 0
        val_total = 0
        
        for batch_x, batch_y in iterate_minibatches(X_val, y_val, BATCH_SIZE):
            batch_x_tf = TF_constant(batch_x, dtype=TF_FLOAT32)
            batch_y_tf = TF_constant(batch_y, dtype=TF_INT32)
            
            logits = model.forward(batch_x_tf, training=False)
            loss = compute_loss(logits, batch_y_tf)
            val_losses.append(loss.numpy())
            
            preds = tf.argmax(logits, axis=1)
            val_correct += tf.reduce_sum(tf.cast(preds == batch_y_tf, TF_INT32)).numpy()
            val_total += len(batch_y)

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        history_loss.append(train_loss)
        history_accuracy.append(train_acc)
        history_val_loss.append(val_loss)
        history_val_accuracy.append(val_acc)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
        )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_correct = 0
    test_total = 0
    top3_correct = 0

    for batch_x, batch_y in iterate_minibatches(X_test, y_test, BATCH_SIZE):
        batch_x_tf = TF_constant(batch_x, dtype=TF_FLOAT32)
        batch_y_tf = TF_constant(batch_y, dtype=TF_INT32)
        
        logits = model.forward(batch_x_tf, training=False)
        preds = tf.argmax(logits, axis=1)
        test_correct += tf.reduce_sum(tf.cast(preds == batch_y_tf, TF_INT32)).numpy()
        test_total += len(batch_y)
        
        # Top-3 accuracy
        # Get top-3 predictions (indices sorted in descending order)
        # tf.argsort sorts in ascending order, so we negate to get descending
        top3_indices = tf.argsort(-logits, axis=1)[:, :3]
        for i in range(len(batch_y)):
            if batch_y[i] in top3_indices[i].numpy():
                top3_correct += 1

    test_accuracy = test_correct / test_total if test_total > 0 else 0.0
    test_top3 = top3_correct / test_total if test_total > 0 else 0.0

    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Test Top-3 Accuracy: {test_top3:.4f}")

    # Save model using TensorFlow SavedModel format
    print("\n💾 Saving model...")
    saved_model_path = os.path.join(output_dir, 'edge_sentinel_model')
    
    # Create a module for saving
    # TensorFlow will automatically track variables when assigned as attributes
    class ModelModule(tf.Module):
        def __init__(self, model):
            super().__init__()
            # Assign model as attribute so TensorFlow can track its variables
            self.model = model
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, IMG_SIZE, IMG_SIZE, 3], dtype=TF_FLOAT32)])
        def __call__(self, x):
            return self.model.forward(x, training=False)
    
    module = ModelModule(model)
    # Save the module - TensorFlow will track variables through the model reference
    tf.saved_model.save(module, saved_model_path)
    print(f"✅ Saved TensorFlow model: {saved_model_path}")
    
    # Also save weights as NumPy arrays for compatibility
    weights_path = os.path.join(output_dir, 'edge_sentinel_model_weights.npz')
    np.savez_compressed(
        weights_path,
        w1=model.w1.numpy(), b1=model.b1.numpy(),
        w2=model.w2.numpy(), b2=model.b2.numpy(),
        w3=model.w3.numpy(), b3=model.b3.numpy(),
        w4=model.w4.numpy(), b4=model.b4.numpy(),
        w_fc1=model.w_fc1.numpy(), b_fc1=model.b_fc1.numpy(),
        w_fc2=model.w_fc2.numpy(), b_fc2=model.b_fc2.numpy(),
        num_classes=num_classes,
        img_size=IMG_SIZE
    )
    print(f"✅ Saved model weights: {weights_path}")
    
    # Convert to TFLite
    try:
        print("\n🔄 Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(output_dir, 'model_unquant.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ Saved TFLite model: {tflite_path}")
    except Exception as e:
        print(f"⚠️  TFLite conversion failed: {e}")
        print("   Model saved in TensorFlow SavedModel format instead")
    
    # Save labels
    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w') as f:
        for i, class_name in enumerate(class_names):
            f.write(f"{i} {class_name}\n")
    
    print(f"✅ Saved labels: {labels_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    history_dict = {
        'accuracy': [float(x) for x in history_accuracy],
        'val_accuracy': [float(x) for x in history_val_accuracy],
        'loss': [float(x) for x in history_loss],
        'val_loss': [float(x) for x in history_val_loss]
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
        'total_params': int(total_params),
        'model_format': 'tensorflow',
        'note': 'Model saved in TensorFlow SavedModel format and TFLite format.'
    }
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Saved model info: {info_path}")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"📁 Models saved to: {output_dir}")
    print(f"📦 Model format: TensorFlow SavedModel + TFLite")
    print("=" * 60)

    # For backward compatibility, return model and a simple history dict
    history = {
        'accuracy': history_accuracy,
        'val_accuracy': history_val_accuracy,
        'loss': history_loss,
        'val_loss': history_val_loss,
    }
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
