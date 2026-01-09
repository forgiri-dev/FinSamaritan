"""
Convert Keras Model to TensorFlow Lite Format
Takes the output from train_simple.py and converts it to .tflite format
"""
import os
import sys
import json
import tensorflow as tf
from tensorflow import keras

# Fix Windows console encoding for emojis/logging
if sys.platform == "win32":
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        # If this fails, continue without special encoding handling
        pass

def convert_to_tflite(model_path: str, output_path: str):
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model_path: Path to the saved Keras model (.keras file)
        output_path: Path where the .tflite file should be saved
    """
    print("🔄 Converting Keras model to TensorFlow Lite...")
    print("=" * 60)
    
    # Load the Keras model
    print(f"📂 Loading model from: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Convert to TFLite
    print("\n🔄 Converting to TFLite format...")
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Use default optimizations (can be changed if needed)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"✅ TFLite model saved: {output_path}")
        print(f"📦 Model size: {file_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to TFLite: {e}")
        return False


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite')
    parser.add_argument('--model-path', type=str, default='models/model.keras',
                       help='Path to the Keras model file')
    parser.add_argument('--output-path', type=str, default='models/model_unquant.tflite',
                       help='Path where the TFLite model should be saved')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        print("💡 Make sure you've run train_simple.py first")
        return
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert model
    success = convert_to_tflite(args.model_path, args.output_path)
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 Conversion complete!")
        print(f"📁 TFLite model: {args.output_path}")
        print("💡 You can now use this model in your frontend")
        print("=" * 60)
    else:
        print("\n❌ Conversion failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
