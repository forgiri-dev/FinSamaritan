"""
Test the trained Edge Sentinel model
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import json
from typing import Tuple, List

def load_tflite_model(model_path: str):
    """Load TensorFlow Lite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def load_labels(labels_path: str) -> List[str]:
    """Load labels from labels.txt"""
    labels = []
    with open(labels_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                labels.append(parts[1])
    return labels

def preprocess_image(image_path: str, img_size: int = 224) -> np.ndarray:
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(
    interpreter: tf.lite.Interpreter,
    image_path: str,
    labels: List[str],
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Predict pattern and trend for an image
    
    Returns:
        List of (class_name, confidence) tuples
    """
    # Preprocess image
    img_array = preprocess_image(image_path)
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get top-k predictions
    top_indices = np.argsort(output)[-top_k:][::-1]
    
    predictions = []
    for idx in top_indices:
        class_name = labels[idx]
        confidence = float(output[idx])
        predictions.append((class_name, confidence))
    
    return predictions

def test_model(
    model_path: str,
    labels_path: str,
    test_image_path: str = None,
    test_dir: str = None
):
    """Test the trained model"""
    print("🧪 Testing Edge Sentinel Model")
    print("=" * 60)
    
    # Load model and labels
    print(f"📂 Loading model: {model_path}")
    interpreter = load_tflite_model(model_path)
    
    print(f"📂 Loading labels: {labels_path}")
    labels = load_labels(labels_path)
    
    print(f"✅ Loaded {len(labels)} classes")
    print(f"   Classes: {', '.join(labels[:5])}...")
    
    # Test single image
    if test_image_path and os.path.exists(test_image_path):
        print(f"\n🔍 Testing image: {test_image_path}")
        predictions = predict_image(interpreter, test_image_path, labels, top_k=3)
        
        print("\n📊 Predictions:")
        for i, (class_name, confidence) in enumerate(predictions, 1):
            pattern, trend = class_name.split('_', 1)
            print(f"  {i}. {pattern.upper()} - {trend.upper()}: {confidence:.2%}")
    
    # Test directory
    if test_dir and os.path.exists(test_dir):
        print(f"\n📁 Testing directory: {test_dir}")
        
        correct = 0
        total = 0
        
        for class_dir in os.listdir(test_dir):
            class_path = os.path.join(test_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            
            true_label = class_dir
            
            # Test images in this class
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files[:10]:  # Test first 10 images per class
                img_path = os.path.join(class_path, img_file)
                predictions = predict_image(interpreter, img_path, labels, top_k=1)
                
                predicted_label = predictions[0][0]
                if predicted_label == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Test Results:")
        print(f"   Correct: {correct}/{total}")
        print(f"   Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Edge Sentinel Model')
    parser.add_argument('--model', type=str, default='models/model_unquant.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--labels', type=str, default='models/labels.txt',
                       help='Path to labels file')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to test image')
    parser.add_argument('--test-dir', type=str, default=None,
                       help='Directory with test images')
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model,
        labels_path=args.labels,
        test_image_path=args.image,
        test_dir=args.test_dir
    )

