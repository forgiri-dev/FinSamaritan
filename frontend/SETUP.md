# FinSights Frontend Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd frontend
npm install
```

### 2. Configure Backend URL

Edit `src/api/agent.ts` and update the `API_BASE_URL`:

- **Android Emulator**: `http://10.0.2.2:8000`
- **iOS Simulator**: `http://localhost:8000`
- **Physical Device**: `http://YOUR_COMPUTER_IP:8000` (e.g., `http://192.168.1.100:8000`)

### 3. Start Backend Server

In a separate terminal:
```bash
cd backend
uvicorn main:app --reload
```

### 4. Run the App

**Android:**
```bash
npm start
# In another terminal:
npm run android
```

**iOS (Mac only):**
```bash
cd ios && pod install && cd ..
npm start
# In another terminal:
npm run ios
```

## Edge Sentinel Integration

The Edge Sentinel service (`src/services/EdgeSentinel.ts`) currently uses a placeholder implementation. To integrate the actual TensorFlow Lite model:

### Option 1: Using Teachable Machine

1. Go to [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Create an Image Project
3. Train with:
   - Class 1: 30+ images of financial charts
   - Class 2: 30+ images of random objects/selfies
4. Export as TensorFlow Lite (Floating Point)
5. Download `model_unquant.tflite` and `labels.txt`
6. Place in `frontend/assets/`

### Option 2: Using TensorFlow

1. Train a model using TensorFlow/Keras
2. Convert to TFLite:
   ```python
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   with open('model_unquant.tflite', 'wb') as f:
       f.write(tflite_model)
   ```
3. Place in `frontend/assets/`

### Installing TFLite Library

For React Native, you'll need a TFLite library. Options:

1. **react-native-fast-tflite** (if available)
2. **@tensorflow/tfjs-react-native** with TensorFlow.js
3. Custom native module

Update `src/services/EdgeSentinel.ts` to load and use the actual model.

## Android Permissions

Add to `android/app/src/main/AndroidManifest.xml`:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
<uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
```

## iOS Permissions

Add to `ios/FinSights/Info.plist`:

```xml
<key>NSCameraUsageDescription</key>
<string>We need access to your camera to take photos of charts</string>
<key>NSPhotoLibraryUsageDescription</key>
<string>We need access to your photos to select chart images</string>
```

## Troubleshooting

### Metro Bundler Issues
```bash
npm start -- --reset-cache
```

### Android Build Issues
```bash
cd android
./gradlew clean
cd ..
npm run android
```

### iOS Build Issues
```bash
cd ios
pod deintegrate
pod install
cd ..
npm run ios
```

### Backend Connection
- Ensure backend is running on port 8000
- Check firewall settings
- For physical devices, ensure same WiFi network
- Test backend: `curl http://localhost:8000/health`

## Project Structure

```
frontend/
├── src/
│   ├── api/              # Backend API client
│   ├── components/        # Reusable UI components
│   ├── navigation/        # Navigation setup
│   ├── screens/           # Screen components
│   └── services/          # Business logic (Edge Sentinel)
├── assets/                # Images, models, etc.
├── App.tsx               # App entry point
└── package.json
```

## Key Features

✅ **Agent Chat**: Interactive chat with AI Wealth Manager  
✅ **Edge Sentinel**: Local image filtering (placeholder)  
✅ **Markdown Rendering**: Formatted responses with tables  
✅ **Image Analysis**: Chart upload and analysis  
✅ **TypeScript**: Full type safety  

## Next Steps

1. Integrate actual TensorFlow Lite model for Edge Sentinel
2. Add error boundaries for better error handling
3. Implement offline mode with AsyncStorage
4. Add push notifications for portfolio alerts
5. Enhance UI/UX with animations

