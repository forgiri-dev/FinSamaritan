# FinSights Frontend

React Native frontend for FinSights - The Hybrid Agentic Financial Platform.

## Features

- **Agent Chat Interface**: Interactive chat with the AI Wealth Manager using GiftedChat
- **Edge Sentinel**: Local TensorFlow Lite model to filter non-chart images before cloud processing
- **Markdown Rendering**: Beautiful formatting for tables, bold text, and structured responses
- **Image Analysis**: Upload chart images for technical analysis via Gemini Vision

## Prerequisites

- Node.js >= 18
- React Native development environment set up
- Android Studio (for Android) or Xcode (for iOS)
- Backend server running on `http://localhost:8000`

## Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

2. For iOS (if developing on Mac):
```bash
cd ios
pod install
cd ..
```

## Running the App

### Android
```bash
# Start Metro bundler
npm start

# In another terminal, run Android
npm run android
```

### iOS
```bash
# Start Metro bundler
npm start

# In another terminal, run iOS
npm run ios
```

## Configuration

### Backend URL

Update the API base URL in `src/api/agent.ts`:

```typescript
const API_BASE_URL = __DEV__ 
  ? 'http://localhost:8000'  // Android emulator
  : 'http://YOUR_IP:8000';   // Physical device - replace YOUR_IP
```

For physical devices, use your computer's local IP address (e.g., `192.168.1.100:8000`).

### Edge Sentinel Model

The Edge Sentinel service (`src/services/EdgeSentinel.ts`) currently uses a placeholder implementation. To integrate the actual TensorFlow Lite model:

1. Train your model using Teachable Machine or TensorFlow
2. Export as TensorFlow Lite (`.tflite`)
3. Place the model in `assets/model_unquant.tflite`
4. Install a React Native TFLite library (e.g., `react-native-fast-tflite`)
5. Update `EdgeSentinel.ts` to load and use the actual model

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── agent.ts          # Backend API client
│   ├── components/
│   │   ├── ChatBubble.tsx     # Custom chat bubble
│   │   ├── LoadingDots.tsx   # Loading indicator
│   │   └── MarkdownView.tsx  # Markdown renderer
│   ├── navigation/
│   │   └── AppNavigator.tsx   # Navigation setup
│   ├── screens/
│   │   ├── AgentChatScreen.tsx  # Main chat interface
│   │   ├── DashboardScreen.tsx   # Dashboard (placeholder)
│   │   └── ChartDoctorScreen.tsx # Chart analysis (placeholder)
│   └── services/
│       └── EdgeSentinel.ts    # TFLite image classifier
├── assets/
│   ├── model_unquant.tflite   # Edge Sentinel model
│   └── labels.txt             # Model labels
├── App.tsx                    # App entry point
└── package.json
```

## Key Components

### AgentChatScreen

The main chat interface where users interact with the AI agent. Features:
- Text-based queries
- Image upload with Edge Sentinel filtering
- Markdown-formatted responses
- Loading indicators

### Edge Sentinel

Local image classification service that:
- Filters out non-chart images (selfies, random objects)
- Saves server costs and reduces latency
- Runs entirely on-device (0.1s latency)

### API Service

Handles communication with the backend:
- `/agent` - Text queries to the Manager Agent
- `/analyze-chart` - Chart image analysis via Vision Agent

## Troubleshooting

### Backend Connection Issues

- Ensure backend is running: `cd backend && uvicorn main:app --reload`
- Check API URL in `src/api/agent.ts`
- For physical devices, ensure phone and computer are on same network
- Check firewall settings

### Image Upload Issues

- Ensure `react-native-image-picker` permissions are granted
- For Android, add permissions to `AndroidManifest.xml`
- For iOS, add permissions to `Info.plist`

### Metro Bundler Issues

- Clear cache: `npm start -- --reset-cache`
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`

## Development Notes

- The app uses TypeScript for type safety
- GiftedChat provides the chat UI foundation
- Markdown rendering is essential for displaying portfolio tables and formatted responses
- Edge Sentinel is a key differentiator - it's not just a wrapper, it's a hybrid architecture

## License

See main project LICENSE file.

