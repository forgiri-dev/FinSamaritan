# FinSights Web Frontend

React web application for FinSamaritan - The Hybrid Agentic Financial Platform.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Then open your browser to `http://localhost:3000`

### Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

### Preview Production Build

```bash
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── api/              # Backend API client
│   ├── components/       # Reusable UI components
│   ├── screens/          # Main application screens
│   ├── services/         # Edge Sentinel service
│   ├── App.tsx           # Main app component
│   └── main.tsx          # Entry point
├── index.html            # HTML template
├── vite.config.ts        # Vite configuration
└── package.json          # Dependencies
```

## 🛠️ Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Markdown** - Markdown rendering
- **Axios** - HTTP client
- **TensorFlow.js** - Edge AI (for future model integration)

## 🔧 Configuration

### API Endpoint

Set the backend API URL via environment variable:

```bash
# .env file
VITE_API_URL=http://localhost:8000
```

Or edit `src/api/agent.ts` directly.

## 📝 Features

- ✅ Chat interface with AI agent
- ✅ Markdown rendering for formatted responses
- ✅ Image upload for chart analysis
- ✅ Edge Sentinel image filtering
- ✅ Responsive design
- ✅ Real-time messaging

## 🐛 Troubleshooting

**Port already in use:**
```bash
npm run dev -- --port 3001
```

**Build errors:**
```bash
# Check TypeScript errors
npm run build
```

**API connection issues:**
- Verify backend is running on `http://localhost:8000`
- Check browser console for CORS errors
- Verify `VITE_API_URL` environment variable

## 📚 Documentation

For complete setup instructions, see:
- [Main README](../README.md)
- [Setup Guide](../SETUP_AND_TESTING_GUIDE.md)
- [Quick Start](../QUICK_START.md)
