import React from 'react';
import AgentChatScreen from './screens/AgentChatScreen';
import './App.css';

/**
 * FinSights Web App
 * 
 * The Hybrid Agentic Financial Platform
 * - Cloud Hive (Backend): Manager Agent (Gemini) routes to 7 specialized tools
 * - Edge Sentinel (Frontend): Offline Neural Network filters visual data
 */
const App: React.FC = () => {
  return (
    <div className="app-container">
      <AgentChatScreen />
    </div>
  );
};

export default App;

