import React, { useState, useEffect } from 'react';
import './App.css';
import SearchBar from './components/SearchBar';
import Watchlist from './components/Watchlist';
import Portfolio from './components/Portfolio';
import AIChatSidebar from './components/AIChatSidebar';
import ImageUpload from './components/ImageUpload';

export type View = 'watchlist' | 'portfolio';

function App() {
  const [currentView, setCurrentView] = useState<View>('watchlist');
  const [searchQuery, setSearchQuery] = useState('');
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1 className="app-title">FinSamaritan</h1>
          <nav className="nav-tabs">
            <button
              className={`nav-tab ${currentView === 'watchlist' ? 'active' : ''}`}
              onClick={() => setCurrentView('watchlist')}
            >
              Watchlist
            </button>
            <button
              className={`nav-tab ${currentView === 'portfolio' ? 'active' : ''}`}
              onClick={() => setCurrentView('portfolio')}
            >
              Portfolio
            </button>
          </nav>
        </div>
        <SearchBar 
          value={searchQuery} 
          onChange={setSearchQuery}
          onAddToWatchlist={(symbol) => {
            // This will be handled by the Watchlist component
            setCurrentView('watchlist');
          }}
        />
      </header>

      <main className="app-main">
        <div className="main-content">
          {currentView === 'watchlist' && (
            <Watchlist searchQuery={searchQuery} />
          )}
          {currentView === 'portfolio' && (
            <Portfolio />
          )}
        </div>
      </main>

      <AIChatSidebar isOpen={chatOpen} onClose={() => setChatOpen(false)} />
      
      <button 
        className="chat-toggle-button"
        onClick={() => setChatOpen(!chatOpen)}
        title="Open AI Chat"
      >
        💬
      </button>

      <ImageUpload />
    </div>
  );
}

export default App;
