import React, { useState } from 'react';
import '../App.css';
import apiService from '../services/api';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onAddToWatchlist: (symbol: string) => void;
}

const SearchBar: React.FC<SearchBarProps> = ({ value, onChange, onAddToWatchlist }) => {
  const [isAdding, setIsAdding] = useState(false);

  const handleAdd = async () => {
    if (!value.trim()) return;
    
    const symbol = value.trim().toUpperCase();
    
    // Basic validation
    if (symbol.length === 0) {
      alert('Please enter a symbol');
      return;
    }
    
    if (symbol.length > 10) {
      alert('Symbol is too long');
      return;
    }
    
    setIsAdding(true);
    try {
      await apiService.addToWatchlist(symbol);
      onChange('');
      onAddToWatchlist(symbol);
      // Show success message
      console.log(`Successfully added ${symbol} to watchlist`);
    } catch (error: any) {
      console.error('Failed to add to watchlist:', error);
      const errorMsg = error?.response?.data?.error || error?.message || 'Failed to add symbol to watchlist';
      alert(errorMsg);
    } finally {
      setIsAdding(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleAdd();
    }
  };

  return (
    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
      <input
        type="text"
        className="input"
        placeholder="Search symbol (e.g., AAPL, TSLA, MSFT)..."
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyPress={handleKeyPress}
        style={{ flex: 1, maxWidth: '400px' }}
      />
      <button
        className="btn btn-primary"
        onClick={handleAdd}
        disabled={isAdding || !value.trim()}
      >
        {isAdding ? 'Adding...' : 'Add to Watchlist'}
      </button>
    </div>
  );
};

export default SearchBar;

