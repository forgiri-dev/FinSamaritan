import React, { useState, useEffect } from 'react';
import '../App.css';
import apiService, { WatchlistItem } from '../services/api';

interface WatchlistProps {
  searchQuery: string;
}

const Watchlist: React.FC<WatchlistProps> = ({ searchQuery }) => {
  const [symbols, setSymbols] = useState<string[]>([]);
  const [stocks, setStocks] = useState<WatchlistItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadWatchlist();
  }, []);

  const loadWatchlist = async () => {
    try {
      setLoading(true);
      const symbolList = await apiService.getWatchlist();
      setSymbols(symbolList);
      
      // Fetch data for each symbol
      const stockData = await Promise.all(
        symbolList.map(async (symbol) => {
          try {
            const result = await apiService.callTool('view_watchlist', {});
            const stock = result.stocks?.find((s: any) => s.symbol === symbol);
            return stock || { symbol };
          } catch {
            return { symbol };
          }
        })
      );
      setStocks(stockData.filter(Boolean));
    } catch (error) {
      console.error('Failed to load watchlist:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (symbol: string) => {
    try {
      await apiService.removeFromWatchlist(symbol);
      loadWatchlist();
    } catch (error) {
      console.error('Failed to remove from watchlist:', error);
      alert('Failed to remove symbol');
    }
  };

  const filteredStocks = stocks.filter((stock) =>
    stock.symbol.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (loading) {
    return (
      <div className="card">
        <div className="card-title">Watchlist</div>
        <div>Loading...</div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="card-title">My Watchlist ({symbols.length})</div>
      {filteredStocks.length === 0 ? (
        <div style={{ color: 'var(--text-secondary)', padding: '2rem', textAlign: 'center' }}>
          {symbols.length === 0
            ? 'Your watchlist is empty. Add symbols using the search bar above.'
            : 'No symbols match your search.'}
        </div>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Name</th>
              <th>Price</th>
              <th>Change %</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredStocks.map((stock) => (
              <tr key={stock.symbol}>
                <td style={{ fontWeight: 600 }}>{stock.symbol}</td>
                <td>{stock.name || '-'}</td>
                <td>
                  {stock.current_price
                    ? `$${stock.current_price.toFixed(2)}`
                    : '-'}
                </td>
                <td>
                  {stock.change_percent !== undefined ? (
                    <span
                      className={`badge ${
                        stock.change_percent >= 0
                          ? 'badge-success'
                          : 'badge-danger'
                      }`}
                    >
                      {stock.change_percent >= 0 ? '+' : ''}
                      {stock.change_percent.toFixed(2)}%
                    </span>
                  ) : (
                    '-'
                  )}
                </td>
                <td>
                  <button
                    className="btn btn-danger"
                    style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
                    onClick={() => handleRemove(stock.symbol)}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default Watchlist;

