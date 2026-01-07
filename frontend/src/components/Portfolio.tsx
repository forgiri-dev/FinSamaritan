import React, { useState, useEffect } from 'react';
import '../App.css';
import apiService, { PortfolioHolding } from '../services/api';

const Portfolio: React.FC = () => {
  const [holdings, setHoldings] = useState<PortfolioHolding[]>([]);
  const [analysis, setAnalysis] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showAddForm, setShowAddForm] = useState(false);
  const [newHolding, setNewHolding] = useState({
    symbol: '',
    shares: '',
    buy_price: '',
  });

  useEffect(() => {
    loadPortfolio();
    loadAnalysis();
  }, []);

  const loadPortfolio = async () => {
    try {
      setLoading(true);
      const data = await apiService.getPortfolio();
      setHoldings(data);
    } catch (error) {
      console.error('Failed to load portfolio:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadAnalysis = async () => {
    try {
      const result = await apiService.callTool('analyze_portfolio', {});
      setAnalysis(result);
    } catch (error) {
      console.error('Failed to load analysis:', error);
    }
  };

  const handleAdd = async () => {
    if (!newHolding.symbol || !newHolding.shares || !newHolding.buy_price) {
      alert('Please fill all fields');
      return;
    }

    try {
      await apiService.callTool('manage_portfolio', {
        action: 'buy',
        symbol: newHolding.symbol.toUpperCase(),
        shares: parseInt(newHolding.shares),
        buy_price: parseFloat(newHolding.buy_price),
      });
      setNewHolding({ symbol: '', shares: '', buy_price: '' });
      setShowAddForm(false);
      loadPortfolio();
      loadAnalysis();
    } catch (error) {
      console.error('Failed to add holding:', error);
      alert('Failed to add holding');
    }
  };

  const handleRemove = async (symbol: string) => {
    if (!confirm(`Remove ${symbol} from portfolio?`)) return;

    try {
      await apiService.callTool('manage_portfolio', {
        action: 'remove',
        symbol,
      });
      loadPortfolio();
      loadAnalysis();
    } catch (error) {
      console.error('Failed to remove holding:', error);
      alert('Failed to remove holding');
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="card-title">Portfolio</div>
        <div>Loading...</div>
      </div>
    );
  }

  return (
    <div>
      {/* Summary Card */}
      {analysis && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-title">Portfolio Summary</div>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1rem',
            }}
          >
            <div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                Total Invested
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>
                ${analysis.total_invested?.toFixed(2) || '0.00'}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                Current Value
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 600 }}>
                ${analysis.current_value?.toFixed(2) || '0.00'}
              </div>
            </div>
            <div>
              <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                Total P&L
              </div>
              <div
                style={{
                  fontSize: '1.5rem',
                  fontWeight: 600,
                  color:
                    (analysis.total_pnl || 0) >= 0
                      ? 'var(--accent-secondary)'
                      : 'var(--accent-danger)',
                }}
              >
                {analysis.total_pnl >= 0 ? '+' : ''}
                ${analysis.total_pnl?.toFixed(2) || '0.00'} (
                {analysis.total_pnl_percent >= 0 ? '+' : ''}
                {analysis.total_pnl_percent?.toFixed(2) || '0.00'}%)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Holdings Card */}
      <div className="card">
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1rem',
          }}
        >
          <div className="card-title">Holdings ({holdings.length})</div>
          <button
            className="btn btn-primary"
            onClick={() => setShowAddForm(!showAddForm)}
          >
            {showAddForm ? 'Cancel' : '+ Add Holding'}
          </button>
        </div>

        {showAddForm && (
          <div
            style={{
              padding: '1rem',
              background: 'var(--bg-tertiary)',
              borderRadius: '8px',
              marginBottom: '1rem',
            }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr auto', gap: '0.5rem' }}>
              <input
                type="text"
                className="input"
                placeholder="Symbol"
                value={newHolding.symbol}
                onChange={(e) =>
                  setNewHolding({ ...newHolding, symbol: e.target.value })
                }
              />
              <input
                type="number"
                className="input"
                placeholder="Shares"
                value={newHolding.shares}
                onChange={(e) =>
                  setNewHolding({ ...newHolding, shares: e.target.value })
                }
              />
              <input
                type="number"
                className="input"
                placeholder="Buy Price"
                value={newHolding.buy_price}
                onChange={(e) =>
                  setNewHolding({ ...newHolding, buy_price: e.target.value })
                }
              />
              <button className="btn btn-primary" onClick={handleAdd}>
                Add
              </button>
            </div>
          </div>
        )}

        {holdings.length === 0 ? (
          <div style={{ color: 'var(--text-secondary)', padding: '2rem', textAlign: 'center' }}>
            Your portfolio is empty. Add holdings to get started.
          </div>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Shares</th>
                <th>Avg Buy Price</th>
                <th>Current Price</th>
                <th>P&L</th>
                <th>P&L %</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {analysis?.holdings?.map((holding: any) => (
                <tr key={holding.symbol}>
                  <td style={{ fontWeight: 600 }}>{holding.symbol}</td>
                  <td>{holding.shares}</td>
                  <td>${holding.buy_price?.toFixed(2)}</td>
                  <td>${holding.current_price?.toFixed(2)}</td>
                  <td
                    style={{
                      color:
                        holding.pnl >= 0
                          ? 'var(--accent-secondary)'
                          : 'var(--accent-danger)',
                    }}
                  >
                    {holding.pnl >= 0 ? '+' : ''}${holding.pnl?.toFixed(2)}
                  </td>
                  <td>
                    <span
                      className={`badge ${
                        holding.pnl_percent >= 0
                          ? 'badge-success'
                          : 'badge-danger'
                      }`}
                    >
                      {holding.pnl_percent >= 0 ? '+' : ''}
                      {holding.pnl_percent?.toFixed(2)}%
                    </span>
                  </td>
                  <td>
                    <button
                      className="btn btn-danger"
                      style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}
                      onClick={() => handleRemove(holding.symbol)}
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
    </div>
  );
};

export default Portfolio;

