import React, { useState, useRef, useEffect } from 'react';
import '../App.css';
import apiService, { ChatMessage } from '../services/api';

interface AIChatSidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const AIChatSidebar: React.FC<AIChatSidebarProps> = ({ isOpen, onClose }) => {
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string; tools?: string[] }>>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);

    try {
      // Convert messages to chat history format
      const history: ChatMessage[] = messages
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m, i) => {
          if (m.role === 'user') {
            return { user: m.content, assistant: '' };
          }
          return { user: '', assistant: m.content };
        })
        .filter((h) => h.user || h.assistant);

      const response = await apiService.chat(userMessage, history);

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.response,
          tools: response.tools_used,
        },
      ]);
    } catch (error) {
      console.error('Chat error:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error. Please try again.',
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!isOpen) return null;

  return (
    <>
      <div className="chat-overlay" onClick={onClose} />
      <div className="chat-sidebar">
        <div className="chat-header">
          <h2>AI Financial Advisor</h2>
          <button className="chat-close-btn" onClick={onClose}>
            ×
          </button>
        </div>

        <div className="chat-messages">
          {messages.length === 0 && (
            <div className="chat-welcome">
              <p>👋 Hello! I'm your AI financial advisor.</p>
              <p>I can help you with:</p>
              <ul>
                <li>📊 Portfolio analysis</li>
                <li>🔍 Stock screening</li>
                <li>📈 Strategy backtesting</li>
                <li>📰 News and research</li>
                <li>💼 Portfolio management</li>
              </ul>
              <p>Ask me anything about your investments!</p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`chat-message ${msg.role}`}>
              <div className="chat-message-content">
                {msg.content.split('\n').map((line, i) => (
                  <React.Fragment key={i}>
                    {line}
                    {i < msg.content.split('\n').length - 1 && <br />}
                  </React.Fragment>
                ))}
              </div>
              {msg.tools && msg.tools.length > 0 && (
                <div className="chat-tools-used">
                  <strong>Tools used:</strong>{' '}
                  {msg.tools.map((tool, i) => (
                    <span key={i} className="tool-badge">
                      {tool}
                      {i < msg.tools!.length - 1 && ', '}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="chat-message assistant">
              <div className="chat-message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <textarea
            className="chat-input"
            placeholder="Ask me anything about your portfolio or stocks..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            rows={3}
          />
          <button
            className="btn btn-primary chat-send-btn"
            onClick={handleSend}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
        </div>
      </div>
    </>
  );
};

export default AIChatSidebar;

