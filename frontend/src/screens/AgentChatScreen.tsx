import React, { useState, useEffect, useRef, useCallback } from 'react';
import { sendAgentMessage, analyzeChart } from '../api/agent';
import { scanImage, ChartAnalysis } from '../services/EdgeSentinel';
import { MarkdownView } from '../components/MarkdownView';
import { LoadingDots } from '../components/LoadingDots';
import './AgentChatScreen.css';

export interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
  imageUrl?: string;
}

/**
 * AgentChatScreen
 * 
 * Main chat interface where users interact with the FinSights AI agent.
 * Features:
 * - Text-based queries to the agent
 * - Image upload with Edge Sentinel filtering
 * - Markdown rendering for formatted responses
 */
export const AgentChatScreen: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    // Welcome message
    setMessages([
      {
        id: 1,
        text: '👋 Welcome to FinSights! I\'m your AI Wealth Manager.\n\nI can help you:\n• Manage your portfolio\n• Analyze stocks\n• Screen investments\n• Backtest strategies\n• Compare peers\n• Analyze chart images\n\nHow can I assist you today?',
        isUser: false,
        timestamp: new Date(),
      },
    ]);
  }, []);

  useEffect(() => {
    // Scroll to bottom when new messages arrive
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = useCallback(async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now(),
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await sendAgentMessage(inputText);
      
      const aiMessage: Message = {
        id: Date.now() + 1,
        text: response,
        isUser: false,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: `❌ Error: ${error.message}\n\nPlease make sure the backend server is running.`,
        isUser: false,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [inputText, isLoading]);

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(e.target.value);
    // Auto-resize textarea
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
  };

  const handleImageSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please select an image file');
      return;
    }

    // Show user that image is being processed
    const processingMessage: Message = {
      id: Date.now(),
      text: '🖼️ Processing image...',
      isUser: true,
      timestamp: new Date(),
      imageUrl: URL.createObjectURL(file),
    };

    setMessages((prev) => [...prev, processingMessage]);

    try {
      // Convert file to base64
      const reader = new FileReader();
      reader.onloadend = async () => {
        const base64String = reader.result as string;

        try {
          // Step 1: Edge Sentinel - Analyze chart patterns and trends
          const chartAnalysis = await scanImage(base64String);

          if (!chartAnalysis.isChart) {
            // Remove processing message
            setMessages((prev) =>
              prev.filter((msg) => msg.id !== processingMessage.id)
            );

            alert('Not a Chart - Edge Sentinel detected this is not a financial chart. Please upload a chart image.');
            return;
          }

          // Show detected pattern and trend
          let patternInfo = '';
          if (chartAnalysis.pattern && chartAnalysis.trend) {
            patternInfo = `📊 Edge Sentinel detected: ${chartAnalysis.pattern} pattern in ${chartAnalysis.trend} trend\n\n`;
          }

          // Update processing message
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === processingMessage.id
                ? {
                    ...msg,
                    text: `${patternInfo}🔍 Analyzing chart with Vision Agent...`,
                    isUser: false,
                  }
                : msg
            )
          );

          // Step 2: Send to backend for analysis
          setIsLoading(true);

          const analysis = await analyzeChart(base64String);

          // Replace processing message with analysis
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === processingMessage.id
                ? {
                    ...msg,
                    id: Date.now() + 1,
                    text: analysis,
                    isUser: false,
                  }
                : msg
            )
          );
        } catch (error: any) {
          // Remove processing message
          setMessages((prev) =>
            prev.filter((msg) => msg.id !== processingMessage.id)
          );

          const errorMessage: Message = {
            id: Date.now() + 1,
            text: `❌ Error analyzing chart: ${error.message}`,
            isUser: false,
            timestamp: new Date(),
          };

          setMessages((prev) => [...prev, errorMessage]);
        } finally {
          setIsLoading(false);
          // Reset file input
          if (fileInputRef.current) {
            fileInputRef.current.value = '';
          }
        }
      };

      reader.readAsDataURL(file);
    } catch (error: any) {
      // Remove processing message
      setMessages((prev) =>
        prev.filter((msg) => msg.id !== processingMessage.id)
      );

      const errorMessage: Message = {
        id: Date.now() + 1,
        text: `❌ Error: ${error.message}`,
        isUser: false,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
    }
  }, []);

  return (
    <div className="chat-screen">
      <div className="chat-header">
        <h1>FinSights AI</h1>
        <p>Your AI Wealth Manager</p>
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`message ${message.isUser ? 'message-user' : 'message-ai'}`}
          >
            {message.imageUrl && (
              <img
                src={message.imageUrl}
                alt="Uploaded chart"
                className="message-image"
              />
            )}
            <div className="message-content">
              <MarkdownView content={message.text} />
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message message-ai">
            <div className="message-content">
              <LoadingDots />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-container">
        <input
          type="file"
          ref={fileInputRef}
          accept="image/*"
          onChange={handleImageSelect}
          style={{ display: 'none' }}
        />
        <button
          className="image-button"
          onClick={() => fileInputRef.current?.click()}
          title="Upload chart image"
        >
          📷
        </button>
        <textarea
          className="chat-input"
          value={inputText}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          placeholder="Ask about stocks, portfolio, or upload a chart..."
          rows={1}
          disabled={isLoading}
        />
        <button
          className="send-button"
          onClick={handleSend}
          disabled={!inputText.trim() || isLoading}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default AgentChatScreen;
