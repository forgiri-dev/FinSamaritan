import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface ChatMessage {
  user: string;
  assistant: string;
}

export interface ChatResponse {
  response: string;
  tools_used: string[];
}

export interface WatchlistItem {
  symbol: string;
  name?: string;
  current_price?: number;
  change_percent?: number;
}

export interface PortfolioHolding {
  symbol: string;
  shares: number;
  buy_price: number;
  current_price?: number;
  pnl?: number;
  pnl_percent?: number;
}

export interface ImageAnalysisResult {
  edge_sentinel: {
    pattern: string;
    trend: string;
    full_classification: string;
    confidence: number;
  };
  gemini_analysis: string;
}

export const apiService = {
  // Chat
  chat: async (message: string, history: ChatMessage[]): Promise<ChatResponse> => {
    const response = await api.post<ChatResponse>('/chat', { message, history });
    return response.data;
  },

  // Watchlist
  getWatchlist: async (): Promise<string[]> => {
    const response = await api.get<{ symbols: string[] }>('/watchlist');
    return response.data.symbols;
  },

  addToWatchlist: async (symbol: string): Promise<void> => {
    const response = await api.post('/watchlist', { symbol });
    if (!response.data.success) {
      throw new Error(response.data.error || 'Failed to add to watchlist');
    }
  },

  removeFromWatchlist: async (symbol: string): Promise<void> => {
    await api.delete('/watchlist', { data: { symbol } });
  },

  // Portfolio
  getPortfolio: async (): Promise<PortfolioHolding[]> => {
    const response = await api.get<{ holdings: PortfolioHolding[] }>('/portfolio');
    return response.data.holdings;
  },

  // Tools
  callTool: async (toolName: string, params: any): Promise<any> => {
    const response = await api.post(`/tools/${toolName}`, params);
    return response.data;
  },

  // Image Analysis
  analyzeImage: async (imageFile: File): Promise<ImageAnalysisResult> => {
    const formData = new FormData();
    formData.append('image', imageFile);
    const response = await api.post<ImageAnalysisResult>('/analyze-image', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },
};

export default apiService;

