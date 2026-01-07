import axios from 'axios';

// Backend API base URL - for web, use localhost or your backend URL
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface AgentRequest {
  text: string;
}

export interface AgentResponse {
  success: boolean;
  response: string;
}

export interface ChartRequest {
  image: string; // Base64 encoded
}

export interface ChartResponse {
  success: boolean;
  analysis: string;
}

export interface InferenceResponse {
  success: boolean;
  label: string;
  confidence: number;
  top3: { label: string; confidence: number }[];
}

/**
 * Send a text message to the agent endpoint
 */
export const sendAgentMessage = async (text: string): Promise<string> => {
  try {
    const response = await api.post<AgentResponse>('/agent', { text });
    if (response.data.success) {
      return response.data.response;
    }
    throw new Error('Agent request failed');
  } catch (error: any) {
    if (error.response) {
      throw new Error(`Server error: ${error.response.data.detail || error.message}`);
    } else if (error.request) {
      throw new Error('Network error: Could not reach server. Make sure backend is running.');
    } else {
      throw new Error(`Error: ${error.message}`);
    }
  }
};

/**
 * Analyze a chart image
 */
export const analyzeChart = async (imageBase64: string): Promise<string> => {
  try {
    // Remove data URL prefix if present
    const base64Data = imageBase64.includes(',') 
      ? imageBase64.split(',')[1] 
      : imageBase64;
    
    // Prefer server-side TFLite inference; fallback to Gemini vision if needed
    try {
      const inf = await api.post<InferenceResponse>('/inference/chart', {
        image: base64Data,
      });
      if (inf.data.success) {
        const { label, confidence, top3 } = inf.data;
        const percent = (confidence * 100).toFixed(1);
        const breakdown = top3
          .map((t) => `- ${t.label}: ${(t.confidence * 100).toFixed(1)}%`)
          .join('\n');
        return `Edge Sentinel (server):\n\nTop label: **${label}** (${percent}% confidence)\n\nTop 3:\n${breakdown}`;
      }
    } catch (e) {
      // If inference endpoint fails, fall back to Gemini vision
      console.warn('Inference endpoint failed, falling back to /analyze-chart', e);
    }

    const response = await api.post<ChartResponse>('/analyze-chart', {
      image: base64Data,
    });
    if (response.data.success) {
      return response.data.analysis;
    }
    throw new Error('Chart analysis failed');
  } catch (error: any) {
    if (error.response) {
      throw new Error(`Server error: ${error.response.data.detail || error.message}`);
    } else if (error.request) {
      throw new Error('Network error: Could not reach server.');
    } else {
      throw new Error(`Error: ${error.message}`);
    }
  }
};

/**
 * Health check endpoint
 */
export const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await api.get('/health');
    return response.data.status === 'healthy';
  } catch (error) {
    return false;
  }
};
