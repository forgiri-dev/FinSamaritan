import axios from 'axios';

// Backend API base URL - adjust for your environment
const API_BASE_URL = __DEV__ 
  ? 'http://localhost:8000'  // For Android emulator
  : 'http://10.0.2.2:8000';   // For Android physical device, adjust IP as needed

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
    const response = await api.post<ChartResponse>('/analyze-chart', {
      image: imageBase64,
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

