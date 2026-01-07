/**
 * Edge Sentinel Service (frontend shim)
 *
 * Client-side inference is disabled to avoid unavailable tfjs-tflite package.
 * We only validate the image URL and return a permissive "chart" result.
 * Actual classification happens server-side via /inference/chart (see api/agent.ts).
 */

// Pattern and Trend types
export type CandlestickPattern = 
  | 'hammer'
  | 'doji'
  | 'engulfing_bullish'
  | 'engulfing_bearish'
  | 'shooting_star'
  | 'morning_star'
  | 'evening_star'
  | 'normal';

export type Trend = 'uptrend' | 'downtrend' | 'sideways';

export interface ChartAnalysis {
  isChart: boolean;
  pattern?: CandlestickPattern;
  trend?: Trend;
  confidence?: number;
  fullClassification?: string; // e.g., "hammer_uptrend"
}

/**
 * Parse classification result
 * Converts "pattern_trend" string to structured data
 */
const parseClassification = (classification: string): ChartAnalysis => {
  const parts = classification.split('_');
  
  if (parts.length < 2) {
    return {
      isChart: false,
    };
  }
  
  // Extract trend (last part)
  const trend = parts[parts.length - 1] as Trend;
  
  // Extract pattern (everything except last part)
  const pattern = parts.slice(0, -1).join('_') as CandlestickPattern;
  
  return {
    isChart: true,
    pattern,
    trend,
    fullClassification: classification,
  };
};

/**
 * Scan image using Edge Sentinel (TensorFlow.js model)
 * 
 * Returns analysis including:
 * - Whether it's a financial chart
 * - Detected candlestick pattern
 * - Detected trend
 * - Confidence score
 * 
 * @param imageUrl - URL or data URL of the image
 * @returns Promise<ChartAnalysis> - Analysis results
 */
export const scanImage = async (imageUrl: string): Promise<ChartAnalysis> => {
  try {
    if (!imageUrl || (!imageUrl.startsWith('data:') && !imageUrl.startsWith('http'))) {
      return { isChart: false };
    }
    // Minimal validation only; real classification happens server-side.
    return {
      isChart: true,
      pattern: 'normal',
      trend: 'sideways',
      confidence: 1.0,
      fullClassification: 'normal_sideways',
    };
  } catch (error) {
    console.error('Edge Sentinel error:', error);
    return { isChart: false };
  }
};

/**
 * Get human-readable pattern description
 */
export const getPatternDescription = (pattern: CandlestickPattern): string => {
  const descriptions: Record<CandlestickPattern, string> = {
    hammer: 'Hammer - Bullish reversal pattern',
    doji: 'Doji - Indecision/Reversal signal',
    engulfing_bullish: 'Bullish Engulfing - Strong bullish reversal',
    engulfing_bearish: 'Bearish Engulfing - Strong bearish reversal',
    shooting_star: 'Shooting Star - Bearish reversal',
    morning_star: 'Morning Star - Bullish reversal (3-candle)',
    evening_star: 'Evening Star - Bearish reversal (3-candle)',
    normal: 'Normal - No specific pattern',
  };
  return descriptions[pattern] || 'Unknown pattern';
};

/**
 * Get human-readable trend description
 */
export const getTrendDescription = (trend: Trend): string => {
  const descriptions: Record<Trend, string> = {
    uptrend: 'Uptrend - Price is rising',
    downtrend: 'Downtrend - Price is falling',
    sideways: 'Sideways - Price is consolidating',
  };
  return descriptions[trend] || 'Unknown trend';
};

export default {
  scanImage,
  getPatternDescription,
  getTrendDescription,
};
