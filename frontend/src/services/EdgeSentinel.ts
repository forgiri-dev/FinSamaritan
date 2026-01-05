/**
 * Edge Sentinel Service
 * 
 * This service uses TensorFlow.js to classify financial chart images locally.
 * It detects:
 * - Candlestick patterns (Hammer, Doji, Engulfing, etc.)
 * - Trend analysis (Uptrend, Downtrend, Sideways)
 * 
 * This runs entirely in the browser before sending images to the cloud, saving
 * server costs and reducing latency.
 * 
 * Note: For production, you would load the actual TensorFlow.js model.
 * This is a placeholder implementation that can be enhanced with actual model integration.
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

// Labels mapping (should match training labels)
const LABELS: string[] = [
  'hammer_uptrend', 'hammer_downtrend', 'hammer_sideways',
  'doji_uptrend', 'doji_downtrend', 'doji_sideways',
  'engulfing_bullish_uptrend', 'engulfing_bullish_downtrend', 'engulfing_bullish_sideways',
  'engulfing_bearish_uptrend', 'engulfing_bearish_downtrend', 'engulfing_bearish_sideways',
  'shooting_star_uptrend', 'shooting_star_downtrend', 'shooting_star_sideways',
  'morning_star_uptrend', 'morning_star_downtrend', 'morning_star_sideways',
  'evening_star_uptrend', 'evening_star_downtrend', 'evening_star_sideways',
  'normal_uptrend', 'normal_downtrend', 'normal_sideways',
];

/**
 * Load labels from assets
 * In production, read from assets/labels.txt
 */
const loadLabels = async (): Promise<string[]> => {
  try {
    // In production, fetch from public folder:
    // const response = await fetch('/labels.txt');
    // const labelsText = await response.text();
    // return labelsText.split('\n').map(line => line.split(' ')[1]).filter(Boolean);
    
    // For now, return hardcoded labels
    return LABELS;
  } catch (error) {
    console.warn('Failed to load labels, using default');
    return LABELS;
  }
};

/**
 * Pre-process image for model input
 * Resize to 224x224, normalize, convert to tensor format
 */
const preprocessImage = async (imageUrl: string): Promise<number[]> => {
  // Placeholder: In production, use TensorFlow.js to:
  // 1. Load image
  // 2. Resize to 224x224
  // 3. Normalize pixel values
  // 4. Convert to tensor format
  // For now, return dummy data
  return new Array(224 * 224 * 3).fill(0).map(() => Math.random());
};

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
    // In production, this would:
    // 1. Load the TensorFlow.js model (if not already loaded)
    // 2. Pre-process the image (resize to 224x224, normalize)
    // 3. Run inference
    // 4. Get top prediction from output
    // 5. Parse pattern and trend from label
    
    // Placeholder implementation:
    // For demo purposes, we'll do a simple validation
    // In production, replace this with actual TensorFlow.js inference
    
    // Check if it's a valid image URL/data URL
    if (!imageUrl || (!imageUrl.startsWith('data:') && !imageUrl.startsWith('http'))) {
      return {
        isChart: false,
      };
    }
    
    // Simulate model inference
    // In production:
    // const labels = await loadLabels();
    // const preprocessed = await preprocessImage(imageUrl);
    // const output = await model.predict(preprocessed);
    // const topIndex = output.indexOf(Math.max(...output));
    // const classification = labels[topIndex];
    // const confidence = output[topIndex];
    
    // Placeholder: Simulate classification
    // In production, this would come from the actual model
    const labels = await loadLabels();
    const randomIndex = Math.floor(Math.random() * labels.length);
    const classification = labels[randomIndex];
    const confidence = 0.85; // Simulated confidence
    
    const analysis = parseClassification(classification);
    analysis.confidence = confidence;
    
    // Only return as chart if confidence is high enough
    if (confidence > 0.7) {
      return analysis;
    } else {
      return {
        isChart: false,
      };
    }
    
  } catch (error) {
    console.error('Edge Sentinel error:', error);
    // On error, allow the image through (fail open)
    // In production, you might want to fail closed
    return {
      isChart: true,
      pattern: 'normal',
      trend: 'sideways',
      confidence: 0.5,
    };
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
