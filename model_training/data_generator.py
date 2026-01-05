"""
Data Generator for Edge Sentinel Model Training
Generates candlestick chart images with labeled patterns and trends
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image
import os
from typing import List, Tuple, Dict
import random
from datetime import datetime, timedelta

# Pattern and Trend Labels
PATTERNS = [
    'hammer',           # Bullish reversal
    'doji',             # Indecision
    'engulfing_bullish', # Bullish reversal
    'engulfing_bearish', # Bearish reversal
    'shooting_star',    # Bearish reversal
    'morning_star',     # Bullish reversal
    'evening_star',     # Bearish reversal
    'normal'            # No specific pattern
]

TRENDS = [
    'uptrend',
    'downtrend',
    'sideways'
]

def generate_synthetic_candlestick_data(
    days: int = 50,
    trend: str = 'uptrend',
    pattern: str = 'normal',
    volatility: float = 0.02
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with specified trend and pattern
    
    Args:
        days: Number of days of data
        trend: 'uptrend', 'downtrend', or 'sideways'
        pattern: Pattern to inject at the end
        volatility: Price volatility factor
    
    Returns:
        DataFrame with OHLCV data
    """
    np.random.seed(random.randint(0, 10000))
    
    # Base price
    base_price = 100.0
    
    # Generate price series based on trend
    if trend == 'uptrend':
        trend_factor = np.linspace(0, 0.3, days)  # 30% upward trend
    elif trend == 'downtrend':
        trend_factor = np.linspace(0, -0.3, days)  # 30% downward trend
    else:  # sideways
        trend_factor = np.linspace(0, 0, days)
    
    # Generate random walk with trend
    returns = np.random.normal(0, volatility, days) + trend_factor / days
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from prices
    data = []
    for i in range(days):
        close = prices[i]
        open_price = close * (1 + np.random.uniform(-0.01, 0.01))
        high = max(open_price, close) * (1 + abs(np.random.uniform(0, 0.02)))
        low = min(open_price, close) * (1 - abs(np.random.uniform(0, 0.02)))
        volume = np.random.uniform(1000000, 10000000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.index = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')
    
    # Inject specific pattern at the end
    if pattern != 'normal':
        df = inject_pattern(df, pattern)
    
    return df

def inject_pattern(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """
    Inject a specific candlestick pattern at the end of the dataframe
    """
    last_idx = len(df) - 1
    prev_close = df.iloc[last_idx - 1]['Close']
    prev_open = df.iloc[last_idx - 1]['Open']
    
    if pattern == 'hammer':
        # Small body at top, long lower wick
        body = prev_close * 0.01
        df.iloc[last_idx, df.columns.get_loc('Open')] = prev_close + body
        df.iloc[last_idx, df.columns.get_loc('Close')] = prev_close + body * 0.5
        df.iloc[last_idx, df.columns.get_loc('High')] = prev_close + body * 1.2
        df.iloc[last_idx, df.columns.get_loc('Low')] = prev_close - body * 3
    
    elif pattern == 'doji':
        # Very small body, wicks on both sides
        mid = prev_close
        df.iloc[last_idx, df.columns.get_loc('Open')] = mid
        df.iloc[last_idx, df.columns.get_loc('Close')] = mid * 1.001
        df.iloc[last_idx, df.columns.get_loc('High')] = mid * 1.02
        df.iloc[last_idx, df.columns.get_loc('Low')] = mid * 0.98
    
    elif pattern == 'engulfing_bullish':
        # Previous candle is bearish, current is large bullish
        df.iloc[last_idx - 1, df.columns.get_loc('Close')] = prev_open * 0.98
        df.iloc[last_idx, df.columns.get_loc('Open')] = prev_close * 0.99
        df.iloc[last_idx, df.columns.get_loc('Close')] = prev_open * 1.03
    
    elif pattern == 'engulfing_bearish':
        # Previous candle is bullish, current is large bearish
        df.iloc[last_idx - 1, df.columns.get_loc('Close')] = prev_open * 1.02
        df.iloc[last_idx, df.columns.get_loc('Open')] = prev_close * 1.01
        df.iloc[last_idx, df.columns.get_loc('Close')] = prev_open * 0.97
    
    elif pattern == 'shooting_star':
        # Small body at bottom, long upper wick
        body = prev_close * 0.01
        df.iloc[last_idx, df.columns.get_loc('Open')] = prev_close - body
        df.iloc[last_idx, df.columns.get_loc('Close')] = prev_close - body * 0.5
        df.iloc[last_idx, df.columns.get_loc('High')] = prev_close + body * 3
        df.iloc[last_idx, df.columns.get_loc('Low')] = prev_close - body * 1.2
    
    elif pattern == 'morning_star':
        # Three-candle pattern: bearish, small, bullish
        if last_idx >= 2:
            df.iloc[last_idx - 2, df.columns.get_loc('Close')] = prev_open * 0.98
            df.iloc[last_idx - 1, df.columns.get_loc('Open')] = prev_close * 0.99
            df.iloc[last_idx - 1, df.columns.get_loc('Close')] = prev_close * 1.001
            df.iloc[last_idx, df.columns.get_loc('Open')] = prev_close * 1.001
            df.iloc[last_idx, df.columns.get_loc('Close')] = prev_open * 1.03
    
    elif pattern == 'evening_star':
        # Three-candle pattern: bullish, small, bearish
        if last_idx >= 2:
            df.iloc[last_idx - 2, df.columns.get_loc('Close')] = prev_open * 1.02
            df.iloc[last_idx - 1, df.columns.get_loc('Open')] = prev_close * 1.01
            df.iloc[last_idx - 1, df.columns.get_loc('Close')] = prev_close * 0.999
            df.iloc[last_idx, df.columns.get_loc('Open')] = prev_close * 0.999
            df.iloc[last_idx, df.columns.get_loc('Close')] = prev_open * 0.97
    
    return df

def fetch_real_stock_data(symbol: str, period: str = '3mo') -> pd.DataFrame:
    """
    Fetch real stock data from yfinance
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        if df.empty:
            return None
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def create_chart_image(
    df: pd.DataFrame,
    output_path: str,
    style: str = 'yahoo',
    figsize: Tuple[int, int] = (8, 6)
) -> bool:
    """
    Create a candlestick chart image from OHLCV data
    
    Args:
        df: DataFrame with OHLCV data
        output_path: Path to save the image
        style: Chart style ('yahoo', 'charles', etc.)
        figsize: Figure size (width, height)
    
    Returns:
        True if successful
    """
    try:
        # Create figure with specific size
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        # Plot candlestick chart
        mpf.plot(
            df,
            type='candle',
            style=style,
            ax=ax,
            volume=False,
            show_nontrading=False,
            tight_layout=True
        )
        
        # Remove axes labels for cleaner image
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        # Save as image
        plt.savefig(
            output_path,
            bbox_inches='tight',
            pad_inches=0,
            dpi=100,
            facecolor='white',
            edgecolor='none'
        )
        plt.close(fig)
        
        # Resize to 224x224 for model input
        img = Image.open(output_path)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
        
        return True
    except Exception as e:
        print(f"Error creating chart: {e}")
        plt.close('all')
        return False

def generate_training_dataset(
    output_dir: str = 'training_data',
    samples_per_class: int = 200,
    use_real_data: bool = True
):
    """
    Generate training dataset with labeled images
    
    Args:
        output_dir: Directory to save training images
        samples_per_class: Number of samples per pattern/trend combination
        use_real_data: Whether to use real stock data when available
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for each class
    for pattern in PATTERNS:
        for trend in TRENDS:
            class_dir = os.path.join(output_dir, f"{pattern}_{trend}")
            os.makedirs(class_dir, exist_ok=True)
    
    # Real stock symbols for data augmentation
    real_symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
        "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "BAJFINANCE.NS"
    ]
    
    total_samples = 0
    
    print("🔄 Generating training dataset...")
    print(f"Target: {samples_per_class} samples per class")
    print(f"Total classes: {len(PATTERNS) * len(TRENDS)}")
    
    for pattern in PATTERNS:
        for trend in TRENDS:
            class_name = f"{pattern}_{trend}"
            class_dir = os.path.join(output_dir, class_name)
            count = 0
            
            print(f"\n📊 Generating {class_name}...")
            
            while count < samples_per_class:
                # Mix synthetic and real data
                if use_real_data and random.random() < 0.3 and pattern == 'normal':
                    # Use real stock data (30% of normal patterns)
                    symbol = random.choice(real_symbols)
                    df = fetch_real_stock_data(symbol)
                    if df is None or len(df) < 30:
                        df = generate_synthetic_candlestick_data(
                            days=random.randint(30, 60),
                            trend=trend,
                            pattern=pattern
                        )
                else:
                    # Generate synthetic data
                    df = generate_synthetic_candlestick_data(
                        days=random.randint(30, 60),
                        trend=trend,
                        pattern=pattern,
                        volatility=random.uniform(0.01, 0.04)
                    )
                
                # Create chart image
                output_path = os.path.join(class_dir, f"{class_name}_{count:04d}.jpg")
                
                if create_chart_image(df, output_path):
                    count += 1
                    total_samples += 1
                    
                    if count % 50 == 0:
                        print(f"  ✅ Generated {count}/{samples_per_class}")
            
            print(f"  ✅ Completed {class_name}: {count} samples")
    
    print(f"\n✅ Dataset generation complete!")
    print(f"Total samples: {total_samples}")
    print(f"Saved to: {output_dir}")
    
    # Create labels file
    create_labels_file(output_dir)
    
    return output_dir

def create_labels_file(output_dir: str):
    """Create labels.txt file for the model"""
    labels = []
    for pattern in PATTERNS:
        for trend in TRENDS:
            labels.append(f"{pattern}_{trend}")
    
    labels_path = os.path.join(output_dir, 'labels.txt')
    with open(labels_path, 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"{i} {label}\n")
    
    print(f"✅ Created labels file: {labels_path}")

if __name__ == "__main__":
    # Generate training dataset
    dataset_dir = generate_training_dataset(
        output_dir='training_data',
        samples_per_class=200,  # Adjust based on your needs
        use_real_data=True
    )
    print(f"\n🎉 Training data ready at: {dataset_dir}")

