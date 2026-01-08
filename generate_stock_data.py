import yfinance as yf
import pandas as pd
import time


def get_nifty500_tickers():
    """Get Nifty 500 ticker symbols"""
    # Download the official Nifty 500 list from NSE
    url = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv'

    try:
        df = pd.read_csv(url)
        # The CSV has columns: Company Name, Industry, Symbol, Series, ISIN Code
        tickers = []
        for _, row in df.iterrows():
            symbol = row['Symbol']
            name = row['Company Name']
            sector = row['Industry']
            tickers.append([symbol, name, sector])
        return tickers
    except Exception as e:
        print(f"Error downloading Nifty 500 list: {e}")
        print("Using backup method...")
        return get_nifty500_backup()


def get_nifty500_backup():
    """Backup method: manually curated list of major Nifty stocks"""
    # Top companies from various sectors
    backup_stocks = [
        ['RELIANCE', 'Reliance Industries Ltd', 'Oil & Gas'],
        ['TCS', 'Tata Consultancy Services Ltd', 'IT Services'],
        ['HDFCBANK', 'HDFC Bank Ltd', 'Banks'],
        ['INFY', 'Infosys Ltd', 'IT Services'],
        ['ICICIBANK', 'ICICI Bank Ltd', 'Banks'],
        ['HINDUNILVR', 'Hindustan Unilever Ltd', 'FMCG'],
        ['ITC', 'ITC Ltd', 'FMCG'],
        ['SBIN', 'State Bank of India', 'Banks'],
        ['BHARTIARTL', 'Bharti Airtel Ltd', 'Telecom'],
        ['KOTAKBANK', 'Kotak Mahindra Bank Ltd', 'Banks'],
        ['LT', 'Larsen & Toubro Ltd', 'Construction'],
        ['AXISBANK', 'Axis Bank Ltd', 'Banks'],
        ['BAJFINANCE', 'Bajaj Finance Ltd', 'Financial Services'],
        ['ASIANPAINT', 'Asian Paints Ltd', 'Paints'],
        ['MARUTI', 'Maruti Suzuki India Ltd', 'Automobiles'],
        ['HCLTECH', 'HCL Technologies Ltd', 'IT Services'],
        ['WIPRO', 'Wipro Ltd', 'IT Services'],
        ['ULTRACEMCO', 'UltraTech Cement Ltd', 'Cement'],
        ['TITAN', 'Titan Company Ltd', 'Jewellery'],
        ['SUNPHARMA', 'Sun Pharmaceutical Industries Ltd', 'Pharmaceuticals'],
        ['NESTLEIND', 'Nestle India Ltd', 'FMCG'],
        ['POWERGRID', 'Power Grid Corporation of India Ltd', 'Power'],
        ['NTPC', 'NTPC Ltd', 'Power'],
        ['TATAMOTORS', 'Tata Motors Ltd', 'Automobiles'],
        ['ADANIENT', 'Adani Enterprises Ltd', 'Diversified'],
        ['ONGC', 'Oil & Natural Gas Corporation Ltd', 'Oil & Gas'],
        ['JSWSTEEL', 'JSW Steel Ltd', 'Steel'],
        ['TATASTEEL', 'Tata Steel Ltd', 'Steel'],
        ['INDUSINDBK', 'IndusInd Bank Ltd', 'Banks'],
        ['BAJAJFINSV', 'Bajaj Finserv Ltd', 'Financial Services'],
        ['DRREDDY', 'Dr. Reddy\'s Laboratories Ltd', 'Pharmaceuticals'],
        ['TECHM', 'Tech Mahindra Ltd', 'IT Services'],
        ['HINDALCO', 'Hindalco Industries Ltd', 'Metals'],
        ['COALINDIA', 'Coal India Ltd', 'Mining'],
        ['GRASIM', 'Grasim Industries Ltd', 'Diversified'],
        ['CIPLA', 'Cipla Ltd', 'Pharmaceuticals'],
        ['DIVISLAB', 'Divi\'s Laboratories Ltd', 'Pharmaceuticals'],
        ['EICHERMOT', 'Eicher Motors Ltd', 'Automobiles'],
        ['HEROMOTOCO', 'Hero MotoCorp Ltd', 'Automobiles'],
        ['BRITANNIA', 'Britannia Industries Ltd', 'FMCG'],
        ['APOLLOHOSP', 'Apollo Hospitals Enterprise Ltd', 'Healthcare'],
        ['ADANIPORTS', 'Adani Ports and Special Economic Zone Ltd', 'Ports'],
        ['SHREECEM', 'Shree Cement Ltd', 'Cement'],
        ['PIDILITIND', 'Pidilite Industries Ltd', 'Chemicals'],
        ['BAJAJ-AUTO', 'Bajaj Auto Ltd', 'Automobiles'],
        ['LTIM', 'LTIMindtree Ltd', 'IT Services'],
        ['SIEMENS', 'Siemens Ltd', 'Engineering'],
        ['HDFCLIFE', 'HDFC Life Insurance Company Ltd', 'Insurance'],
        ['SBILIFE', 'SBI Life Insurance Company Ltd', 'Insurance'],
        ['DABUR', 'Dabur India Ltd', 'FMCG']
    ]
    return backup_stocks


def get_stock_data(symbol, name, sector):
    """Fetch stock data for a given NSE symbol"""
    try:
        # Add .NS suffix for NSE stocks in yfinance
        ticker = f"{symbol}.NS"
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract required data with fallbacks
        data = {
            'Symbol': symbol,
            'Name': name,
            'Sector': sector,
            'Price': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0),
            'PE_Ratio': info.get('trailingPE') or info.get('forwardPE', 0),
            'ROE': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
            'MarketCap': info.get('marketCap', 0)
        }

        return data
    except Exception as e:
        print(f"  ⚠ Error fetching data for {symbol}: {e}")
        return None


def generate_stock_csv(output_file='backend/stock_data.csv', max_stocks=100):
    """
    Generate a CSV file with stock data from Nifty 500

    Parameters:
    - output_file: Path where the CSV will be saved
    - max_stocks: Maximum number of stocks to fetch
    """
    print("=" * 60)
    print("Nifty 500 Stock Data Generator")
    print("=" * 60)

    print("\n📥 Fetching Nifty 500 ticker list...")
    nifty500_list = get_nifty500_tickers()

    stock_data_list = []
    total_stocks = min(len(nifty500_list), max_stocks)

    print(f"\n📊 Fetching data for {total_stocks} stocks from NSE...")
    print("-" * 60)

    for i, (symbol, name, sector) in enumerate(nifty500_list[:max_stocks], 1):
        print(f"[{i}/{total_stocks}] {symbol:<15} {name[:40]:<40}", end='')

        data = get_stock_data(symbol, name, sector)
        if data and data['Price'] > 0:
            stock_data_list.append(data)
            print(" ✓")
        else:
            print(" ✗")

        # Add delay to avoid rate limiting
        if i % 10 == 0:
            print(f"\n⏸  Pausing to avoid rate limits...\n")
            time.sleep(3)
        else:
            time.sleep(0.5)

    print("-" * 60)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(stock_data_list)

    if len(df) == 0:
        print("\n❌ No data was fetched. Please check your internet connection.")
        return None

    # Round numerical values for better readability
    df['Price'] = df['Price'].round(2)
    df['PE_Ratio'] = df['PE_Ratio'].round(2)
    df['ROE'] = df['ROE'].round(2)

    # Sort by Market Cap (largest first)
    df = df.sort_values('MarketCap', ascending=False).reset_index(drop=True)

    df.to_csv(output_file, index=False)

    print(f"\n✅ Successfully created {output_file}")
    print(f"   Total stocks: {len(df)}")
    print("\n" + "=" * 60)
    print("📈 Sample Data (Top 5 by Market Cap):")
    print("=" * 60)
    print(df.head().to_string(index=False))

    print("\n" + "=" * 60)
    print("📊 Summary Statistics:")
    print("=" * 60)
    print(f"Total stocks: {len(df)}")
    print(f"Sectors covered: {df['Sector'].nunique()}")
    print(f"Average Price: ₹{df['Price'].mean():.2f}")
    print(f"Average P/E Ratio: {df['PE_Ratio'].mean():.2f}")
    print(f"Average ROE: {df['ROE'].mean():.2f}%")
    print(f"Total Market Cap: ₹{df['MarketCap'].sum() / 1e12:.2f} Trillion")

    print("\n" + "=" * 60)
    print("📁 File saved successfully!")
    print("=" * 60)

    return df


if __name__ == "__main__":
    # Configuration
    # For testing: max_stocks=20 (~2-3 minutes)
    # For medium dataset: max_stocks=100 (~10-12 minutes)
    # For full Nifty 500: max_stocks=500 (~40-50 minutes)

    print("\n🇮🇳 Fetching data from National Stock Exchange (NSE)")
    print("Note: This may take some time depending on the number of stocks\n")

    df = generate_stock_csv(
        output_file='backend/stock_data.csv',
        max_stocks=500
    )

    if df is not None:
        print("\n✨ Done! Your stock data is ready to use.")
    else:
        print("\n⚠ Failed to generate stock data. Please try again.")






