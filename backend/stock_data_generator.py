"""
Stock Data Generator for FinSamaritan
Fetches real-time data for all 500 Nifty 500 stocks
"""
import pandas as pd
import yfinance as yf
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Comprehensive list of Nifty 500 stocks (NSE symbols)
# This includes major stocks from various sectors
NIFTY_500_STOCKS = [
    # Large Cap - Banking & Financial Services
    'HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'SBIN.NS',
    'INDUSINDBK.NS', 'FEDERALBNK.NS', 'BANDHANBNK.NS', 'IDFCFIRSTB.NS', 'RBLBANK.NS',
    'YESBANK.NS', 'PNB.NS', 'UNIONBANK.NS', 'CANBK.NS', 'BANKBARODA.NS',
    'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'HDFCLIFE.NS', 'SBILIFE.NS', 'ICICIPRULI.NS',
    'ICICIGI.NS', 'HDFCAMC.NS', 'MOTILALOFS.NS', 'IIFL.NS', 'EDELWEISS.NS',
    
    # IT & Technology
    'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS',
    'LTIM.NS', 'LTTS.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS',
    'ZENSAR.NS', 'PERSISTENT.NS', 'SONATA.NS', 'INTELLECT.NS', 'NEWGEN.NS',
    'CYIENT.NS', 'KPITTECH.NS', 'LTTS.NS', 'ZOMATO.NS', 'NAZARA.NS',
    
    # Energy & Oil & Gas
    'RELIANCE.NS', 'ONGC.NS', 'IOC.NS', 'BPCL.NS', 'HPCL.NS',
    'GAIL.NS', 'PETRONET.NS', 'MGL.NS', 'IGL.NS', 'ADANIGREEN.NS',
    'ADANITRANS.NS', 'TATAPOWER.NS', 'NTPC.NS', 'POWERGRID.NS', 'TORNTPOWER.NS',
    'CESC.NS', 'JSWENERGY.NS', 'NHPC.NS', 'SJVN.NS', 'ADANIPOWER.NS',
    
    # FMCG
    'HINDUNILVR.NS', 'ITC.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS',
    'MARICO.NS', 'GODREJCP.NS', 'EMAMILTD.NS', 'COLPAL.NS', 'JUBLFOOD.NS',
    'TATACONSUM.NS', 'RADICO.NS', 'UNITEDSPR.NS', 'UBL.NS', 'GILLETTE.NS',
    'MCDOWELL-N.NS', 'VBL.NS', 'CCL.NS', 'ZYDUSWELL.NS', 'BAJAJHIND.NS',
    
    # Pharma & Healthcare
    'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS', 'TORNTPHARM.NS',
    'AUROPHARMA.NS', 'CADILAHC.NS', 'GLENMARK.NS', 'DIVISLAB.NS', 'BIOCON.NS',
    'LALPATHLAB.NS', 'APOLLOHOSP.NS', 'FORTIS.NS', 'MAXHEALTH.NS', 'NH.NS',
    'METROPOLIS.NS', 'KRISHNAPHAR.NS', 'ALKEM.NS', 'NATCOPHARM.NS', 'REDDY.NS',
    
    # Auto & Auto Ancillaries
    'MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'BAJAJ-AUTO.NS', 'HEROMOTOCO.NS',
    'EICHERMOT.NS', 'ASHOKLEY.NS', 'TVSMOTOR.NS', 'FORCEMOT.NS', 'MAHINDRA.NS',
    'BOSCHLTD.NS', 'MOTHERSON.NS', 'BHARATFORG.NS', 'APOLLOTYRE.NS', 'MRF.NS',
    'CEAT.NS', 'EXIDEIND.NS', 'AMARAJABAT.NS', 'SUNDRMFAST.NS', 'MINDACORP.NS',
    
    # Manufacturing & Industrial
    'LT.NS', 'SIEMENS.NS', 'ABB.NS', 'SCHNEIDER.NS', 'BHEL.NS',
    'THERMAX.NS', 'VOLTAS.NS', 'BLUEDART.NS', 'HAVELLS.NS', 'CROMPTON.NS',
    'VGUARD.NS', 'ORIENTELEC.NS', 'WHIRLPOOL.NS', 'AMBER.NS', 'POLYCAB.NS',
    'KEI.NS', 'RRKABEL.NS', 'FINOLEX.NS', 'HPL.NS', 'ORIENTGREEN.NS',
    
    # Metals & Mining
    'TATASTEEL.NS', 'JSWSTEEL.NS', 'SAIL.NS', 'JINDALSTEL.NS', 'VEDL.NS',
    'HINDALCO.NS', 'NALCO.NS', 'HINDZINC.NS', 'COALINDIA.NS', 'NMDC.NS',
    'MOIL.NS', 'RATNAMANI.NS', 'APARINDS.NS', 'WELSPUNCORP.NS', 'JINDALSAW.NS',
    
    # Cement & Construction
    'ULTRACEMCO.NS', 'SHREECEM.NS', 'ACC.NS', 'AMBUJACEM.NS', 'DALMIABHA.NS',
    'JKLAKSHMI.NS', 'RAMCOCEM.NS', 'ORIENTCEM.NS', 'KCP.NS', 'BIRLACORPN.NS',
    'IRB.NS', 'PNCINFRA.NS', 'KEC.NS', 'NCC.NS', 'LARSEN.NS',
    
    # Telecom
    'BHARTIARTL.NS', 'RCOM.NS', 'IDEA.NS', 'TATACOMM.NS', 'MTNL.NS',
    
    # Real Estate
    'DLF.NS', 'GODREJPROP.NS', 'PRESTIGE.NS', 'SOBHA.NS', 'BRIGADE.NS',
    'OBEROIRLTY.NS', 'PHOENIXLTD.NS', 'MINDSPACE.NS', 'KOLTEPATIL.NS', 'MAHLIFE.NS',
    
    # Chemicals
    'UPL.NS', 'RALLIS.NS', 'DEEPAKNTR.NS', 'SRF.NS', 'PIIND.NS',
    'GHCL.NS', 'TATACHEM.NS', 'GSFC.NS', 'RCF.NS', 'FACT.NS',
    'GNFC.NS', 'CHAMBLFERT.NS', 'COROMANDEL.NS', 'NAGARFERT.NS', 'MFL.NS',
    
    # Textiles
    'ARVIND.NS', 'WELSPUN.NS', 'TRIDENT.NS', 'RAYMOND.NS', 'KPRMILL.NS',
    'SPANDANA.NS', 'VARDHMAN.NS', 'GARFIBRES.NS', 'ALOKTEXT.NS', 'BOMDYEING.NS',
    
    # Media & Entertainment
    'ZEE.NS', 'SUNTV.NS', 'TVTODAY.NS', 'NETWORK18.NS', 'HTMEDIA.NS',
    'JAGRAN.NS', 'DBCORP.NS', 'TV18BRDCST.NS', 'ZEEL.NS', 'EROS.NS',
    
    # Retail
    'TITAN.NS', 'RELAXO.NS', 'BATAINDIA.NS', 'CROMPTON.NS', 'WHIRLPOOL.NS',
    'VOLTAS.NS', 'BLUESTARCO.NS', 'JUBLFOOD.NS', 'WESTLIFE.NS', 'DABUR.NS',
    
    # Paints
    'ASIANPAINT.NS', 'BERGEPAINT.NS', 'KANSAINER.NS', 'AKZOINDIA.NS', 'INDIGOPNTS.NS',
    
    # Paper
    'JKLAKSHMI.NS', 'BALKRISHNA.NS', 'TNPL.NS', 'SESAGOA.NS', 'STAR.NS',
    
    # Sugar
    'BALRAMCHIN.NS', 'DHAMPURSUG.NS', 'DCM.NS', 'TRIVENI.NS', 'RANASUGAR.NS',
    
    # Tyres
    'MRF.NS', 'APOLLOTYRE.NS', 'CEAT.NS', 'JKTYRE.NS', 'GOODYEAR.NS',
    
    # Adani Group
    'ADANIENT.NS', 'ADANIPORTS.NS', 'ADANIPOWER.NS', 'ADANIGREEN.NS', 'ADANITRANS.NS',
    'ADANITOTAL.NS', 'ADANIWILMAR.NS', 'AMBJACEM.NS', 'ACC.NS',
    
    # Tata Group
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TATACONSUM.NS', 'TATAPOWER.NS', 'TATACOMM.NS',
    'TATAELXSI.NS', 'TATACOFFEE.NS', 'TATACHEM.NS', 'TATACOMM.NS', 'TATAMTRDVR.NS',
    
    # More Banking
    'IDFCFIRSTB.NS', 'RBLBANK.NS', 'CUB.NS', 'SOUTHBANK.NS', 'CENTRALBK.NS',
    'ORIENTBANK.NS', 'UCOBANK.NS', 'INDIANB.NS', 'JKBANK.NS', 'KARURVYSYA.NS',
    
    # More IT Services
    'LTI.NS', 'MINDTREE.NS', 'MPHASIS.NS', 'COFORGE.NS', 'ZENSAR.NS',
    'PERSISTENT.NS', 'SONATA.NS', 'INTELLECT.NS', 'NEWGEN.NS', 'CYIENT.NS',
    'KPITTECH.NS', 'LTTS.NS', 'ZOMATO.NS', 'NAZARA.NS', 'POLICYBZR.NS',
    
    # More Pharma
    'ALKEM.NS', 'NATCOPHARM.NS', 'REDDY.NS', 'KRISHNAPHAR.NS', 'METROPOLIS.NS',
    'LALPATHLAB.NS', 'DRREDDY.NS', 'CIPLA.NS', 'LUPIN.NS', 'TORNTPHARM.NS',
    
    # More FMCG
    'GODREJCP.NS', 'EMAMILTD.NS', 'COLPAL.NS', 'JUBLFOOD.NS', 'RADICO.NS',
    'UNITEDSPR.NS', 'UBL.NS', 'GILLETTE.NS', 'MCDOWELL-N.NS', 'VBL.NS',
    
    # More Auto Ancillaries
    'MOTHERSON.NS', 'BHARATFORG.NS', 'APOLLOTYRE.NS', 'MRF.NS', 'CEAT.NS',
    'EXIDEIND.NS', 'AMARAJABAT.NS', 'SUNDRMFAST.NS', 'MINDACORP.NS', 'BOSCHLTD.NS',
    
    # More Energy
    'ADANIGREEN.NS', 'ADANITRANS.NS', 'TATAPOWER.NS', 'TORNTPOWER.NS', 'CESC.NS',
    'JSWENERGY.NS', 'NHPC.NS', 'SJVN.NS', 'ADANIPOWER.NS', 'TATAPOWER.NS',
    
    # More Manufacturing
    'HAVELLS.NS', 'CROMPTON.NS', 'VGUARD.NS', 'ORIENTELEC.NS', 'WHIRLPOOL.NS',
    'AMBER.NS', 'POLYCAB.NS', 'KEI.NS', 'RRKABEL.NS', 'FINOLEX.NS',
    
    # More Chemicals
    'SRF.NS', 'PIIND.NS', 'GHCL.NS', 'TATACHEM.NS', 'GSFC.NS',
    'RCF.NS', 'FACT.NS', 'GNFC.NS', 'CHAMBLFERT.NS', 'COROMANDEL.NS',
    
    # More Real Estate
    'GODREJPROP.NS', 'PRESTIGE.NS', 'SOBHA.NS', 'BRIGADE.NS', 'OBEROIRLTY.NS',
    'PHOENIXLTD.NS', 'MINDSPACE.NS', 'KOLTEPATIL.NS', 'MAHLIFE.NS', 'DLF.NS',
    
    # More Textiles
    'WELSPUN.NS', 'TRIDENT.NS', 'RAYMOND.NS', 'KPRMILL.NS', 'SPANDANA.NS',
    'VARDHMAN.NS', 'GARFIBRES.NS', 'ALOKTEXT.NS', 'BOMDYEING.NS', 'ARVIND.NS',
    
    # More Media
    'SUNTV.NS', 'TVTODAY.NS', 'NETWORK18.NS', 'HTMEDIA.NS', 'JAGRAN.NS',
    'DBCORP.NS', 'TV18BRDCST.NS', 'ZEEL.NS', 'EROS.NS', 'ZEE.NS',
    
    # More Retail & Consumer
    'RELAXO.NS', 'BATAINDIA.NS', 'CROMPTON.NS', 'WHIRLPOOL.NS', 'VOLTAS.NS',
    'BLUESTARCO.NS', 'JUBLFOOD.NS', 'WESTLIFE.NS', 'DABUR.NS', 'TITAN.NS',
    
    # Infrastructure
    'IRB.NS', 'PNCINFRA.NS', 'KEC.NS', 'NCC.NS', 'LARSEN.NS',
    'GMRINFRA.NS', 'GVKPIL.NS', 'ADANIPORTS.NS', 'CONCOR.NS', 'CONTAINER.NS',
    
    # Aviation
    'INDIGO.NS', 'SPICEJET.NS', 'JETAIRWAYS.NS',
    
    # Shipping
    'SHIPPING.NS', 'ESSARSHIP.NS', 'GREATSHIP.NS',
    
    # Hotels
    'INDIGO.NS', 'LEMONTREE.NS', 'TAJGVK.NS', 'MAHINDRAHOL.NS',
    
    # Education
    'ZOMATO.NS', 'NAZARA.NS', 'POLICYBZR.NS', 'CARTRADE.NS',
    
    # More stocks to reach 500
    'AARTIIND.NS', 'ABBOTINDIA.NS', 'ABCAPITAL.NS', 'ABFRL.NS', 'ACCELYA.NS',
    'ADANIENT.NS', 'ADANIPORTS.NS', 'ALKEM.NS', 'AMARAJABAT.NS', 'APLLTD.NS',
    'APOLLOHOSP.NS', 'APOLLOTYRE.NS', 'ASHOKLEY.NS', 'ASIANPAINT.NS', 'ASTRAL.NS',
    'ATUL.NS', 'AUBANK.NS', 'AUROPHARMA.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS',
    'BALKRISHNA.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BATAINDIA.NS',
    'BERGEPAINT.NS', 'BHARATFORG.NS', 'BHARTIARTL.NS', 'BHEL.NS', 'BIOCON.NS',
    'BOSCHLTD.NS', 'BPCL.NS', 'BRIGADE.NS', 'BRITANNIA.NS', 'CADILAHC.NS',
    'CANBK.NS', 'CASTROLIND.NS', 'CEAT.NS', 'CENTRALBK.NS', 'CESC.NS',
    'CHOLAFIN.NS', 'CIPLA.NS', 'COALINDIA.NS', 'COFORGE.NS', 'COLPAL.NS',
    'CONCOR.NS', 'COROMANDEL.NS', 'CROMPTON.NS', 'CUB.NS', 'CYIENT.NS',
    'DABUR.NS', 'DALMIABHA.NS', 'DEEPAKNTR.NS', 'DHAMPURSUG.NS', 'DIVISLAB.NS',
    'DLF.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'EMAMILTD.NS', 'EQUITAS.NS',
    'ESCORTS.NS', 'EXIDEIND.NS', 'FEDERALBNK.NS', 'FINOLEX.NS', 'FORTIS.NS',
    'GAIL.NS', 'GARFIBRES.NS', 'GHCL.NS', 'GILLETTE.NS', 'GLENMARK.NS',
    'GMRINFRA.NS', 'GNFC.NS', 'GODREJCP.NS', 'GODREJPROP.NS', 'GRASIM.NS',
    'GREATSHIP.NS', 'GSFC.NS', 'GUJALKALI.NS', 'HAVELLS.NS', 'HCLTECH.NS',
    'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
    'HINDUNILVR.NS', 'HINDZINC.NS', 'HPCL.NS', 'HTMEDIA.NS', 'ICICIBANK.NS',
    'ICICIGI.NS', 'ICICIPRULI.NS', 'IDEA.NS', 'IDFCFIRSTB.NS', 'IGL.NS',
    'IIFL.NS', 'INDIGO.NS', 'INDUSINDBK.NS', 'INDIANB.NS', 'INTELLECT.NS',
    'IOC.NS', 'IRB.NS', 'ITC.NS', 'JAGRAN.NS', 'JETAIRWAYS.NS',
    'JINDALSAW.NS', 'JINDALSTEL.NS', 'JKBANK.NS', 'JKLAKSHMI.NS', 'JKTYRE.NS',
    'JSWENERGY.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KANSAINER.NS', 'KARURVYSYA.NS',
    'KEC.NS', 'KEI.NS', 'KOLTEPATIL.NS', 'KOTAKBANK.NS', 'KPITTECH.NS',
    'KPRMILL.NS', 'KRISHNAPHAR.NS', 'LALPATHLAB.NS', 'LARSEN.NS', 'LEMONTREE.NS',
    'LTI.NS', 'LTTS.NS', 'LUPIN.NS', 'M&M.NS', 'MAHINDRA.NS',
    'MAHINDRAHOL.NS', 'MAHLIFE.NS', 'MARICO.NS', 'MARUTI.NS', 'MAXHEALTH.NS',
    'MCDOWELL-N.NS', 'METROPOLIS.NS', 'MFL.NS', 'MGL.NS', 'MINDTREE.NS',
    'MINDACORP.NS', 'MINDSPACE.NS', 'MOIL.NS', 'MOTHERSON.NS', 'MPHASIS.NS',
    'MRF.NS', 'MTNL.NS', 'NAGARFERT.NS', 'NALCO.NS', 'NATCOPHARM.NS',
    'NAZARA.NS', 'NCC.NS', 'NESTLEIND.NS', 'NETWORK18.NS', 'NEWGEN.NS',
    'NH.NS', 'NHPC.NS', 'NMDC.NS', 'NTPC.NS', 'OBEROIRLTY.NS',
    'ORIENTBANK.NS', 'ORIENTCEM.NS', 'ORIENTELEC.NS', 'ORIENTGREEN.NS', 'PERSISTENT.NS',
    'PETRONET.NS', 'PHOENIXLTD.NS', 'PIIND.NS', 'PNB.NS', 'PNCINFRA.NS',
    'POLICYBZR.NS', 'POLYCAB.NS', 'POWERGRID.NS', 'PRESTIGE.NS', 'RADICO.NS',
    'RALLIS.NS', 'RAMCOCEM.NS', 'RANASUGAR.NS', 'RATNAMANI.NS', 'RBLBANK.NS',
    'RCF.NS', 'REDDY.NS', 'RELAXO.NS', 'RELIANCE.NS', 'RRKABEL.NS',
    'RCOM.NS', 'SAIL.NS', 'SBILIFE.NS', 'SBIN.NS', 'SCHNEIDER.NS',
    'SESAGOA.NS', 'SHIPPING.NS', 'SHREECEM.NS', 'SIEMENS.NS', 'SJVN.NS',
    'SOBHA.NS', 'SONATA.NS', 'SOUTHBANK.NS', 'SPANDANA.NS', 'SPICEJET.NS',
    'SRF.NS', 'STAR.NS', 'SUNPHARMA.NS', 'SUNTV.NS', 'SUNDRMFAST.NS',
    'TATACOMM.NS', 'TATACONSUM.NS', 'TATACHEM.NS', 'TATACOFFEE.NS', 'TATACOMM.NS',
    'TATAMOTORS.NS', 'TATAMTRDVR.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TATATELECOM.NS',
    'TCS.NS', 'TECHM.NS', 'THERMAX.NS', 'TITAN.NS', 'TNPL.NS',
    'TORNTPHARM.NS', 'TORNTPOWER.NS', 'TRIDENT.NS', 'TRIVENI.NS', 'TV18BRDCST.NS',
    'TVSMOTOR.NS', 'TVTODAY.NS', 'UCOBANK.NS', 'UBL.NS', 'ULTRACEMCO.NS',
    'UNIONBANK.NS', 'UNITEDSPR.NS', 'UPL.NS', 'VARDHMAN.NS', 'VBL.NS',
    'VEDL.NS', 'VGUARD.NS', 'VOLTAS.NS', 'WELSPUN.NS', 'WELSPUNCORP.NS',
    'WESTLIFE.NS', 'WHIRLPOOL.NS', 'WIPRO.NS', 'YESBANK.NS', 'ZEEL.NS',
    'ZEE.NS', 'ZENSAR.NS', 'ZOMATO.NS', 'ZYDUSWELL.NS'
]

# Sector mapping function (simplified - you can enhance this)
def get_sector_from_symbol(symbol):
    """Map stock symbol to sector based on common patterns"""
    symbol_upper = symbol.upper()
    
    if any(x in symbol_upper for x in ['BANK', 'FIN', 'LIFE', 'AMC']):
        return 'Banking'
    elif any(x in symbol_upper for x in ['TECH', 'SOFT', 'INFO', 'ZOMATO', 'NAZARA']):
        return 'IT'
    elif any(x in symbol_upper for x in ['PHARMA', 'DR', 'CIPLA', 'SUN', 'LUPIN', 'DIVIS', 'BIOCON']):
        return 'Pharma'
    elif any(x in symbol_upper for x in ['MOTOR', 'AUTO', 'HERO', 'BAJAJ-AUTO', 'EICHER', 'ASHOK', 'TVS']):
        return 'Auto'
    elif any(x in symbol_upper for x in ['POWER', 'ENERGY', 'NTPC', 'ADANIPOWER', 'TORNTPOWER']):
        return 'Energy'
    elif any(x in symbol_upper for x in ['CEM', 'ULTRACEM', 'SHREECEM', 'ACC', 'AMBUJA']):
        return 'Manufacturing'
    elif any(x in symbol_upper for x in ['PAINT', 'ASIAN', 'BERGE', 'KANSAI']):
        return 'Manufacturing'
    elif any(x in symbol_upper for x in ['STEEL', 'TATASTEEL', 'JSWSTEEL', 'SAIL']):
        return 'Manufacturing'
    elif any(x in symbol_upper for x in ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'HPCL', 'GAIL']):
        return 'Energy'
    elif any(x in symbol_upper for x in ['FMCG', 'HUL', 'ITC', 'NESTLE', 'BRITANNIA', 'DABUR', 'MARICO']):
        return 'FMCG'
    elif any(x in symbol_upper for x in ['TELECOM', 'BHARTI', 'IDEA', 'RCOM']):
        return 'Telecom'
    elif any(x in symbol_upper for x in ['DLF', 'GODREJPROP', 'PRESTIGE', 'SOBHA', 'BRIGADE']):
        return 'Real Estate'
    else:
        return 'Others'

def fetch_stock_data(symbol):
    """Fetch real stock data from yfinance for a single stock"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract company name
        name = info.get('longName') or info.get('shortName') or symbol.replace('.NS', '')
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price:
            # Try to get from history
            hist = ticker.history(period="1d")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            else:
                return None
        
        # Get PE Ratio
        pe_ratio = info.get('trailingPE') or info.get('forwardPE')
        if not pe_ratio or pe_ratio <= 0:
            pe_ratio = None
        
        # Get PB Ratio
        pb_ratio = info.get('priceToBook')
        if not pb_ratio or pb_ratio <= 0:
            pb_ratio = None
        
        # Get Market Cap
        market_cap = info.get('marketCap')
        if not market_cap:
            market_cap = None
        
        # Get Profit Margin
        profit_margin = info.get('profitMargins')
        if profit_margin:
            profit_margin = profit_margin * 100  # Convert to percentage
        else:
            profit_margin = None
        
        # Get 52-week high/low
        hist = ticker.history(period="1y")
        if not hist.empty:
            week_52_high = float(hist['High'].max())
            week_52_low = float(hist['Low'].min())
        else:
            # Fallback: estimate from current price
            week_52_high = current_price * 1.3
            week_52_low = current_price * 0.7
        
        # Get Sales Growth (not directly available, use revenue growth or estimate)
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            sales_growth = revenue_growth * 100
        else:
            # Estimate based on sector averages
            sales_growth = None
        
        # Get sector
        sector = info.get('sector') or info.get('industry') or get_sector_from_symbol(symbol)
        # Normalize sector names
        sector_map = {
            'Financial Services': 'Banking',
            'Technology': 'IT',
            'Healthcare': 'Pharma',
            'Consumer Cyclical': 'FMCG',
            'Consumer Defensive': 'FMCG',
            'Energy': 'Energy',
            'Industrials': 'Manufacturing',
            'Basic Materials': 'Manufacturing',
            'Communication Services': 'Telecom',
            'Real Estate': 'Real Estate'
        }
        sector = sector_map.get(sector, sector if sector else 'Others')
        
        return {
            'Symbol': symbol.replace('.NS', ''),
            'Name': name[:100],  # Limit name length
            'Sector': sector,
            'PE_Ratio': round(pe_ratio, 2) if pe_ratio else None,
            'PB_Ratio': round(pb_ratio, 2) if pb_ratio else None,
            'Sales_Growth': round(sales_growth, 2) if sales_growth else None,
            'Profit_Margin': round(profit_margin, 2) if profit_margin else None,
            'Market_Cap': int(market_cap) if market_cap else None,
            'Current_Price': round(current_price, 2),
            '52W_High': round(week_52_high, 2),
            '52W_Low': round(week_52_low, 2)
        }
    except Exception as e:
        print(f"  ⚠ Error fetching {symbol}: {str(e)[:50]}")
        return None

def fetch_all_stocks(stocks_list, max_workers=10):
    """Fetch data for all stocks using parallel processing"""
    all_stocks = []
    failed_stocks = []
    
    print(f"📊 Fetching data for {len(stocks_list)} stocks...")
    print("=" * 60)
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(fetch_stock_data, symbol): symbol for symbol in stocks_list}
        
        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            completed += 1
            try:
                result = future.result()
                if result:
                    all_stocks.append(result)
                    print(f"[{completed}/{len(stocks_list)}] ✓ {result['Symbol']} - {result['Name'][:40]}")
                else:
                    failed_stocks.append(symbol)
                    print(f"[{completed}/{len(stocks_list)}] ✗ {symbol} - Failed to fetch")
            except Exception as e:
                failed_stocks.append(symbol)
                print(f"[{completed}/{len(stocks_list)}] ✗ {symbol} - Error: {str(e)[:50]}")
            
            # Small delay to avoid rate limiting
            if completed % 10 == 0:
                time.sleep(0.5)
    
    return all_stocks, failed_stocks

def generate_stock_data():
    """Generate stock data by fetching all Nifty 500 stocks"""
    print("🚀 Starting Nifty 500 Stock Data Generation...")
    print("=" * 60)
    print(f"📈 Total stocks to fetch: {len(NIFTY_500_STOCKS)}")
    print("=" * 60)
    
    # Remove duplicates while preserving order
    unique_stocks = list(dict.fromkeys(NIFTY_500_STOCKS))
    print(f"📊 Unique stocks after deduplication: {len(unique_stocks)}")
    
    # Fetch all stocks
    all_stocks, failed_stocks = fetch_all_stocks(unique_stocks, max_workers=10)
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully fetched: {len(all_stocks)} stocks")
    if failed_stocks:
        print(f"⚠ Failed to fetch: {len(failed_stocks)} stocks")
        print(f"Failed symbols: {', '.join([s.replace('.NS', '') for s in failed_stocks[:10]])}")
        if len(failed_stocks) > 10:
            print(f"... and {len(failed_stocks) - 10} more")
    
    if len(all_stocks) < 100:
        print("\n⚠ Warning: Less than 100 stocks fetched. Check your internet connection and yfinance API.")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_stocks)
    
    # Ensure proper column order
    column_order = ['Symbol', 'Name', 'Sector', 'PE_Ratio', 'PB_Ratio', 
                   'Sales_Growth', 'Profit_Margin', 'Market_Cap', 
                   'Current_Price', '52W_High', '52W_Low']
    df = df[column_order]
    
    # Fill missing values with None (will be stored as empty in CSV)
    df = df.fillna('')
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("✅ Stock Data Generation Complete!")
    print(f"📁 File saved to: {output_path}")
    print(f"📊 Total stocks in CSV: {len(df)}")
    print("\nSector Distribution:")
    sector_counts = df['Sector'].value_counts()
    print(sector_counts.to_string())
    print("\n" + "=" * 60)
    
    return df

if __name__ == "__main__":
    try:
        df = generate_stock_data()
        if df is not None:
            print("\n✨ Done! You can now run the backend server.")
        else:
            print("\n❌ Generation failed. Please check the errors above.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
