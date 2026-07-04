import argparse
import pandas as pd
from datetime import datetime
import warnings
import time
import os
import re
from pathlib import Path

try:
    from vnstock import Quote, Listing
except ImportError as exc:  # pragma: no cover - guard for missing dependency
    raise ImportError("Please install vnstock v4+: uv pip install vnstock") from exc

warnings.filterwarnings('ignore')

FALLBACK_COMPANIES = ['ACB', 'BCM', 'BID', 'CTG', 'DGC', 'FPT',
'GAS', 'GVR', 'HDB', 'HPG', 'LPB', 'MBB',
'MSN', 'MWG', 'PLX', 'SAB', 'SHB', 'SSB',
'SSI', 'STB', 'TCB', 'TPB', 'VCB', 'VHM',
'VIB', 'VIC', 'VJC', 'VNM', 'VPB', 'VRE']

DEFAULT_START_DATE = '2020-01-01'
DEFAULT_END_DATE = '2025-12-31'
DATA_SOURCE = 'KBS'
DEFAULT_UNIVERSE_GROUP = os.environ.get("VN_UNIVERSE_GROUP", "VN100").upper()
REQUEST_INTERVAL_SEC = float(os.environ.get("VN_REQUEST_INTERVAL_SEC", "5.0"))
MAX_RETRIES = int(os.environ.get("VN_MAX_RETRIES", "5"))
BATCH_SIZE = int(os.environ.get("VN_BATCH_SIZE", "15"))
BATCH_PAUSE_SEC = float(os.environ.get("VN_BATCH_PAUSE_SEC", "75.0"))
_LAST_REQUEST_TS = 0.0

def resolve_companies(group='VN100', source='KBS'):
    try:
        listing = Listing(source=source)
        symbols = listing.symbols_by_group(group=group)
        if symbols is None:
            return []
        if isinstance(symbols, pd.Series):
            values = symbols.dropna().astype(str).str.upper().tolist()
        else:
            values = [str(x).upper() for x in symbols if pd.notna(x)]
        return sorted(list(dict.fromkeys(values)))
    except Exception:
        return []

def extract_wait_seconds(msg):
    # Match Vietnamese "Chờ 49 giây"
    m = re.search(r"Chờ\s+(\d+)\s+giây", str(msg))
    if m:
        return int(m.group(1))
    # Match English "Wait 49 seconds"
    m = re.search(r"Wait\s+(\d+)\s+seconds?", str(msg), flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Match "Chờ" anywhere with a number nearby (more lenient)
    m = re.search(r"Chờ\s+(\d+)", str(msg))
    if m:
        return int(m.group(1))
    # Match rate limit / "giới hạn" style messages
    m = re.search(r"(\d+)\s*giây", str(msg))
    if m:
        return int(m.group(1))
    return None

def throttle():
    global _LAST_REQUEST_TS
    now = time.time()
    wait = REQUEST_INTERVAL_SEC - (now - _LAST_REQUEST_TS)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_TS = time.time()

def crawl_company_data(symbol, start_date, end_date, source='KBS'):
    """Get stock data for one company using vnstock."""
    sources = [source]
    if source.upper() == "VCI":
        sources.append("KBS")
    for src in sources:
        for attempt in range(MAX_RETRIES):
            try:
                throttle()
                quote = Quote(symbol=symbol, source=src, show_log=False)
                hist_data = quote.history(start=start_date, end=end_date, interval='1D')
                if hist_data.empty:
                    break
                hist_data = hist_data.reset_index()
                date_col = "time" if "time" in hist_data.columns else "date"
                if date_col not in hist_data.columns:
                    break
                date_series = pd.to_datetime(hist_data[date_col])
                csv_data = pd.DataFrame({
                    'date': date_series.dt.strftime('%Y-%m-%d'),
                    'company': symbol,
                    'open': hist_data['open'],
                    'high': hist_data['high'],
                    'low': hist_data['low'],
                    'close': hist_data['close'],
                    'adj_close': hist_data['close'],
                    'volume': hist_data['volume']
                })
                return csv_data
            except Exception as e:
                wait_secs = extract_wait_seconds(e)
                if wait_secs is None:
                    wait_secs = min(60, 2 ** attempt)
                time.sleep(wait_secs + 1)
                continue
    print(f"Error for {symbol}: failed on sources {sources}")
    return None

def save_company_data(data, symbol, output_dir: Path):
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{symbol}.csv"
        data.to_csv(filename, index=False)
        return str(filename)
    except Exception:
        return None

def crawl_multiple_companies(companies, start_date, end_date, source='KBS', output_dir: Path | None = None):
    """Crawl multiple companies"""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "raw_csv" / DEFAULT_UNIVERSE_GROUP.lower()
    print(f"Crawling {len(companies)} companies: {start_date} to {end_date}")
    print(f"CSV output dir: {output_dir}")
    
    results = {'successful': [], 'failed': [], 'files_created': []}
    
    for i, symbol in enumerate(companies, 1):
        # Batch pause to avoid rate limits
        if i > 1 and i % BATCH_SIZE == 1:
            print(f"\n[Batch pause] Waiting {BATCH_PAUSE_SEC:.0f}s after {BATCH_SIZE} symbols to respect rate limits...")
            time.sleep(BATCH_PAUSE_SEC)

        print(f"[{i}/{len(companies)}] {symbol}...", end=' ')

        data = crawl_company_data(symbol, start_date, end_date, source)
        
        if data is not None:
            filename = save_company_data(data, symbol, output_dir)
            if filename:
                results['successful'].append(symbol)
                results['files_created'].append(filename)
                print(f"OK ({len(data)} days)")
            else:
                results['failed'].append(symbol)
                print("SAVE ERROR")
        else:
            results['failed'].append(symbol)
            print("NO DATA")
    
    return results

def main(argv=None):
    """Main function"""
    global REQUEST_INTERVAL_SEC

    parser = argparse.ArgumentParser(
        description="Crawl Vietnamese stock data via vnstock and save as CSV files."
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START_DATE}).",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help=f"End date YYYY-MM-DD (default: {DEFAULT_END_DATE}).",
    )
    parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE_GROUP,
        help=f"Universe group name (default: {DEFAULT_UNIVERSE_GROUP}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save CSV files (default: raw_csv/<universe_lower>).",
    )
    parser.add_argument(
        "--source",
        default=DATA_SOURCE,
        help=f"Data source for vnstock (default: {DATA_SOURCE}).",
    )
    parser.add_argument(
        "--interval-sec",
        type=float,
        default=REQUEST_INTERVAL_SEC,
        help=f"Throttle interval between requests in seconds (default: {REQUEST_INTERVAL_SEC}).",
    )
    args = parser.parse_args(argv)

    REQUEST_INTERVAL_SEC = args.interval_sec

    start_date = args.start_date
    end_date = args.end_date
    universe_group = args.universe.upper()
    output_dir = args.output_dir or Path(
        os.environ.get(
            "VN_CSV_OUTPUT_DIR",
            str(Path(__file__).resolve().parent / "raw_csv" / universe_group.lower()),
        )
    )

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if start_dt >= end_dt:
            print("ERROR: Invalid date range")
            return
    except ValueError:
        print("ERROR: Invalid date format")
        return

    companies = resolve_companies(universe_group, args.source)
    if not companies:
        companies = FALLBACK_COMPANIES
    if not companies:
        print("ERROR: No companies resolved")
        return

    # Start crawling
    print(f"Universe group: {universe_group}, symbols={len(companies)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output dir: {output_dir}")
    results = crawl_multiple_companies(
        companies, start_date, end_date, args.source, output_dir=output_dir
    )

    # Summary
    print(f"\nCompleted: {len(results['successful'])} success, {len(results['failed'])} failed")
    if results['failed']:
        print(f"Failed: {', '.join(results['failed'])}")

if __name__ == "__main__":
    main()