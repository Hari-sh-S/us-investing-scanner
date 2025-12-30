# US Portfolio Engine - Modified for US stocks (no .NS suffix)
import pandas as pd
import numpy as np
import yfinance as yf
from indicators import IndicatorLibrary
from scoring import ScoreParser
from pathlib import Path
from datetime import timedelta

class DataCache:
    """Efficient Parquet-based cache for stock data."""

    def __init__(self, cache_dir="us_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, ticker):
        """Generate cache file path."""
        # Replace special characters for Windows compatibility
        safe_ticker = ticker.replace("^", "_idx_").replace("/", "_")
        filename = f"{safe_ticker}.parquet"
        return self.cache_dir / filename

    def get(self, ticker):
        """Retrieve cached data if available."""
        cache_path = self._get_cache_path(ticker)

        if not cache_path.exists():
            return None

        try:
            df = pd.read_parquet(cache_path)
            # Fix MultiIndex columns from old cache format
            if isinstance(df.columns, pd.MultiIndex):
                print(f"[CACHE FIX] {ticker}: Converting MultiIndex columns")
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            # Also check for tuple-like strings in column names
            elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                print(f"[CACHE FIX] {ticker}: Converting tuple columns")
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            return df
        except Exception as e:
            print(f"Cache read error for {ticker}: {e}")
            return None

    def set(self, ticker, data):
        """Store data in cache as Parquet."""
        cache_path = self._get_cache_path(ticker)
        try:
            data.to_parquet(cache_path, compression='snappy')
        except Exception as e:
            print(f"Cache save error for {ticker}: {e}")

    def exists(self, ticker):
        """Check if ticker data exists in cache."""
        return self._get_cache_path(ticker).exists()

    def get_cache_info(self):
        """Get cache statistics."""
        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            'total_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'tickers': [f.stem for f in files]
        }

    def clear(self):
        """Clear all cached data."""
        for file in self.cache_dir.glob("*.parquet"):
            file.unlink()

    def delete_ticker(self, ticker):
        """Delete cache for specific ticker."""
        cache_path = self._get_cache_path(ticker)
        if cache_path.exists():
            cache_path.unlink()


class USPortfolioEngine:
    """Portfolio Engine for US Stocks - uses standard Yahoo Finance tickers (no suffix)."""
    
    def __init__(self, universe, start_date, end_date, initial_capital=100000, use_cache=True):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data = {}
        self.portfolio_value = []
        self.trades = []
        self.holdings_history = []
        self.parser = ScoreParser()
        self.cache = DataCache() if use_cache else None
        self.regime_index_data = None

    @staticmethod
    def _get_scalar(value):
        """Safely extract scalar from potential Series or DataFrame."""
        if isinstance(value, (pd.Series, pd.DataFrame)):
            return value.iloc[0] if len(value) > 0 else 0
        return value

    def download_and_cache_universe(self, universe_tickers, progress_callback=None, stop_flag=None):
        """Fast batch download for US stocks (no suffix needed)."""
        import time

        # Filter already cached
        tickers_to_download = []
        for ticker in universe_tickers:
            if self.cache and self.cache.exists(ticker):
                continue
            tickers_to_download.append(ticker)

        if not tickers_to_download:
            return len(universe_tickers)

        success_count = 0
        start_time = time.time()
        
        # US stocks don't need suffix - use as-is
        ticker_map = {t: t for t in tickers_to_download}
        
        # Download in chunks of 200 (optimal batch size)
        CHUNK_SIZE = 200
        chunks = [tickers_to_download[i:i + CHUNK_SIZE] for i in range(0, len(tickers_to_download), CHUNK_SIZE)]
        
        completed = 0
        for chunk_idx, chunk in enumerate(chunks):
            if stop_flag and stop_flag[0]:
                print(f"Stopped at chunk {chunk_idx}/{len(chunks)}")
                break
            
            # Batch download with threads
            batch_result = self._download_batch(chunk, ticker_map)
            
            # Count successes
            for ticker in chunk:
                if batch_result.get(ticker, False):
                    success_count += 1
                completed += 1
            
            # Update progress after each chunk
            if progress_callback:
                elapsed = time.time() - start_time
                avg = elapsed / completed if completed > 0 else 0
                remaining_time = avg * (len(tickers_to_download) - completed)
                try:
                    progress_callback(completed, len(tickers_to_download), f"Batch {chunk_idx + 1}/{len(chunks)}", remaining_time)
                except TypeError:
                    progress_callback(completed, len(tickers_to_download), f"Batch {chunk_idx + 1}")

        if progress_callback:
            try:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done", 0)
            except TypeError:
                progress_callback(len(tickers_to_download), len(tickers_to_download), "Done")

        elapsed = time.time() - start_time
        print(f"Downloaded {success_count}/{len(tickers_to_download)} stocks in {elapsed:.1f}s ({elapsed/max(len(tickers_to_download),1):.2f}s/stock)")
        return len(universe_tickers)
    
    def _download_batch(self, tickers, ticker_map):
        """Batch download multiple tickers at once using threads."""
        try:
            data = yf.download(
                tickers,
                period="max",
                interval="1d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as e:
            print(f"Batch download failed: {e}")
            return {}
        
        saved = {}
        
        if isinstance(data.columns, pd.MultiIndex):
            for ticker in tickers:
                try:
                    df = data[ticker].dropna(how="all")
                    if not df.empty and len(df) >= 100:
                        df = df.reset_index()
                        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        df = df[[col for col in expected_cols if col in df.columns]]
                        if self.cache:
                            self.cache.set(ticker, df)
                        saved[ticker] = True
                    else:
                        saved[ticker] = False
                except Exception:
                    saved[ticker] = False
        else:
            # Single ticker result (different format)
            if len(tickers) == 1 and not data.empty:
                ticker = tickers[0]
                df = data.dropna(how="all").reset_index()
                if len(df) >= 100:
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    if self.cache:
                        self.cache.set(ticker, df)
                    saved[ticker] = True
        
        return saved
    
    def _download_single(self, ticker, retries=2, backoff=1):
        """Download single ticker with minimal retry."""
        import time
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(ticker, period="max", interval="1d", progress=False)
                if not df.empty and len(df) >= 100:
                    df = df.reset_index()
                    expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df = df[[col for col in expected_cols if col in df.columns]]
                    if self.cache:
                        self.cache.set(ticker, df)
                    return True
            except Exception:
                pass
            time.sleep(backoff)
        return False
    
    def _process_and_cache_df(self, ticker, df):
        """Process dataframe and save to cache with indicators."""
        try:
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            df = df[[col for col in expected_cols if col in df.columns]]
            
            # Calculate indicators
            try:
                df_with_date_index = df.set_index('Date')
                df_with_date_index = IndicatorLibrary.add_momentum_volatility_metrics(df_with_date_index)
                df_with_date_index = IndicatorLibrary.add_regime_filters(df_with_date_index)
                df = df_with_date_index.reset_index()
            except:
                pass  # Use raw data if indicators fail
            
            if self.cache:
                self.cache.set(ticker, df)
        except Exception as e:
            print(f"Error caching {ticker}: {e}")

    def fetch_data(self, progress_callback=None):
        """Fetch data from cache. Indicators should already be pre-calculated during download."""
        print(f"Loading data for {len(self.universe)} stocks...")
        tickers_to_download = []

        def clean_dataframe(df, ticker_name="unknown"):
            """Remove duplicates, detect anomalies, and ensure clean data."""
            try:
                if df is None or df.empty:
                    return df
                # Remove duplicate indices (dates)
                if hasattr(df.index, 'duplicated') and df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                # Sort by date
                df = df.sort_index()
                
                # DATA QUALITY: Detect extreme single-day price changes (>50%)
                if 'Close' in df.columns and len(df) > 3:
                    close = df['Close']
                    pct_change = close.pct_change().abs()
                    for i in range(1, len(pct_change) - 1):
                        if pct_change.iloc[i] > 0.5:  # >50% change
                            next_change = pct_change.iloc[i + 1] if i + 1 < len(pct_change) else 0
                            if next_change > 0.3:  # If it reverses next day
                                print(f"⚠️ DATA ANOMALY: {ticker_name} on {df.index[i].date()} - {pct_change.iloc[i]*100:.1f}% drop, then {next_change*100:.1f}% recovery")
                
                return df
            except Exception:
                return df

        # First, try to load from cache
        for i, ticker in enumerate(self.universe):
            if progress_callback:
                progress_callback(i + 1, len(self.universe), ticker)

            if self.cache:
                cached_data = self.cache.get(ticker)
                if cached_data is not None:
                    try:
                        # Fix index - Date should be the index
                        if 'Date' in cached_data.columns:
                            cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                            cached_data.set_index('Date', inplace=True)

                        # Ensure index is datetime
                        if not isinstance(cached_data.index, pd.DatetimeIndex):
                            cached_data.index = pd.to_datetime(cached_data.index)

                        # Clean data
                        cached_data = clean_dataframe(cached_data, ticker)

                        # Include 300 days BEFORE start_date for indicator lookback
                        extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                        mask = (cached_data.index >= extended_start) & \
                               (cached_data.index <= pd.Timestamp(self.end_date))
                        df_filtered = cached_data[mask].copy()
                        
                        # Flatten MultiIndex columns if present
                        if isinstance(df_filtered.columns, pd.MultiIndex):
                            df_filtered.columns = [col[0] if isinstance(col, tuple) else col for col in df_filtered.columns]

                        if not df_filtered.empty and len(df_filtered) >= 100:
                            self.data[ticker] = df_filtered
                            continue
                    except Exception as e:
                        print(f"Error loading {ticker}: {e}")

            # If not in cache or insufficient data, mark for download
            tickers_to_download.append(ticker)

        # Download missing tickers
        if tickers_to_download:
            print(f"Downloading {len(tickers_to_download)} missing stocks...")
            self.download_and_cache_universe(tickers_to_download, progress_callback)

            # Retry loading after download
            for ticker in tickers_to_download:
                if self.cache:
                    cached_data = self.cache.get(ticker)
                    if cached_data is not None:
                        try:
                            if 'Date' in cached_data.columns:
                                cached_data['Date'] = pd.to_datetime(cached_data['Date'])
                                cached_data.set_index('Date', inplace=True)

                            if not isinstance(cached_data.index, pd.DatetimeIndex):
                                cached_data.index = pd.to_datetime(cached_data.index)

                            cached_data = clean_dataframe(cached_data, ticker)

                            extended_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=300)
                            mask = (cached_data.index >= extended_start) & \
                                   (cached_data.index <= pd.Timestamp(self.end_date))
                            df_filtered = cached_data[mask].copy()
                            
                            if isinstance(df_filtered.columns, pd.MultiIndex):
                                df_filtered.columns = [col[0] if isinstance(col, tuple) else col for col in df_filtered.columns]

                            if not df_filtered.empty:
                                self.data[ticker] = df_filtered
                        except Exception as e:
                            print(f"Error loading {ticker} after download: {e}")

        print(f"Successfully loaded {len(self.data)} stocks")
        return len(self.data) > 0
    
    def calculate_indicators_for_formula(self, formula, regime_config=None):
        """Calculate only the indicators needed for the formula and regime filter."""
        needs_momentum = any(x in formula.upper() for x in ['PERFORMANCE', 'SHARPE', 'SORTINO', 'CALMAR', 'VOLATILITY', 'DRAWDOWN'])
        needs_regime = regime_config is not None and regime_config.get('type') != 'EQUITY'
        
        if not needs_momentum and not needs_regime:
            return
        
        print(f"Calculating indicators (momentum={needs_momentum}, regime={needs_regime})...")
        
        for ticker in self.data:
            try:
                df = self.data[ticker]
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                elif len(df.columns) > 0 and isinstance(df.columns[0], tuple):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                if needs_momentum and '6 Month Performance' not in df.columns:
                    df = IndicatorLibrary.add_momentum_volatility_metrics(df)
                if needs_regime and 'EMA_200' not in df.columns:
                    df = IndicatorLibrary.add_regime_filters(df)
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                self.data[ticker] = df
            except Exception as e:
                print(f"Error calculating indicators for {ticker}: {e}")

    def _get_rebalance_dates(self, all_dates, rebal_config):
        """Generate rebalance dates based on config."""
        freq = rebal_config['frequency']
        all_dates_set = set(all_dates)
        
        if freq == 'Weekly':
            day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4}
            target_day = day_map[rebal_config['day']]
            rebalance_dates = [d for d in all_dates if d.weekday() == target_day]
        else:  # Monthly
            target_date = rebal_config['date']
            alt_option = rebal_config.get('alt_day', 'Next Day')
            
            rebalance_dates = []
            month_groups = {}
            for date in all_dates:
                key = (date.year, date.month)
                if key not in month_groups:
                    month_groups[key] = []
                month_groups[key].append(date)
            
            for (year, month), month_dates in month_groups.items():
                month_dates_sorted = sorted(month_dates)
                rebalance_date = None
                
                for d in month_dates_sorted:
                    if d.day == target_date:
                        rebalance_date = d
                        break
                
                if rebalance_date is None:
                    if alt_option == 'Previous Day':
                        for d in reversed(month_dates_sorted):
                            if d.day < target_date:
                                rebalance_date = d
                                break
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[0]
                    else:  # Next Day
                        for d in month_dates_sorted:
                            if d.day > target_date:
                                rebalance_date = d
                                break
                        if rebalance_date is None and month_dates_sorted:
                            rebalance_date = month_dates_sorted[-1]
                
                if rebalance_date:
                    rebalance_dates.append(rebalance_date)
        
        print(f"Generated {len(rebalance_dates)} rebalance dates from {len(all_dates)} trading days")
        return sorted(rebalance_dates)

    def _check_regime_filter(self, date, regime_config, current_equity=0, peak_equity=0):
        """Check if regime filter is triggered.

        For EQUITY type: checks drawdown from peak equity
        For EQUITY_MA type: handled in main loop (checks equity curve MA)
        For other types: checks technical indicators on index

        Returns: (triggered: bool, action: str, drawdown_pct: float)
        """
        if not regime_config:
            return False, 'none', 0.0

        regime_type = regime_config['type']

        if regime_type == 'EQUITY':
            # Check drawdown from peak equity
            sl_pct = regime_config['value']
            if peak_equity > 0:
                drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
            else:
                drawdown_pct = 0.0

            if drawdown_pct > sl_pct:
                return True, regime_config['action'], drawdown_pct
            return False, 'none', drawdown_pct

        if regime_type == 'EQUITY_MA':
            # EQUITY_MA is handled separately in the main loop
            # This is just a placeholder - actual check done in run_rebalance_strategy
            return False, 'none', 0.0

        # For EMA, MACD, SUPERTREND - need index data
        if self.regime_index_data is None or self.regime_index_data.empty:
            return False, 'none', 0.0

        # Use nearest available date if exact date not found (handles holidays)
        if date not in self.regime_index_data.index:
            nearest = self.regime_index_data.index.asof(date)
            if pd.isna(nearest):
                return False, 'none', 0.0
            row = self.regime_index_data.loc[nearest]
        else:
            row = self.regime_index_data.loc[date]

        def get_scalar(val):
            if hasattr(val, 'iloc'):
                return float(val.iloc[0])
            return float(val) if val is not None else 0.0

        if regime_type == 'EMA':
            ema_period = regime_config['value']
            ema_col = f'EMA_{ema_period}'
            close_price = get_scalar(row.get('Close', 0))
            ema_value = get_scalar(row.get(ema_col, 0))

            triggered = ema_col in row.index and ema_value > 0 and close_price < ema_value

            if triggered:
                return True, regime_config['action'], 0.0

        elif regime_type == 'MACD':
            macd_val = get_scalar(row.get('MACD', 0))
            signal_val = get_scalar(row.get('MACD_Signal', 0))
            triggered = macd_val < signal_val
            if triggered:
                return True, regime_config['action'], 0.0

        elif regime_type == 'SUPERTREND':
            st_direction = row.get('Supertrend_Direction', 'BUY')
            if hasattr(st_direction, 'iloc'):
                st_direction = st_direction.iloc[0]
            triggered = st_direction == 'SELL'
            if triggered:
                return True, regime_config['action'], 0.0

        return False, 'none', 0.0

    def run_rebalance_strategy(self, scoring_formula, num_stocks, exit_rank, 
                              rebal_config, regime_config=None, uncorrelated_config=None, reinvest_profits=True):
        """Advanced backtesting engine with all features."""
        if not self.data:
            print("No data available")
            return
        
        is_valid, msg = self.parser.validate_formula(scoring_formula)
        if not is_valid:
            print(f"Invalid formula: {msg}")
            return
        
        self.calculate_indicators_for_formula(scoring_formula, regime_config)
        
        # Load regime filter index data if needed (not for EQUITY or EQUITY_MA)
        if regime_config and regime_config['type'] not in ['EQUITY', 'EQUITY_MA']:
            regime_index = regime_config['index']
            # Map universe names to Yahoo Finance tickers for US indices
            index_map = {
                'S&P 500': '^GSPC',
                'NASDAQ 100': '^NDX',
                'DOW 30': '^DJI',
                'Russell 2000': '^RUT',
                'S&P MidCap 400': '^MID',
                'S&P SmallCap 600': '^SML',
                'VIX': '^VIX',
            }
            index_ticker = index_map.get(regime_index, '^GSPC')
            
            try:
                extended_start = pd.Timestamp(self.start_date) - timedelta(days=400)
                regime_data = yf.download(index_ticker, start=extended_start, end=self.end_date, progress=False)
                if not regime_data.empty:
                    print(f"Downloaded {len(regime_data)} days of regime index data (with 400-day pre-buffer for EMA)")
                    regime_data = IndicatorLibrary.add_regime_filters(regime_data)
                    self.regime_index_data = regime_data
            except Exception as e:
                print(f"Could not load regime index data: {e}")
        
        # Get common date range
        all_dates = sorted(list(set().union(*[df.index for df in self.data.values()])))
        rebalance_dates = self._get_rebalance_dates(all_dates, rebal_config)
        
        # Initialize portfolio
        cash = self.initial_capital
        holdings = {}
        portfolio_history = []
        regime_active = False
        regime_cash_reserve = 0
        last_known_prices = {}
        
        # EQUITY regime filter tracking
        is_equity_regime = regime_config and regime_config['type'] == 'EQUITY'
        is_equity_ma_regime = regime_config and regime_config['type'] == 'EQUITY_MA'
        peak_equity = self.initial_capital
        equity_regime_active = False  # True when drawdown exceeds threshold, waiting for recovery
        equity_sl_pct = regime_config['value'] if is_equity_regime else 0
        # Recovery threshold - defaults to same as trigger if not specified
        recovery_dd_pct = regime_config.get('recovery_dd', equity_sl_pct) if is_equity_regime else 0
        if recovery_dd_pct is None or recovery_dd_pct >= equity_sl_pct:
            recovery_dd_pct = equity_sl_pct  # Fallback to same as trigger
        self.regime_trigger_events = []
        
        # EQUITY_MA tracking
        equity_ma_period = regime_config.get('ma_period', 50) if is_equity_ma_regime else 50
        equity_values_history = []  # Track equity values for MA calculation
        
        for date in all_dates:
            is_rebalance = date in rebalance_dates
            
            if is_rebalance:
                # Sell all current holdings
                for ticker, shares in holdings.items():
                    if ticker in self.data and date in self.data[ticker].index:
                        sell_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        proceeds = shares * sell_price
                        cash += proceeds
                        
                        self.trades.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Action': 'SELL',
                            'Shares': shares,
                            'Price': sell_price,
                            'Value': proceeds
                        })
                
                holdings = {}
                
                if reinvest_profits:
                    investable_capital = float(cash)
                else:
                    investable_capital = min(float(cash), self.initial_capital)
                
                # Check regime filter with current equity state
                current_equity = investable_capital
                
                # Handle EQUITY regime with recovery logic
                if is_equity_regime:
                    # Update peak equity
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    
                    regime_triggered, regime_action, drawdown_pct = self._check_regime_filter(
                        date, regime_config, current_equity=current_equity, peak_equity=peak_equity
                    )
                    
                    if regime_triggered and not equity_regime_active:
                        # Trigger regime filter
                        equity_regime_active = True
                        self.regime_trigger_events.append({
                            'date': date,
                            'type': 'trigger',
                            'drawdown': drawdown_pct,
                            'current': current_equity,
                            'peak': peak_equity
                        })
                    elif equity_regime_active:
                        # Check for recovery
                        if drawdown_pct <= recovery_dd_pct:
                            equity_regime_active = False
                            self.regime_trigger_events.append({
                                'date': date,
                                'type': 'recovery',
                                'drawdown': drawdown_pct,
                                'current': current_equity,
                                'peak': peak_equity
                            })
                            regime_triggered = False
                        else:
                            # Still in drawdown, keep filter active
                            regime_triggered = True
                    
                # Handle EQUITY_MA regime
                elif is_equity_ma_regime:
                    # Calculate equity curve MA
                    equity_values_history.append(current_equity)
                    if len(equity_values_history) >= equity_ma_period:
                        equity_ma = sum(equity_values_history[-equity_ma_period:]) / equity_ma_period
                        regime_triggered = current_equity < equity_ma
                        regime_action = regime_config['action'] if regime_triggered else 'none'
                    else:
                        regime_triggered = False
                        regime_action = 'none'
                    drawdown_pct = 0.0
                    
                else:
                    # Other regime types (EMA, MACD, SUPERTREND)
                    regime_triggered, regime_action, drawdown_pct = self._check_regime_filter(
                        date, regime_config, current_equity=current_equity, peak_equity=peak_equity
                    )
                
                stocks_target = 0.0
                uncorrelated_target = 0.0
                
                if regime_triggered:
                    if regime_action == 'Go Cash':
                        stocks_target = 0.0
                        if uncorrelated_config:
                            allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                            uncorrelated_target = investable_capital * allocation_pct
                        regime_active = True
                        
                    elif regime_action == 'Half Portfolio':
                        stocks_target = investable_capital * 0.5
                        if uncorrelated_config:
                            allocation_pct = uncorrelated_config['allocation_pct'] / 100.0
                            uncorrelated_target = (investable_capital * 0.5) * allocation_pct
                        regime_active = True
                else:
                    regime_active = False
                    uncorrelated_target = 0.0
                    stocks_target = investable_capital
                
                # Execute uncorrelated asset purchase
                if uncorrelated_target > 0 and uncorrelated_config:
                    uncorrelated_asset = uncorrelated_config['asset']
                    
                    if uncorrelated_asset not in self.data:
                        try:
                            unc_df = yf.download(uncorrelated_asset, start=self.start_date, end=self.end_date, progress=False, auto_adjust=True)
                            if not unc_df.empty:
                                unc_df.reset_index(inplace=True)
                                unc_df['Date'] = pd.to_datetime(unc_df['Date'])
                                unc_df.set_index('Date', inplace=True)
                                unc_df = unc_df.reindex(all_dates).ffill().bfill()
                                self.data[uncorrelated_asset] = unc_df
                        except Exception as e:
                            print(f"Could not download {uncorrelated_asset}: {e}")
                    
                    if uncorrelated_asset in self.data and date in self.data[uncorrelated_asset].index:
                        unc_price = self._get_scalar(self.data[uncorrelated_asset].loc[date, 'Close'])
                        unc_shares = int(uncorrelated_target / unc_price)
                        
                        if unc_shares > 0:
                            unc_cost = unc_shares * unc_price
                            cash -= unc_cost
                            holdings[uncorrelated_asset] = unc_shares
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': uncorrelated_asset,
                                'Action': 'BUY',
                                'Shares': unc_shares,
                                'Price': unc_price,
                                'Value': unc_cost,
                                'Score': 0,
                                'Rank': 'Uncorrelated'
                            })
                
                available_for_stocks = stocks_target
                
                # Calculate scores for all stocks
                scores = {}
                uncorrelated_asset_ticker = uncorrelated_config['asset'] if uncorrelated_config else None
                
                date_rows = {}
                for ticker, df in self.data.items():
                    if ticker == uncorrelated_asset_ticker:
                        continue
                    if date in df.index:
                        date_rows[ticker] = df.loc[date]
                
                if date_rows:
                    all_rows_df = pd.DataFrame(date_rows).T
                    
                    try:
                        scores_series = self.parser.calculate_scores(all_rows_df, scoring_formula)
                        scores = scores_series.to_dict()
                        scores = {k: v for k, v in scores.items() if v > -999999}
                    except:
                        for ticker, row in date_rows.items():
                            score = self.parser.parse_and_calculate(scoring_formula, row)
                            if score > -999999:
                                scores[ticker] = score
                
                ranked_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                if len(scores) == 0:
                    print(f"   [WARN] No stocks scored on {date.date()}")
                
                top_stocks = ranked_stocks[:num_stocks]
                
                if top_stocks and available_for_stocks > 0:
                    position_value = available_for_stocks / len(top_stocks)
                    
                    for ticker, score in top_stocks:
                        buy_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        shares = int(position_value / buy_price)
                        
                        if shares > 0:
                            cost = shares * buy_price
                            cash -= cost
                            holdings[ticker] = shares
                            
                            self.trades.append({
                                'Date': date,
                                'Ticker': ticker,
                                'Action': 'BUY',
                                'Shares': shares,
                                'Price': buy_price,
                                'Value': cost,
                                'Score': score,
                                'Rank': ranked_stocks.index((ticker, score)) + 1
                            })
            
            # Calculate portfolio value
            holdings_value = 0.0
            for ticker, shares in holdings.items():
                if ticker in self.data:
                    if date in self.data[ticker].index:
                        close_price = self._get_scalar(self.data[ticker].loc[date, 'Close'])
                        last_known_prices[ticker] = close_price
                    elif ticker in last_known_prices:
                        close_price = last_known_prices[ticker]
                    else:
                        continue
                    holdings_value += shares * close_price
            
            total_value = cash + holdings_value
            
            portfolio_history.append({
                'Date': date,
                'Portfolio Value': total_value,
                'Cash': cash,
                'Holdings Value': holdings_value,
                'Positions': len(holdings)
            })
        
        # Create DataFrames
        self.portfolio_df = pd.DataFrame(portfolio_history)
        self.portfolio_df.set_index('Date', inplace=True)
        self.trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        print(f"Backtest complete: {len(self.portfolio_df)} days, {len(self.trades)} trades")

    def get_metrics(self):
        """Calculate comprehensive performance metrics."""
        if self.portfolio_df.empty:
            return None
        
        portfolio_values = self.portfolio_df['Portfolio Value']
        
        # Basic Returns
        initial_value = portfolio_values.iloc[0]
        final_value = portfolio_values.iloc[-1]
        total_return = final_value - initial_value
        return_pct = (total_return / initial_value) * 100
        
        # CAGR
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = days / 365.25
        cagr = ((final_value / initial_value) ** (1 / max(years, 0.1)) - 1) * 100 if years > 0 else 0
        
        # Drawdown
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Volatility
        daily_returns = portfolio_values.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Sharpe Ratio (assuming 5% risk-free rate)
        risk_free_rate = 0.05
        excess_return = (cagr / 100) - risk_free_rate
        sharpe = excess_return / (volatility / 100) if volatility > 0 else 0
        
        # Win Rate and Trade Statistics
        wins = 0
        losses = 0
        win_amounts = []
        loss_amounts = []
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        last_was_win = None
        total_trades = 0
        
        if not self.trades_df.empty and 'Action' in self.trades_df.columns:
            buy_trades = self.trades_df[self.trades_df['Action'] == 'BUY'].copy()
            sell_trades = self.trades_df[self.trades_df['Action'] == 'SELL'].copy()
            
            if not sell_trades.empty:
                for _, sell_row in sell_trades.iterrows():
                    ticker = sell_row['Ticker']
                    sell_date = sell_row['Date']
                    sell_value = sell_row['Value']
                    
                    # Find previous BUY for this ticker
                    prev_buys = buy_trades[(buy_trades['Ticker'] == ticker) & (buy_trades['Date'] < sell_date)]
                    if not prev_buys.empty:
                        buy_row = prev_buys.iloc[-1]
                        buy_value = buy_row['Value']
                        pnl = sell_value - buy_value
                        
                        total_trades += 1
                        
                        if pnl > 0:
                            wins += 1
                            win_amounts.append(pnl)
                            if last_was_win == True:
                                current_streak += 1
                            else:
                                current_streak = 1
                            max_consecutive_wins = max(max_consecutive_wins, current_streak)
                            last_was_win = True
                        elif pnl < 0:
                            losses += 1
                            loss_amounts.append(abs(pnl))
                            if last_was_win == False:
                                current_streak += 1
                            else:
                                current_streak = 1
                            max_consecutive_losses = max(max_consecutive_losses, current_streak)
                            last_was_win = False
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0
        
        # Expectancy
        avg_win = np.mean(win_amounts) if win_amounts else 0
        avg_loss = np.mean(loss_amounts) if loss_amounts else 0
        win_pct = wins / total_trades if total_trades > 0 else 0
        loss_pct = losses / total_trades if total_trades > 0 else 0
        expectancy = (win_pct * avg_win) - (loss_pct * avg_loss)
        
        # Drawdown Recovery Analysis
        is_in_drawdown = portfolio_values < running_max
        max_recovery_days = 0
        max_recovery_trades = 0
        
        if is_in_drawdown.any():
            drawdown_start = None
            for i, (date, in_dd) in enumerate(is_in_drawdown.items()):
                if in_dd and drawdown_start is None:
                    drawdown_start = date
                elif not in_dd and drawdown_start is not None:
                    days_in_dd = (date - drawdown_start).days
                    max_recovery_days = max(max_recovery_days, days_in_dd)
                    
                    if not self.trades_df.empty and 'Date' in self.trades_df.columns:
                        trades_in_period = self.trades_df[
                            (self.trades_df['Date'] >= drawdown_start) & 
                            (self.trades_df['Date'] <= date)
                        ]
                        max_recovery_trades = max(max_recovery_trades, len(trades_in_period) // 2)
                    
                    drawdown_start = None
        
        # Total Turnover and SEC Fees (US market)
        total_turnover = 0
        total_buy_value = 0
        total_sell_value = 0
        
        if not self.trades_df.empty and 'Action' in self.trades_df.columns:
            buy_trades_vals = self.trades_df[self.trades_df['Action'] == 'BUY']
            sell_trades_vals = self.trades_df[self.trades_df['Action'] == 'SELL']
            total_buy_value = buy_trades_vals['Value'].sum() if not buy_trades_vals.empty else 0
            total_sell_value = sell_trades_vals['Value'].sum() if not sell_trades_vals.empty else 0
            total_turnover = total_buy_value + total_sell_value
        
        # SEC fees (US) - approximately $22.90 per $1M of sales
        sec_fee = total_sell_value * 0.0000229
        # FINRA TAF - $0.000119 per share (simplified estimate)
        taf_fee = total_trades * 0.50  # Rough estimate
        total_charges = sec_fee + taf_fee

        return {
            'Initial Capital': initial_value,
            'Final Value': final_value,
            'Total Return': total_return,
            'Return %': return_pct,
            'CAGR %': cagr,
            'Max Drawdown %': max_drawdown,
            'Volatility %': volatility,
            'Sharpe Ratio': sharpe,
            'Win Rate %': win_rate,
            'Total Trades': total_trades,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Expectancy': expectancy,
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses,
            'Days to Recover from DD': max_recovery_days,
            'Trades to Recover from DD': max_recovery_trades,
            'Total Turnover': total_turnover,
            'Total Charges': total_charges,
        }

    def get_monthly_returns(self):
        """Calculate monthly returns breakdown."""
        if self.portfolio_df.empty:
            return pd.DataFrame()
        
        portfolio = self.portfolio_df['Portfolio Value']
        monthly = portfolio.resample('M').last()
        monthly_returns = monthly.pct_change() * 100
        
        # Create pivot table
        monthly_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        }).dropna()
        
        if monthly_df.empty:
            return pd.DataFrame()
        
        pivot = monthly_df.pivot(index='Year', columns='Month', values='Return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
        
        # Add Total column
        pivot['Total'] = pivot.sum(axis=1)
        
        return pivot
