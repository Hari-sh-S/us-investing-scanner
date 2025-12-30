import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from us_portfolio_engine import USPortfolioEngine, DataCache
from scoring import ScoreParser
from us_universe import (get_all_universe_names, get_universe, 
                         get_broad_market_universes, get_sectoral_universes,
                         get_cap_based_universes, get_thematic_universes)
from report_generator import create_excel_with_charts, create_pdf_report, prepare_complete_log_data
import datetime
import io
import time
import json
from pathlib import Path
from monte_carlo import MonteCarloSimulator, PortfolioMonteCarloSimulator, extract_trade_pnls, extract_monthly_returns

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="US Investing Scanner",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Show loading indicator immediately for wake-up
with st.spinner("🔄 App is waking up... Please wait..."):
    pass

# Initialize session state for backtest logs
BACKTEST_LOG_FILE = Path("us_backtest_logs.json")

@st.cache_data(ttl=3600)
def load_backtest_logs_cached():
    """Load backtest logs from file with caching."""
    if BACKTEST_LOG_FILE.exists():
        try:
            with open(BACKTEST_LOG_FILE, 'r') as f:
                logs_data = json.load(f)
                return logs_data
        except Exception as e:
            print(f"Error loading logs: {e}")
            return []
    return []

def load_backtest_logs():
    """Load backtest logs from file."""
    return load_backtest_logs_cached()

def save_backtest_logs(logs):
    """Save backtest logs to file."""
    try:
        serializable_logs = []
        for log in logs:
            serializable_log = {
                'timestamp': log['timestamp'],
                'name': log['name'],
                'config': log['config'],
                'metrics': log['metrics'],
                'portfolio_values': log.get('portfolio_values', []),
                'trades': log.get('trades', []),
                'monthly_returns': log.get('monthly_returns', {})
            }
            serializable_logs.append(serializable_log)
        
        with open(BACKTEST_LOG_FILE, 'w') as f:
            json.dump(serializable_logs, f, indent=2, default=str)
        load_backtest_logs_cached.clear()
    except Exception as e:
        print(f"Error saving logs: {e}")

@st.cache_data(ttl=86400)
def get_cached_universe_names():
    """Cache universe names to speed up app loading."""
    try:
        return sorted(get_all_universe_names())
    except Exception:
        return ["S&P 500", "NASDAQ 100", "DOW 30"]

# Initialize session state
try:
    if 'backtest_logs' not in st.session_state:
        st.session_state.backtest_logs = load_backtest_logs()
    if 'backtest_engines' not in st.session_state:
        st.session_state.backtest_engines = {}
    if 'app_ready' not in st.session_state:
        st.session_state.app_ready = True
except Exception as e:
    st.error(f"Error initializing app: {e}. Please refresh the page.")
    st.session_state.backtest_logs = []
    st.session_state.backtest_engines = {}
    st.session_state.app_ready = True

# CSS
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: white;
        border-radius: 6px;
        padding: 0 20px;
        border: 1px solid #ddd;
        color: #1a1a1a !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white !important;
        font-weight: 600;
    }
    .progress-text {
        font-size: 16px;
        color: #ffffff;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .stock-name {
        color: #007bff;
        font-weight: 700;
        font-size: 18px;
        text-shadow: 0 0 10px rgba(0, 123, 255, 0.5);
    }
    .time-remaining {
        font-size: 14px;
        color: #aaaaaa;
        margin-left: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header - compact with last git commit timestamp
def get_last_update_time():
    """Get last git commit timestamp"""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cd', '--date=format:%d %b %H:%M'],
            capture_output=True, text=True, cwd='.'
        )
        if result.returncode == 0:
            return result.stdout.strip() + " ET"
    except:
        pass
    return "Unknown"

last_update = get_last_update_time()
st.markdown(f"### 🇺🇸 US Investing Scanner <span style='font-size: 14px; color: #888;'>Updated: {last_update}</span>", unsafe_allow_html=True)

# Main Tabs
main_tabs = st.tabs(["Backtest", "Backtest Logs", "Data Download"])

# ==================== TAB 1: BACKTEST ====================
with main_tabs[0]:
    col_config, col_scoring = st.columns([1, 1.2])
    
    with col_config:
        st.subheader("Configuration")
        
        # ===== BASIC SETTINGS (always visible) =====
        st.markdown("**Universe**")
        
        # Get all available universes
        all_universes = sorted(get_all_universe_names()) + ["Custom"]
        
        selected_universe = st.selectbox(
            "Select", 
            all_universes,
            label_visibility="collapsed"
        )
        
        if selected_universe == "Custom":
            custom_input = st.text_input("Stocks (comma-separated)", "AAPL, MSFT, GOOGL, AMZN, NVDA")
            universe = [s.strip() for s in custom_input.split(',')]
        else:
            universe = get_universe(selected_universe)
            st.caption(f"{len(universe)} stocks")
        
        # Capital, Stocks, Exit Rank in compact rows
        st.markdown("**Portfolio Settings**")
        cap_col1, cap_col2 = st.columns(2)
        with cap_col1:
            initial_capital = st.number_input("Capital ($)", 10000, 100000000, 100000, 10000)
        with cap_col2:
            num_stocks = st.number_input("Stocks", 1, 50, 5)
        
        exit_col1, exit_col2 = st.columns(2)
        with exit_col1:
            exit_rank = st.number_input("Exit Rank", num_stocks, 200, num_stocks * 2, 
                                        help="Stocks exit if they fall below this rank")
        with exit_col2:
            reinvest_profits = st.checkbox("Reinvest Profits", value=True)
        
        use_historical_universe = st.checkbox("Historical Universe (Beta)", value=False,
                                             help="Use point-in-time index constituents to avoid survivorship bias")
        
        # ===== TIME PERIOD & REBALANCING (in expander) =====
        with st.expander("📅 Time Period & Rebalancing", expanded=False):
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Start Date", datetime.date(2020, 1, 1))
            with date_col2:
                end_date = st.date_input("End Date", datetime.date.today())
            
            rebal_freq_options = ["Weekly", "Every 2 Weeks", "Monthly", "Bi-Monthly", "Quarterly", "Half-Yearly", "Annually"]
            rebalance_label = st.selectbox("Frequency", rebal_freq_options, index=2)
            
            if rebalance_label == "Weekly":
                rebal_day = st.selectbox("Rebalance Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                rebalance_date = None
            elif rebalance_label == "Every 2 Weeks":
                rebal_day = st.selectbox("Rebalance Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
                rebalance_date = None
            else:  # Monthly and above
                rebalance_date = st.number_input("Rebalance Date (1-30)", 1, 30, 1,
                                                help="Day of month to rebalance portfolio")
                rebal_day = None
            
            alt_day_option = st.selectbox("If Holiday", 
                                         ["Previous Day", "Next Day"],
                                         index=1,
                                         help="If rebalance day is holiday, use this option")

        
        # ===== POSITION SIZING (in expander) =====
        with st.expander("⚖️ Position Sizing", expanded=False):
            sizing_methods = ["Equal Weight", "Inverse Volatility", "Score-Weighted", "Risk Parity"]
            sizing_method = st.selectbox("Sizing Method", sizing_methods, index=0)
            
            use_max_cap = st.checkbox("Max Position Cap", value=False)
            max_cap_pct = st.number_input("Max Cap (%)", 1.0, 100.0, 25.0) if use_max_cap else None
            
            position_sizing_config = {
                'method': sizing_method,
                'max_cap_pct': max_cap_pct if use_max_cap else None
            }
        
        # ===== REGIME FILTER (in expander) =====
        with st.expander("🛡️ Regime Filter", expanded=False):
            use_regime_filter = st.checkbox("Enable Regime Filter", value=False)
            
            regime_config = None
            if use_regime_filter:
                regime_type = st.selectbox("Regime Filter Type", 
                                          ["EMA", "MACD", "SUPERTREND", "EQUITY", "EQUITY_MA"],
                                          help="EQUITY_MA: Uses moving average of your equity curve")
                
                # Initialize defaults
                recovery_dd = None
                ma_period = None
                
                if regime_type == "EMA":
                    ema_period = st.selectbox("EMA Period", [34, 68, 100, 150, 200])
                    regime_value = ema_period
                elif regime_type == "MACD":
                    macd_preset = st.selectbox("MACD Settings", 
                                              ["35-70-12", "50-100-15", "75-150-12"])
                    regime_value = macd_preset
                elif regime_type == "SUPERTREND":
                    st_preset = st.selectbox("SuperTrend (Period-Multiplier)", 
                                            ["1-1", "1-2", "1-2.5"])
                    regime_value = st_preset
                elif regime_type == "EQUITY":
                    eq_col1, eq_col2 = st.columns(2)
                    with eq_col1:
                        realized_sl = st.number_input("DD SL % (Trigger)", 1, 50, 10,
                                                      help="Sell when drawdown exceeds this %")
                    with eq_col2:
                        recovery_dd = st.number_input("Recovery DD %", 0, 49, 5,
                                                      help="Re-enter when drawdown below this %")
                    regime_value = realized_sl
                else:  # EQUITY_MA
                    ma_period = st.selectbox("Equity Curve MA Period", 
                                            [20, 30, 50, 100, 200],
                                            index=2,
                                            help="Reduce exposure when equity falls below this MA")
                    regime_value = ma_period
                
                # Regime action
                regime_action = st.selectbox("Regime Filter Action",
                                            ["Go Cash", "Half Portfolio"],
                                            help="Action when regime filter triggers")
                
                # Index selection (only for non-EQUITY types)
                if regime_type not in ["EQUITY", "EQUITY_MA"]:
                    regime_indices = ["S&P 500", "NASDAQ 100", "DOW 30", "Russell 2000", "VIX"]
                    regime_index = st.selectbox("Regime Filter Index", regime_indices)
                else:
                    regime_index = None
                
                regime_config = {
                    'type': regime_type,
                    'value': regime_value,
                    'action': regime_action,
                    'index': regime_index,
                    'recovery_dd': recovery_dd,
                    'ma_period': ma_period if regime_type == "EQUITY_MA" else None
                }
                
                # Uncorrelated Asset
                st.markdown("---")
                use_uncorrelated = st.checkbox("Invest in Uncorrelated Asset", value=False,
                                              help="Allocate to uncorrelated asset when regime triggers")
                
                uncorrelated_config = None
                if use_uncorrelated:
                    unc_col1, unc_col2 = st.columns(2)
                    with unc_col1:
                        asset_type = st.text_input("Asset Ticker", "GLD",
                                                  help="e.g., GLD for Gold ETF, TLT for Bonds")
                    with unc_col2:
                        allocation_pct = st.number_input("Alloc %", 1, 100, 100)
                    
                    uncorrelated_config = {
                        'asset': asset_type,
                        'allocation_pct': allocation_pct
                    }
            else:
                uncorrelated_config = None
    
    with col_scoring:
        st.subheader("Scoring Console")
        
        parser = ScoreParser()
        examples = parser.get_example_formulas()
        
        template = st.selectbox("Template", ["Custom"] + list(examples.keys()))
        default = examples.get(template, "6 Month Performance")
        
        formula = st.text_area("Scoring Formula", default, height=120)
        
        valid, msg = parser.validate_formula(formula)
        if valid:
            st.success("✅ " + msg)
        else:
            st.error("❌ " + msg)
        
        # Compact metrics reference in collapsible expander
        with st.expander("📖 Available Metrics", expanded=False):
            st.caption("💡 **Tip:** Use periods 1-12 months, e.g. `7 Month Performance`, `10 Month Sharpe`")
            
            metric_groups = parser.metric_groups if hasattr(parser, 'metric_groups') else {}
            
            perf = metric_groups.get('Performance', [])
            st.markdown("**Performance:** " + " • ".join(perf[:3]) + "...")
            
            vol = metric_groups.get('Volatility', [])
            st.markdown("**Volatility:** " + " • ".join(vol[:3]) + "...")
            
            dsv = metric_groups.get('Downside Volatility', [])
            if dsv:
                st.markdown("**Downside Vol:** " + " • ".join(dsv[:3]) + "...")
            
            mdd = metric_groups.get('Max Drawdown', [])
            if mdd:
                st.markdown("**Max Drawdown:** " + " • ".join(mdd[:3]) + "...")
            
            sharpe = metric_groups.get('Sharpe Ratio', [])[:2]
            sortino = metric_groups.get('Sortino Ratio', [])[:2]
            calmar = metric_groups.get('Calmar Ratio', [])[:2]
            risk_adj = sharpe + sortino + calmar
            if risk_adj:
                st.markdown("**Risk-Adjusted:** " + " • ".join(risk_adj[:4]) + "...")
        
        st.markdown("---")
        run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    # Results Section
    if run_btn:
        if not valid:
            st.error("Fix formula first")
        else:
            # Initialize tracking variables
            start_time = time.time()
            processed_count = [0]
            total_count = len(universe)
            
            def progress_callback(current, total, ticker):
                processed_count[0] = current

                elapsed = time.time() - start_time
                elapsed_mins = int(elapsed // 60)
                elapsed_secs = int(elapsed % 60)

                if processed_count[0] > 0:
                    avg_time_per_stock = elapsed / processed_count[0]
                    remaining_stocks = total - processed_count[0]
                    time_remaining_sec = avg_time_per_stock * remaining_stocks

                    remaining_mins = int(time_remaining_sec // 60)
                    remaining_secs = int(time_remaining_sec % 60)
                    time_str = f"{remaining_mins:02d}:{remaining_secs:02d}"
                else:
                    time_str = "Calculating..."

                progress = min(processed_count[0] / total, 1.0)
                prog_bar.progress(progress)

                pct = (processed_count[0] / total * 100) if total > 0 else 0
                status_container.markdown(f"""
                <div style="padding: 10px; background: rgba(0,0,0,0.1); border-radius: 5px;">
                    <div style="font-size: 16px; font-weight: bold;">📊 {ticker}</div>
                    <div style="margin-top: 5px;">
                        Progress: {processed_count[0]}/{total} ({pct:.1f}%)
                    </div>
                    <div style="margin-top: 5px;">
                        ⏱️ Remaining: {time_str} | ⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.spinner("Initializing backtest..."):
                prog_bar = st.progress(0)
                status_container = st.empty()
                
                engine = USPortfolioEngine(universe, start_date, end_date, initial_capital)
                if engine.fetch_data(progress_callback=progress_callback):
                    prog_bar.empty()
                    status_container.empty()
                    
                    with st.spinner("Running strategy simulation..."):
                        # Build rebalance config
                        rebal_config = {
                            'frequency': rebalance_label,
                            'date': rebalance_date,
                            'day': rebal_day,
                            'alt_day': alt_day_option
                        }
                        
                        engine.run_rebalance_strategy(
                            formula, 
                            num_stocks,
                            exit_rank,
                            rebal_config,
                            regime_config,
                            uncorrelated_config,
                            reinvest_profits,
                            position_sizing_config=position_sizing_config,
                            historical_universe_config={'use_historical': use_historical_universe}
                        )
                        metrics = engine.get_metrics()
                        
                        # Store in session_state
                        st.session_state['backtest_engine'] = engine
                        st.session_state['backtest_metrics'] = metrics
                        st.session_state['backtest_start_date'] = start_date
                        st.session_state['backtest_end_date'] = end_date
                    
                    if metrics:
                        # Prepare complete log data
                        complete_log_data = prepare_complete_log_data(
                            {
                                'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                                'initial_capital': initial_capital,
                                'universe_name': selected_universe,
                                'num_stocks': num_stocks,
                                'exit_rank': exit_rank,
                                'rebalance_freq': rebalance_label,
                                'start_date': start_date.strftime('%Y-%m-%d'),
                                'end_date': end_date.strftime('%Y-%m-%d'),
                                'regime_config': regime_config if regime_config else {},
                                'uncorrelated_config': uncorrelated_config if uncorrelated_config else {},
                                'formula': formula
                            },
                            metrics,
                            engine
                        )
                        
                        # Save to logs
                        backtest_log = {
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'name': f"Backtest_{datetime.datetime.now().strftime('%m%d_%H%M')}",
                            'config': complete_log_data['config'],
                            'metrics': metrics,
                            'portfolio_values': complete_log_data['portfolio_values'],
                            'trades': complete_log_data['trades'],
                            'monthly_returns': complete_log_data['monthly_returns']
                        }
                        st.session_state.backtest_logs.append(backtest_log)
                        save_backtest_logs(st.session_state.backtest_logs)
                        
                        # Store current backtest data
                        st.session_state['current_backtest'] = {
                            'engine': engine,
                            'metrics': metrics,
                            'backtest_log': backtest_log,
                            'start_date': start_date,
                            'end_date': end_date
                        }
                        st.session_state['current_backtest_active'] = True
                        
                        st.markdown("---")
                        
                        # Action buttons
                        col_h, col_excel, col_pdf = st.columns([3, 1, 1])
                        with col_h:
                            st.subheader("Backtest Results")
                        with col_excel:
                            excel_data = create_excel_with_charts(backtest_log['config'], metrics, engine)
                            st.download_button(
                                label="📥 Excel",
                                data=excel_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        with col_pdf:
                            pdf_data = create_pdf_report(backtest_log['config'], metrics, engine)
                            st.download_button(
                                label="📄 PDF",
                                data=pdf_data,
                                file_name=f"backtest_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                        
                        # Result tabs
                        result_tabs = st.tabs(["Performance", "Charts", "Monthly Report", "Monte Carlo", "Equity Regime", "Trade History"])
                        
                        with result_tabs[0]:
                            st.markdown("### Key Performance Indicators")
                            
                            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
                            
                            kpi_col1.metric("Final Value", f"${metrics['Final Value']:,.0f}")
                            kpi_col1.metric("Total Return", f"${metrics['Total Return']:,.0f}")
                            kpi_col1.metric("Return %", f"{metrics['Return %']:.2f}%")
                            
                            kpi_col2.metric("CAGR %", f"{metrics['CAGR %']:.2f}%")
                            kpi_col2.metric("Max Drawdown %", f"{metrics['Max Drawdown %']:.2f}%")
                            kpi_col2.metric("Volatility %", f"{metrics.get('Volatility %', 0):.2f}%")
                            
                            kpi_col3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
                            kpi_col3.metric("Win Rate %", f"{metrics['Win Rate %']:.2f}%")
                            kpi_col3.metric("Total Trades", metrics['Total Trades'])
                            
                            kpi_col4.metric("Avg Trade/Year", f"{metrics['Total Trades'] / max(1, (end_date - start_date).days / 365.25):.1f}")
                            kpi_col4.metric("Expectancy", f"${metrics.get('Expectancy', 0):,.0f}")
                            
                            # Advanced Metrics Row
                            st.markdown("---")
                            st.markdown("**📊 Advanced Metrics**")
                            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
                            
                            adv_col1.metric("Max Consecutive Wins", metrics.get('Max Consecutive Wins', 0))
                            adv_col1.metric("Max Consecutive Losses", metrics.get('Max Consecutive Losses', 0))
                            
                            adv_col2.metric("Avg Win", f"${metrics.get('Avg Win', 0):,.0f}")
                            adv_col2.metric("Avg Loss", f"${metrics.get('Avg Loss', 0):,.0f}")
                            
                            adv_col3.metric("Days to Recover from DD", metrics.get('Days to Recover from DD', 0))
                            adv_col3.metric("Trades to Recover from DD", metrics.get('Trades to Recover from DD', 0))
                            
                            adv_col4.metric("Total Turnover", f"${metrics.get('Total Turnover', 0):,.0f}")
                            adv_col4.metric("Total Charges", f"${metrics.get('Total Charges', 0):,.0f}")
                            
                            # US Charges Breakdown Expander
                            with st.expander("📋 US Trading Charges Breakdown"):
                                charges_col1, charges_col2 = st.columns(2)
                                charges_col1.write(f"**SEC Fees:** ${metrics.get('SEC Fees', 0):,.2f}")
                                charges_col1.write(f"**FINRA TAF:** ${metrics.get('FINRA TAF', 0):,.2f}")
                                charges_col2.write(f"**Total Turnover:** ${metrics.get('Total Turnover', 0):,.2f}")
                                charges_col2.write(f"**Total Charges:** ${metrics.get('Total Charges', 0):,.2f}")
                        
                        with result_tabs[1]:
                            st.markdown("### Performance Charts")
                            
                            # Equity Curve
                            fig_equity = go.Figure()
                            fig_equity.add_trace(go.Scatter(
                                x=engine.portfolio_df.index,
                                y=engine.portfolio_df['Portfolio Value'],
                                fill='tozeroy',
                                line_color='#007bff',
                                name='Portfolio Value'
                            ))
                            fig_equity.update_layout(
                                title="Equity Curve",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                height=400,
                                margin=dict(l=0,r=0,t=40,b=0),
                                showlegend=False,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_equity, use_container_width=True)
                            
                            # Drawdown Chart
                            running_max = engine.portfolio_df['Portfolio Value'].cummax()
                            dd = (engine.portfolio_df['Portfolio Value'] - running_max) / running_max * 100
                            
                            fig_dd = go.Figure()
                            fig_dd.add_trace(go.Scatter(
                                x=dd.index,
                                y=dd,
                                fill='tozeroy',
                                line_color='#dc3545',
                                name='Drawdown'
                            ))
                            fig_dd.update_layout(
                                title="Drawdown Analysis",
                                xaxis_title="Date",
                                yaxis_title="Drawdown %",
                                height=350,
                                margin=dict(l=0,r=0,t=40,b=0),
                                showlegend=False,
                                template='plotly_white'
                            )
                            st.plotly_chart(fig_dd, use_container_width=True)

                        with result_tabs[2]:
                            st.markdown("### 📅 Monthly Performance Report")
                            if not engine.portfolio_df.empty:
                                monthly_returns = engine.get_monthly_returns()
                                if not monthly_returns.empty:
                                    # Highlight positive/negative months
                                    def highlight_returns(val):
                                        if pd.isna(val) or val == 0: return ""
                                        color = "#28a745" if val > 0 else "#dc3545"
                                        return f"color: {color}; font-weight: bold;"

                                    st.dataframe(
                                        monthly_returns.style.applymap(highlight_returns),
                                        use_container_width=True,
                                        height=400
                                    )
                                    
                                    # Monthly Stats
                                    st.markdown("---")
                                    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                                    
                                    all_m_returns = monthly_returns.drop(columns=['Total']).values.flatten()
                                    all_m_returns = all_m_returns[~pd.isna(all_m_returns)]
                                    
                                    if len(all_m_returns) > 0:
                                        m_col1.metric("Positive Months", f"{(all_m_returns > 0).sum()}")
                                        m_col2.metric("Negative Months", f"{(all_m_returns < 0).sum()}")
                                        m_col3.metric("Best Month", f"{max(all_m_returns):.2f}%")
                                        m_col4.metric("Worst Month", f"{min(all_m_returns):.2f}%")
                            else:
                                st.info("No monthly data available")

                        with result_tabs[3]:
                            st.markdown("### 🎲 Monte Carlo Risk Analysis")
                            
                            mc_col1, mc_col2 = st.columns([1, 2])
                            
                            with mc_col1:
                                mc_type = st.radio("Simulation Level", ["Trade-Level", "Portfolio-Level"], horizontal=True)
                                mc_method = st.selectbox("Method", ["reshuffle", "resample"], help="Reshuffle: Permute original outcomes. Resample: Bootstrap with replacement.")
                                run_mc = st.button("🔄 Run Simulation", type="primary", use_container_width=True)
                            
                            if run_mc:
                                with st.spinner("Running 1000 simulations..."):
                                    if mc_type == "Trade-Level":
                                        pnls = extract_trade_pnls(engine.trades_df)
                                        if len(pnls) < 10:
                                            st.warning("Insufficient trades for robust simulation (min 10 recommended)")
                                        sim = MonteCarloSimulator(pnls, initial_capital)
                                    else:
                                        m_returns = extract_monthly_returns(engine.trades_df, initial_capital)
                                        sim = PortfolioMonteCarloSimulator(m_returns, initial_capital)
                                    
                                    results = sim.run_simulations(method=mc_method)
                                    interpretations = sim.get_interpretation()
                                    
                                    # MC Metrics
                                    st.markdown("#### Simulation Key Results")
                                    res_col1, res_col2, res_col3 = st.columns(3)
                                    
                                    res_col1.metric("Max DD (95% Conf)", f"{results['mc_max_dd_95']:.1f}%")
                                    res_col2.metric("Worst Case DD", f"{results['mc_max_dd_worst']:.1f}%")
                                    res_col3.metric("Prob. of Ruin (10%)", f"{results['ruin_probability_10']:.1f}%")
                                    
                                    # Plot MC Curves
                                    fig_mc = go.Figure()
                                    for i in range(min(50, len(results['sample_curves']))):
                                        fig_mc.add_trace(go.Scatter(
                                            y=results['sample_curves'][i], 
                                            mode='lines', 
                                            line=dict(width=0.5, color='rgba(0,123,255,0.1)'),
                                            showlegend=False
                                        ))
                                    
                                    # Add median curve
                                    median_curve = np.median(results['sample_curves'], axis=0)
                                    fig_mc.add_trace(go.Scatter(
                                        y=median_curve, 
                                        mode='lines', 
                                        line=dict(width=2, color='#007bff'),
                                        name='Median Projection'
                                    ))
                                    
                                    fig_mc.update_layout(title="Monte Carlo Equity Projections (50 Samples)", xaxis_title="Time", yaxis_title="Equity ($)", template="plotly_white")
                                    st.plotly_chart(fig_mc, use_container_width=True)
                                    
                                    # Risk Assessment
                                    st.markdown("#### 🛡️ Risk Assessment")
                                    for key, text in interpretations.items():
                                        st.info(f"**{key.replace('_', ' ').title()}:** {text}")

                        with result_tabs[4]:
                            st.markdown("### 🛡️ Equity Regime Analysis")
                            regime_analysis = engine.get_equity_regime_analysis()
                            
                            if regime_analysis:
                                comp_df = regime_analysis['comparison_df']
                                triggers = regime_analysis['trigger_events']
                                
                                # Plot Comparison
                                fig_regime = go.Figure()
                                fig_regime.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Actual'], name="With Regime Filter", line=dict(color="#007bff")))
                                fig_regime.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Theoretical'], name="Theoretical (No Filter)", line=dict(color="#6c757d", dash='dot')))
                                
                                # Add Vertical Lines for Triggers
                                for event in triggers:
                                    color = "red" if event['type'] == 'trigger' else "green"
                                    fig_regime.add_vline(x=event['date'], line_dash="dash", line_color=color, opacity=0.5)
                                
                                fig_regime.update_layout(title="Actual vs Theoretical Equity Curve", xaxis_title="Date", yaxis_title="Equity ($)", template="plotly_white")
                                st.plotly_chart(fig_regime, use_container_width=True)
                                
                                # Trigger Table
                                if triggers:
                                    st.markdown("#### Regime Trigger Events")
                                    trigger_df = pd.DataFrame(triggers)
                                    st.table(trigger_df)
                                else:
                                    st.info("No regime trigger events recorded during this period.")
                            else:
                                st.info("Run a backtest with a Regime Filter enabled to see analysis here.")

                        with result_tabs[5]:
                            st.markdown("### 📜 Trade History")
                            if not engine.trades_df.empty:
                                trades_df = engine.trades_df.copy()
                                buy_trades = trades_df[trades_df['Action'] == 'BUY'].copy()
                                sell_trades = trades_df[trades_df['Action'] == 'SELL'].copy()
                                
                                consolidated_trades = []
                                
                                for _, sell in sell_trades.iterrows():
                                    ticker = sell['Ticker']
                                    sell_date = sell['Date']
                                    
                                    prev_buys = buy_trades[
                                        (buy_trades['Ticker'] == ticker) & 
                                        (buy_trades['Date'] < sell_date)
                                    ]
                                    
                                    if not prev_buys.empty:
                                        buy = prev_buys.iloc[-1]
                                        buy_price = float(buy['Price'])
                                        sell_price = float(sell['Price'])
                                        shares = int(buy['Shares'])
                                        roi = ((sell_price - buy_price) / buy_price) * 100
                                        
                                        consolidated_trades.append({
                                            'Stock': ticker,
                                            'Buy Date': pd.to_datetime(buy['Date']).strftime('%Y-%m-%d'),
                                            'Buy Price': round(buy_price, 2),
                                            'Exit Date': pd.to_datetime(sell_date).strftime('%Y-%m-%d'),
                                            'Exit Price': round(sell_price, 2),
                                            'Shares': shares,
                                            'ROI %': round(roi, 2)
                                        })
                                
                                if consolidated_trades:
                                    trade_display = pd.DataFrame(consolidated_trades)
                                    
                                    def color_roi(val):
                                        if val > 0:
                                            return 'color: #28a745; font-weight: bold'
                                        elif val < 0:
                                            return 'color: #dc3545; font-weight: bold'
                                        return ''
                                    
                                    styled_trades = trade_display.style.applymap(
                                        color_roi, subset=['ROI %']
                                    )
                                    st.dataframe(styled_trades, use_container_width=True, height=400)
                                    
                                    st.markdown("---")
                                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                    stat_col1.metric("Total Trades", len(consolidated_trades))
                                    profitable = len([t for t in consolidated_trades if t['ROI %'] > 0])
                                    stat_col2.metric("Profitable", f"{profitable} ({profitable/len(consolidated_trades)*100:.1f}%)")
                                    avg_roi = sum(t['ROI %'] for t in consolidated_trades) / len(consolidated_trades)
                                    stat_col3.metric("Avg ROI", f"{avg_roi:.2f}%")
                                    best_trade = max(consolidated_trades, key=lambda x: x['ROI %'])
                                    stat_col4.metric("Best Trade", f"{best_trade['Stock']} ({best_trade['ROI %']:.1f}%)")
                                else:
                                    st.info("No completed trades to display")
                            else:
                                st.info("No trades executed")
                    else:
                        st.warning("No trades generated")
                else:
                    st.error("Data fetch failed")
    
    # BENCHMARK COMPARISON
    if st.session_state.get('current_backtest_active') and 'current_backtest' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Benchmark Comparison")
        
        stored_data = st.session_state['current_backtest']
        stored_engine = stored_data['engine']
        bt_start = stored_data['start_date']
        bt_end = stored_data['end_date']
        
        # US index mappings
        yahoo_index_map = {
            "S&P 500": "^GSPC",
            "NASDAQ 100": "^NDX",
            "NASDAQ Composite": "^IXIC",
            "DOW 30": "^DJI",
            "Russell 2000": "^RUT",
            "S&P MidCap 400": "^MID",
            "S&P SmallCap 600": "^SML",
            "VIX": "^VIX",
        }
        
        benchmark_options = list(yahoo_index_map.keys())
        
        stored_benchmark = st.session_state.get('benchmark_selection', 'S&P 500')
        try:
            default_idx = benchmark_options.index(stored_benchmark)
        except ValueError:
            default_idx = 0
        
        selected_benchmark = st.selectbox(
            "Select Benchmark Index", 
            benchmark_options,
            index=default_idx,
            key="standalone_benchmark_selector"
        )
        st.session_state['benchmark_selection'] = selected_benchmark
        
        try:
            import yfinance as yf
            benchmark_ticker = yahoo_index_map.get(selected_benchmark, "^GSPC")
            benchmark_data = yf.download(benchmark_ticker, start=bt_start, end=bt_end, progress=False)
            
            if not benchmark_data.empty:
                portfolio_values = stored_engine.portfolio_df['Portfolio Value']
                portfolio_norm = (portfolio_values / portfolio_values.iloc[0] - 1) * 100
                
                benchmark_close = benchmark_data['Close']
                if isinstance(benchmark_close, pd.DataFrame):
                    benchmark_close = benchmark_close.iloc[:, 0]
                benchmark_norm = (benchmark_close / benchmark_close.iloc[0] - 1) * 100
                
                # Calculate drawdowns
                portfolio_cummax = portfolio_values.cummax()
                portfolio_dd = ((portfolio_values - portfolio_cummax) / portfolio_cummax) * 100
                benchmark_cummax = benchmark_close.cummax()
                benchmark_dd = ((benchmark_close - benchmark_cummax) / benchmark_cummax) * 100
                
                # PnL Comparison Chart
                fig_pnl = go.Figure()
                fig_pnl.add_trace(go.Scatter(x=portfolio_norm.index, y=portfolio_norm, name="Portfolio", line=dict(color="#007bff", width=2)))
                fig_pnl.add_trace(go.Scatter(x=benchmark_norm.index, y=benchmark_norm, name=selected_benchmark, line=dict(color="#ffc107", width=2)))
                fig_pnl.update_layout(title=f"Cumulative Returns: Portfolio vs {selected_benchmark}", xaxis_title="Date", yaxis_title="Return (%)", height=400, template="plotly_dark")
                st.plotly_chart(fig_pnl, use_container_width=True)
                
                # Summary
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Portfolio Return", f"{portfolio_norm.iloc[-1]:.1f}%")
                col2.metric(f"{selected_benchmark} Return", f"{benchmark_norm.iloc[-1]:.1f}%")
                col3.metric("Portfolio Max DD", f"{portfolio_dd.min():.1f}%")
                col4.metric(f"{selected_benchmark} Max DD", f"{benchmark_dd.min():.1f}%")
                
                alpha = portfolio_norm.iloc[-1] - benchmark_norm.iloc[-1]
                if alpha > 0:
                    st.success(f"🎯 **Alpha Generated: +{alpha:.1f}%**")
                else:
                    st.warning(f"📉 **Alpha: {alpha:.1f}%**")
            else:
                st.warning(f"Could not fetch data for {selected_benchmark}")
        except Exception as e:
            st.error(f"Error loading benchmark: {e}")

# ==================== TAB 2: BACKTEST LOGS ====================
with main_tabs[1]:
    st.subheader("Backtest History")
    
    if not st.session_state.backtest_logs:
        st.info("No backtest logs yet. Run a backtest to see results here.")
    else:
        st.markdown(f"**Total Backtests:** {len(st.session_state.backtest_logs)}")
        
        for idx, log in enumerate(reversed(st.session_state.backtest_logs)):
            with st.expander(f"📊 {log['name']} - {log['timestamp']}"):
                st.markdown(f"**Universe:** {log['config']['universe_name']}")
                st.markdown(f"**Period:** {log['config']['start_date']} to {log['config']['end_date']}")
                st.markdown(f"**Formula:** `{log['config']['formula']}`")
                
                metrics = log['metrics']
                st.markdown(f"**Final Value:** ${metrics['Final Value']:,.0f} | **CAGR:** {metrics['CAGR %']:.2f}% | **Sharpe:** {metrics['Sharpe Ratio']:.2f} | **Win Rate:** {metrics['Win Rate %']:.1f}%")
                
                if log.get('trades'):
                    st.caption(f"📈 {len(log['trades'])} trades recorded")
                if log.get('portfolio_values'):
                    st.caption(f"📊 {len(log['portfolio_values'])} daily values stored")
                
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    excel_data = create_excel_with_charts(log['config'], metrics)
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"{log['name']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"excel_{idx}"
                    )
                with dl_col2:
                    pdf_data = create_pdf_report(log['config'], metrics)
                    st.download_button(
                        label="📄 Download PDF",
                        data=pdf_data,
                        file_name=f"{log['name']}.pdf",
                        mime="application/pdf",
                        key=f"pdf_{idx}"
                    )
        
        if st.button("🗑️ Clear All Logs"):
            st.session_state.backtest_logs = []
            st.session_state.backtest_engines = {}
            save_backtest_logs([])
            st.rerun()

# ==================== TAB 3: DATA DOWNLOAD ====================
with main_tabs[2]:
    st.subheader("📥 Data Download")
    st.markdown("Download historical data for all US universes. This is a one-time setup - data will be cached for fast backtests.")

    cache = DataCache()
    cache_info = cache.get_cache_info()

    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Cached Stocks", cache_info['total_files'])
    col2.metric("Storage Used", f"{cache_info['total_size_mb']:.2f} MB")
    
    with col3:
        st.write("")
        if st.button("🗑️ Clear All Cache", type="secondary", key="clear_cache"):
            cache.clear()
            st.success("✅ Cache cleared! Please refresh the page.")
            st.rerun()

    st.markdown("---")
    
    # Download All Data Button
    st.markdown("### 🔽 Download All Universe Data")
    st.info("This will download and cache data for ALL stocks across ALL US universes. Takes ~5-10 minutes.")
    
    col_clear, col_download = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Cache First", key="clear_data_cache_btn"):
            cache.clear()
            st.success("✅ Cache cleared! Now click 'Download All Data' to get fresh data.")
            st.rerun()
    
    with col_download:
        download_clicked = st.button("📥 Download All Data", type="primary", key="download_all_data_btn")

    if download_clicked:
        all_tickers = set()
        all_universe_names = get_all_universe_names()

        for universe_name in all_universe_names:
            universe = get_universe(universe_name)
            all_tickers.update(universe)

        all_tickers = sorted(list(all_tickers))

        st.markdown(f"### Downloading {len(all_tickers)} unique stocks...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        def download_progress(current, total, ticker, remaining_seconds):
            pct = (current / total) if total > 0 else 0
            progress_bar.progress(min(pct, 1.0))

            mins = int(remaining_seconds // 60)
            secs = int(remaining_seconds % 60)
            elapsed = time.time() - start_time
            elapsed_mins = int(elapsed // 60)
            elapsed_secs = int(elapsed % 60)

            status_text.markdown(f"""
            <div style="padding: 10px; background: rgba(0,123,255,0.1); border-radius: 5px;">
                <div style="font-size: 16px; font-weight: bold;">📊 {ticker}</div>
                <div>Progress: {current}/{total} ({pct*100:.1f}%)</div>
                <div>⏱️ Remaining: {mins:02d}:{secs:02d} | ⏰ Elapsed: {elapsed_mins:02d}:{elapsed_secs:02d}</div>
            </div>
            """, unsafe_allow_html=True)

        temp_engine = USPortfolioEngine(all_tickers, datetime.date(2020, 1, 1), datetime.date.today())
        success_count = temp_engine.download_and_cache_universe(all_tickers, download_progress, None)

        progress_bar.empty()
        status_text.empty()

        total_time = time.time() - start_time
        st.success(f"✅ Downloaded {success_count}/{len(all_tickers)} stocks in {int(total_time)}s!")
        st.balloons()
