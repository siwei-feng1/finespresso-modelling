import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from utils.backtesting.backtest_price_util import create_price_moves, get_intraday_prices
from utils.logging.log_util import get_logger

logger = get_logger(__name__)

def calculate_position_size(capital, position_size_pct):
    return capital * position_size_pct

def calculate_shares(position_size, entry_price):
    return int(position_size / entry_price)

def check_exit(intraday_data, entry_price, target_price, stop_price, is_long, default_exit_time, default_exit_price):
    """
    Check if stop loss or target price is hit, otherwise exit at market close
    
    Args:
        intraday_data: DataFrame with intraday price data
        entry_price: Trade entry price
        target_price: Target exit price
        stop_price: Stop loss price
        is_long: Boolean indicating if trade is long
        default_exit_time: Market close time
        default_exit_price: Market close price
    
    Returns:
        tuple: (exit_price, exit_time, hit_target, hit_stop)
    """
    exit_price = default_exit_price
    exit_time = default_exit_time
    hit_target = False
    hit_stop = False
    
    for idx, price_data in intraday_data.iterrows():
        current_price = float(price_data['Close'].iloc[0])
        
        if is_long:
            if current_price >= target_price:
                exit_price = target_price
                exit_time = idx
                hit_target = True
                break
            elif current_price <= stop_price:
                exit_price = stop_price
                exit_time = idx
                hit_stop = True
                break
        else:  # short trade
            if current_price <= target_price:
                exit_price = target_price
                exit_time = idx
                hit_target = True
                break
            elif current_price >= stop_price:
                exit_price = stop_price
                exit_time = idx
                hit_stop = True
                break
                
    return exit_price, exit_time, hit_target, hit_stop

def run_backtest(news_df, initial_capital, position_size, take_profit, stop_loss, enable_advanced=False):
    logger.info("Starting backtest")
    
    if news_df.empty:
        logger.warning("No news data provided for backtest")
        return None
    
    # Ensure timezone column exists
    if 'timezone' not in news_df.columns:
        logger.warning("No timezone information in news data, defaulting to UTC")
        news_df['timezone'] = 'UTC'
    
    try:
        # Create price moves for the news events
        price_moves_df = create_price_moves(news_df)
        
        if price_moves_df.empty:
            logger.warning("No price moves generated")
            return None
            
        # Check if required columns exist
        required_columns = ['begin_price', 'end_price', 'entry_time', 'exit_time']
        missing_columns = [col for col in required_columns if col not in price_moves_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Initialize trade tracking
        trades = []
        current_capital = initial_capital
        
        # Sort by published date
        price_moves_df = price_moves_df.sort_values('published_date')
        
        for _, row in price_moves_df.iterrows():
            try:
                # Calculate position size and entry details
                trade_position_size = calculate_position_size(current_capital, position_size)
                entry_price = row['begin_price']
                shares = calculate_shares(trade_position_size, entry_price)
                
                if shares == 0:
                    continue
                    
                # Determine trade direction and prices
                is_long = row['predicted_side'] == 'UP'
                target_price = entry_price * (1 + take_profit) if is_long else entry_price * (1 - take_profit)
                stop_price = entry_price * (1 - stop_loss) if is_long else entry_price * (1 + stop_loss)
                
                # Check for exit conditions
                if 'intraday_prices' in row:
                    exit_price, exit_time, hit_target, hit_stop = check_exit(
                        intraday_data=row['intraday_prices'],
                        entry_price=entry_price,
                        target_price=target_price,
                        stop_price=stop_price,
                        is_long=is_long,
                        default_exit_time=row['exit_time'],
                        default_exit_price=row['end_price']
                    )
                else:
                    exit_price = row['end_price']
                    exit_time = row['exit_time']
                    hit_target = False
                    hit_stop = False
                
                # Calculate P&L
                pnl = shares * (exit_price - entry_price) if is_long else shares * (entry_price - exit_price)
                pnl_pct = (pnl / trade_position_size) * 100
                
                # Update capital
                current_capital += pnl
                
                # Record trade
                trade = {
                    'published_date': row['published_date'],
                    'market': row['market'],
                    'entry_time': row['entry_time'],
                    'exit_time': exit_time,
                    'ticker': row['ticker'],
                    'direction': 'LONG' if is_long else 'SHORT',
                    'shares': shares,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'target_price': target_price,
                    'stop_price': stop_price,
                    'hit_target': hit_target,
                    'hit_stop': hit_stop,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'capital_after': current_capital,
                    'news_event': row['event'],
                    'link': row['link']
                }
                trades.append(trade)
                
            except Exception as e:
                logger.error(f"Error processing trade for {row.get('ticker', 'unknown')}: {e}")
                continue
        
        if not trades:
            logger.warning("No trades generated during backtest")
            return None
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate metrics
        metrics = calculate_metrics(trades_df, initial_capital)
        
        return trades_df, metrics
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        return None

def calculate_metrics(trades_df, initial_capital):
    """Calculate backtest performance metrics"""
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    
    # Calculate total PnL in dollars
    total_pnl = trades_df['pnl'].sum()
    
    # Calculate total return (as percentage)
    total_return = (total_pnl / initial_capital) * 100
    
    # Calculate annualized return
    if not trades_df.empty:
        period_begin = pd.to_datetime(trades_df['entry_time'].min())
        period_end = pd.to_datetime(trades_df['exit_time'].max())
        days_diff = (period_end - period_begin).days
        
        if days_diff > 0:
            annualized_return = total_return * 365 / days_diff
        else:
            annualized_return = total_return  # If same day, use total return
    else:
        annualized_return = 0
    
    metrics = {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'annualized_return': annualized_return
    }
    
    return metrics 