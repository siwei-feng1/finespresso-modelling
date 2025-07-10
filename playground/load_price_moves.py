import pandas as pd
import pickle
import os
from pathlib import Path

def load_price_moves_pickle(file_path="data/1d_price_movement.pkl"):
    """
    Load price moves data from pickle file and display first 10 rows
    
    Args:
        file_path (str): Path to the pickle file
        
    Returns:
        pd.DataFrame: Loaded price moves data
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None
            
        print(f"📁 Loading price moves from: {file_path}")
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Convert to DataFrame if it's not already
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data)
            
        print(f"✅ Successfully loaded data with shape: {df.shape}")
        print(f"📊 Columns: {list(df.columns)}")
        
        # Display first 10 rows with better formatting
        print("\n" + "="*100)
        print("FIRST 10 ROWS (DETAILED VIEW):")
        print("="*100)
        
        # Set display options to show all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        # Show first 10 rows with key columns
        key_columns = ['news_id', 'yf_ticker', 'published_date', 'title', 'price_movement', 
                      'predicted_side', 'predicted_move', 'event', 'publisher']
        
        print("Key columns view:")
        print(df[key_columns].head(10).to_string())
        
        print("\n" + "="*100)
        print("SAMPLE OF ALL COLUMNS (first 3 rows):")
        print("="*100)
        print(df.head(3).to_string())
        
        # Show data types and basic info
        print("\n" + "="*100)
        print("DATA INFO:")
        print("="*100)
        print(df.info())
        
        # Show basic statistics for numeric columns
        print("\n" + "="*100)
        print("BASIC STATISTICS (NUMERIC COLUMNS):")
        print("="*100)
        numeric_cols = df.select_dtypes(include=['number']).columns
        print(df[numeric_cols].describe())
        
        # Show value counts for categorical columns
        print("\n" + "="*100)
        print("CATEGORICAL COLUMNS SUMMARY:")
        print("="*100)
        categorical_cols = ['publisher', 'event', 'predicted_side', 'asset_class', 'exchange']
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n{col.upper()} (top 10):")
                print(df[col].value_counts().head(10))
        
        # Save first 3 rows to data/yfinance/sample_first3.csv
        output_dir = Path("data/yfinance")
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_path = output_dir / "sample_first3.csv"
        df.head(3).to_csv(sample_path, index=False)
        print(f"\n💾 First 3 rows saved to: {sample_path}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading pickle file: {str(e)}")
        return None

def main():
    """Main function to load and display price moves data"""
    print("🚀 Loading Price Moves Data from Pickle File")
    print("="*100)
    
    # Load the data
    df = load_price_moves_pickle()
    
    if df is not None:
        print(f"\n✅ Successfully loaded {len(df)} rows of price moves data")
        print(f"📈 Ready for database insertion")
        
        # Show some key insights
        print(f"\n🔍 KEY INSIGHTS:")
        print(f"   • Price movements range: {df['price_movement'].min():.2f}% to {df['price_movement'].max():.2f}%")
        print(f"   • Average price movement: {df['price_movement'].mean():.2f}%")
        print(f"   • News items with predictions: {df['predicted_side'].notna().sum()}")
        print(f"   • Unique tickers: {df['yf_ticker'].nunique()}")
        print(f"   • Date range: {df['published_date'].min()} to {df['published_date'].max()}")
    else:
        print("❌ Failed to load price moves data")

if __name__ == "__main__":
    main()
