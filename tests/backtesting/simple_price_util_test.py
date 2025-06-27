from utils.backtesting.backtest_price_util import load_price_moves_csv, load_price_moves_db

print("Testing CSV price moves loading...")
price_moves_csv = load_price_moves_csv()
if price_moves_csv is not None:
    print(f"Loaded {len(price_moves_csv)} price moves from CSV.")
    print(price_moves_csv.head())
else:
    print("Failed to load price moves from CSV.")

print("\nTesting DB price moves loading...")
price_moves_db = load_price_moves_db()
if price_moves_db is not None:
    print(f"Loaded {len(price_moves_db)} price moves from DB.")
    print(price_moves_db.head())
else:
    print("Failed to load price moves from DB.") 