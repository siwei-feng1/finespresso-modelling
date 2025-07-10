#!/usr/bin/env python3
"""
Script to remove the NOT NULL constraint from the volume column in price_moves table.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.db.price_move_db_util import remove_volume_not_null_constraint

if __name__ == "__main__":
    print("Removing NOT NULL constraint from volume column in price_moves table...")
    try:
        remove_volume_not_null_constraint()
        print("Successfully removed NOT NULL constraint from volume column!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 