"""
Data Preprocessing Module for TinyML Fitness Tracker.

This module provides utilities to convert raw logs captured via 
the LightBlue BLE Sniffer app into structured CSV files. 

Author: nakmuaycoder
Date: 2026/04
"""

import os
import json
import glob
import pandas as pd
from ml.src.parser import AccelLogParser

def preprocess():
    """
    Scans the raw data directory for BLE logs and converts them to formatted CSVs.
    
    Workflow:
    1. Reads UUID mappings from header.json (mappings for x, y, z, and labels).
    2. Instantiates an AccelLogParser with the defined header.
    3. Iterates over all .txt log files in data/raw/.
    4. Parses hex-encoded sensor values into physical acceleration units.
    5. Saves the resulting DataFrames into data/clean/ for model ingestion.
    """
    raw_dir = "data/raw"
    clean_dir = "data/clean"
    header_path = "data_preprocessing/header.json"
    
    os.makedirs(clean_dir, exist_ok=True)
    
    if not os.path.exists(header_path):
        print(f"❌ Header mapping not found at {header_path}")
        return

    with open(header_path, "r") as f:
        header = json.load(f)
    
    parser = AccelLogParser(header)
    log_files = glob.glob(os.path.join(raw_dir, "*.txt"))
    
    if not log_files:
        print(f"❌ No raw log files found in {raw_dir}")
        return

    print(f"🧹 Found {len(log_files)} raw log files.")
    
    for log_path in log_files:
        filename = os.path.basename(log_path).replace(".txt", ".csv")
        output_path = os.path.join(clean_dir, filename)
        
        print(f"  -> Processing {os.path.basename(log_path)}...")
        try:
            # We use the new refactored parser
            df = parser.parse(log_path)
            df.to_csv(output_path, index=False)
        except Exception as e:
            print(f"  ❌ Error processing {log_path}: {e}")

    print(f"\n✨ Done! Processed files are in {clean_dir}")

if __name__ == "__main__":
    preprocess()
