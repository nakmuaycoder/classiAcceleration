"""
Log Parsing Utilities for LightBlue BLE Sniffer Data.

This module provides the logic to convert raw hexadecimal logs captured 
via the LightBlue mobile app (acting as a BLE sniffer) into structured 
Pandas DataFrames.

Modern TinyML projects usually stream directly to a PC, but this parser 
maintains compatibility with sniffer-based data collection workflows.

Author: nakmuaycoder
Date: 2026/04
"""

import struct
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

class AccelLogParser:
    """
    Parser for accelerometer data samples captured via LightBlue BLE Sniffer.
    
    Attributes:
        header (Dict[str, str]): Mapping from BLE Characteristic UUIDs (hex) to column names.
        label_uuid (Optional[str]): UUID used for activity labels (walk/run/etc), usually 2105.
    """

    def __init__(self, header: Dict[str, str]):
        """
        Initializes the parser with a UUID mapping.
        
        Example header: {"2102": "x", "2103": "y", "2105": "label"}
        """
        self.label_uuid = header.get("label")
        # Filter out special 'label' key from column mapping
        self.column_mapping = {k: v for k, v in header.items() if k != "label"}

    def parse(self, path_log: str) -> pd.DataFrame:
        """
        Parses a single log file into a DataFrame.
        
        Args:
            path_log: Path to the .txt log file.
            
        Returns:
            pd.DataFrame: Structured sensor data (x, y, z, label, date).
        """
        if not os.path.exists(path_log):
            raise FileNotFoundError(f"Log file not found: {path_log}")

        # Extract date from filename (Legacy format LBX_LOGS_YYYY-MM-DD_...)
        try:
            date_str = os.path.basename(path_log).split("_")[2].replace('-', '')
        except IndexError:
            date_str = datetime.now().strftime("%Y%m%d")

        with open(path_log, 'r') as f:
            lines = f.readlines()

        data_dict = {col: [] for col in self.column_mapping.values()}
        data_dict["date"] = []
        if self.label_uuid:
            data_dict["label"] = []

        current_label = 0

        for line in lines:
            line = line.strip()
            
            # 1. Update activity label if present
            if self.label_uuid and f"0000{self.label_uuid}" in line:
                hex_val = line[-12:].replace(" ", "")
                try:
                    current_label = struct.unpack('i', bytes.fromhex(hex_val))[0]
                except (ValueError, struct.error):
                    pass

            # 2. Parse sensor data change
            if 'changed | value:' in line:
                # UUID is located between spaces after 'value:' usually at fixed offset
                # Original logic: UUID is line.split(" ")[7][4:8]
                parts = line.split(" ")
                if len(parts) < 8: continue
                uuid = parts[7][4:8]

                if uuid in self.column_mapping:
                    hex_val = line[-13:].replace(" ", "")
                    try:
                        value = struct.unpack('f', bytes.fromhex(hex_val))[0]
                        
                        col_name = self.column_mapping[uuid]
                        data_dict[col_name].append(value)

                        # Sync date and label with the primary axis (usually 'x' / uuid 2102)
                        if uuid == '2102':
                            hour_str = parts[3].replace(":", "")
                            try:
                                full_date = datetime.strptime(date_str + ' ' + hour_str, "%Y%m%d %H%M%S")
                                data_dict['date'].append(full_date)
                            except ValueError:
                                data_dict['date'].append(None)
                                
                            if self.label_uuid:
                                data_dict["label"].append(current_label)
                    except (ValueError, struct.error):
                        pass

        # Ensuring all columns have the same length before creating DataFrame
        # We fill missing values with the last valid one if sync is off
        df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        return df.dropna()
