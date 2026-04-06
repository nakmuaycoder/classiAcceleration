"""
Log Parsing Utilities for LightBlue BLE Sniffer Data.

This module provides the logic to convert raw hexadecimal logs captured
via the LightBlue mobile app into structured Pandas DataFrames.

Author: nakmuaycoder
Date: 2026/04
"""

import json
import os
import struct
from datetime import datetime

import pandas as pd


class AccelLogParser:
    """
    Parser for accelerometer data samples captured via LightBlue BLE Sniffer.
    """

    def __init__(self, mapping: dict[str, str] | str | None = None):
        """
        Initializes the parser with a mapping (dict or path to JSON).

        Default mapping: {"2102": "x", "2103": "y", "2104": "z", "label": "2105"}
        """
        # Default fallback mapping
        default_mapping = {"2102": "x", "2103": "y", "2104": "z", "label": "2105"}

        if mapping is None:
            config = default_mapping
        elif isinstance(mapping, str):
            if os.path.exists(mapping):
                with open(mapping) as f:
                    config = json.load(f)
            else:
                config = default_mapping
        else:
            config = mapping

        self.label_handle = config.get("label", "2105")
        # Column mapping: handle -> axis (e.g., "2102" -> "x")
        self.column_mapping = {k: v for k, v in config.items() if k != "label"}

    def parse(self, path_log: str) -> pd.DataFrame:
        """
        Parses a single log file into a DataFrame.
        """
        if not os.path.exists(path_log):
            raise FileNotFoundError(f"Log file not found: {path_log}")

        # Extract date from filename
        try:
            date_str = os.path.basename(path_log).split("_")[2].replace("-", "")
        except IndexError:
            date_str = datetime.now().strftime("%Y%m%d")

        with open(path_log) as f:
            lines = f.readlines()

        data_dict = {col: [] for col in self.column_mapping.values()}
        data_dict["date"] = []
        data_dict["label"] = []

        current_label = 0

        for line in lines:
            line = line.strip()

            # 1. Update activity label
            if f"0000{self.label_handle}" in line:
                parts = line.split(" ")
                hex_val = parts[-1]
                try:
                    current_label = struct.unpack("<i", bytes.fromhex(hex_val))[0]
                except (ValueError, struct.error):
                    pass

            # 2. Parse sensor data change
            if "changed | value:" in line:
                parts = line.split(" ")
                if len(parts) < 8:
                    continue

                # Check handle (format 0000XXXX)
                handle = parts[7][4:8]

                if handle in self.column_mapping:
                    hex_val = parts[-1]
                    try:
                        # Sensor values are usually little-endian floats (f)
                        value = struct.unpack("<f", bytes.fromhex(hex_val))[0]
                        col_name = self.column_mapping[handle]
                        data_dict[col_name].append(value)

                        # Sync date and label with the 'x' axis trigger
                        if col_name == "x":
                            hour_str = parts[3].replace(":", "")
                            try:
                                full_date = datetime.strptime(
                                    date_str + " " + hour_str, "%Y%m%d %H%M%S"
                                )
                                data_dict["date"].append(full_date)
                            except ValueError:
                                data_dict["date"].append(None)

                            data_dict["label"].append(current_label)
                    except (ValueError, struct.error):
                        pass

        # Clean output
        df = pd.DataFrame.from_dict(data_dict, orient="index").transpose()
        return df.dropna()

    def parse_to_csv(self, source_path: str, target_path: str) -> None:
        """
        Parses a log file and saves it immediately to CSV.
        """
        df = self.parse(source_path)
        df.to_csv(target_path, index=False)
