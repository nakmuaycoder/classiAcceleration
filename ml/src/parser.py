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
    Ensures temporal synchronization across axes (x, y, z) and strictly
    tracked data lineage.
    """

    def __init__(self, mapping: dict[str, str] | str | None = None):
        """
        Initializes the parser with a mapping (dict or path to JSON).
        """
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
        self.column_mapping = {k: v for k, v in config.items() if k != "label"}
        self.axes = list(self.column_mapping.values())

    def parse(self, path_log: str, explicit_date: str | None = None) -> pd.DataFrame:
        """
        Parses a single log file.

        Args:
            path_log: Path to the .txt log.
            explicit_date: Optional date string (YYYYMMDD). If not provided,
                           the parser tries to extract it from the filename.
        """
        if not os.path.exists(path_log):
            raise FileNotFoundError(f"Log file not found: {path_log}")

        # Strictly trying to get the date: Filename > Argument > Error
        date_str = explicit_date
        if date_str is None:
            try:
                # Expecting legacy format LBX_LOGS_YYYY-MM-DD_...
                # On essaie d'extraire la date du nom de fichier LBX_LOGS_2020-11-20_...
                date_str = os.path.basename(path_log).split("_")[2].replace("-", "")
            except (IndexError, AttributeError):
                raise ValueError(
                    f"Could not extract date from filename '{os.path.basename(path_log)}'. "
                    "Please provide an 'explicit_date' (YYYYMMDD) to ensure data lineage."
                ) from None

        with open(path_log) as f:
            lines = f.readlines()

        # Fallback label extraction from filename
        filename_lower = os.path.basename(path_log).lower()
        if "rest" in filename_lower:
            fallback_label = 0
        elif "walk" in filename_lower:
            fallback_label = 1
        elif "run" in filename_lower:
            fallback_label = 2
        else:
            fallback_label = 0

        rows = []
        current_row = {}
        current_label = fallback_label

        for line in lines:
            line = line.strip()

            # 1. Update activity label
            if f"0000{self.label_handle}" in line:
                try:
                    hex_val = "".join(line.split("value: ")[1].split())
                    parsed_label = struct.unpack("<i", bytes.fromhex(hex_val))[0]
                    if fallback_label == 0:
                        current_label = parsed_label
                except (ValueError, struct.error, IndexError):
                    pass

            # 2. Parse sensor data change
            if "changed | value:" in line:
                parts = line.split(" ")
                if len(parts) < 8:
                    continue

                handle = parts[7][4:8]

                if handle in self.column_mapping:
                    try:
                        hex_val = "".join(line.split("value: ")[1].split())
                        # Valeur float (f) 32 bits little-endian
                        value = struct.unpack("<f", bytes.fromhex(hex_val))[0]
                        col_name = self.column_mapping[handle]

                        if col_name == "x":
                            if all(ax in current_row for ax in self.axes):
                                rows.append(current_row)
                            current_row = {
                                "x": value,
                                "label": current_label,
                            }
                            hour_str = parts[3].replace(":", "")
                            try:
                                current_row["date"] = datetime.strptime(
                                    date_str + " " + hour_str, "%Y%m%d %H%M%S"
                                )
                            except ValueError:
                                current_row["date"] = None
                        elif col_name == "y":
                            if (
                                "x" in current_row
                                and "y" not in current_row
                                and "z" not in current_row
                            ):
                                current_row["y"] = value
                            else:
                                current_row = {}
                        elif col_name == "z":
                            if "x" in current_row and "y" in current_row and "z" not in current_row:
                                current_row["z"] = value
                            else:
                                current_row = {}

                    except (ValueError, struct.error, IndexError):
                        pass

        # Flush du dernier enregistrement s'il est complet
        if current_row and all(ax in current_row for ax in self.axes):
            rows.append(current_row)

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=self.axes + ["label", "date"])

        # Ensure all axes columns exist before dropping NaNs to avoid KeyError
        for ax in self.axes:
            if ax not in df.columns:
                df[ax] = None

        return df.dropna(subset=self.axes)

    def parse_to_csv(self, source_path: str, target_path: str) -> None:
        """
        Parses and saves immediately to CSV.
        """
        df = self.parse(source_path)
        df.to_csv(target_path, index=False)
