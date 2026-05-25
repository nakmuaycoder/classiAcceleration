"""
Unit Tests for the AccelLogParser.

Verifies:
1. Handling of complete rows.
2. Discarding rows missing x.
3. Discarding rows missing y or z.
"""

from ml.src.parser import AccelLogParser


def test_parser_complete_and_unmixed(tmp_path):
    # Create a mock BLE log file
    # Float hex representations in little endian:
    # 1.0 -> 00 00 80 3F
    # 2.0 -> 00 00 00 40
    # 3.0 -> 00 00 40 40
    # 4.0 -> 00 00 80 40
    # 6.0 -> 00 00 C0 40
    # 8.0 -> 00 00 00 41
    # 9.0 -> 00 00 10 41
    # 10.0 -> 00 00 20 41
    # 11.0 -> 00 00 30 41
    # 12.0 -> 00 00 40 41
    uuid_x = "00002102-0000-1000-8000-00805f9b34fb"
    uuid_y = "00002103-0000-1000-8000-00805f9b34fb"
    uuid_z = "00002104-0000-1000-8000-00805f9b34fb"
    uuid_lbl = "00002105-0000-1000-8000-00805f9b34fb"

    log_content = [
        f"Mon Nov 16 17:25:11 GMT 2020: Characteristic {uuid_lbl} changed | value: 01 00 00 00\n",
        # Sample 1: Perfect row (x=1.0, y=2.0, z=3.0)
        f"Mon Nov 16 17:25:11 GMT 2020: Characteristic {uuid_x} changed | value: 00 00 80 3F\n",
        f"Mon Nov 16 17:25:11 GMT 2020: Characteristic {uuid_y} changed | value: 00 00 00 40\n",
        f"Mon Nov 16 17:25:11 GMT 2020: Characteristic {uuid_z} changed | value: 00 00 40 40\n",
        # Sample 2: Missing 'y' (x=4.0, z=6.0) -> should be discarded entirely
        f"Mon Nov 16 17:25:12 GMT 2020: Characteristic {uuid_x} changed | value: 00 00 80 40\n",
        f"Mon Nov 16 17:25:12 GMT 2020: Characteristic {uuid_z} changed | value: 00 00 C0 40\n",
        # Sample 3: Missing 'x' (y=8.0, z=9.0) -> y and z should be ignored/discarded
        f"Mon Nov 16 17:25:13 GMT 2020: Characteristic {uuid_y} changed | value: 00 00 00 41\n",
        f"Mon Nov 16 17:25:13 GMT 2020: Characteristic {uuid_z} changed | value: 00 00 10 41\n",
        # Sample 4: Perfect row (x=10.0, y=11.0, z=12.0)
        f"Mon Nov 16 17:25:14 GMT 2020: Characteristic {uuid_x} changed | value: 00 00 20 41\n",
        f"Mon Nov 16 17:25:14 GMT 2020: Characteristic {uuid_y} changed | value: 00 00 30 41\n",
        f"Mon Nov 16 17:25:14 GMT 2020: Characteristic {uuid_z} changed | value: 00 00 40 41\n",
    ]

    log_path = tmp_path / "LBX_LOGS_2020-11-16_17-47_walk.txt"
    with open(log_path, "w") as f:
        f.writelines(log_content)

    parser = AccelLogParser()
    df = parser.parse(str(log_path))

    # We expect exactly 2 rows (Sample 1 and Sample 4)
    # Sample 2 is discarded (missing 'y')
    # Sample 3 is ignored (missing 'x')
    assert len(df) == 2

    # Row 1 values
    assert df.iloc[0]["x"] == 1.0
    assert df.iloc[0]["y"] == 2.0
    assert df.iloc[0]["z"] == 3.0

    # Row 2 values
    assert df.iloc[1]["x"] == 10.0
    assert df.iloc[1]["y"] == 11.0
    assert df.iloc[1]["z"] == 12.0
    print("✅ AccelLogParser unit test passed successfully.")
