"""
Data Preprocessing Pipeline for Accelerometer Logs with Typer CLI.
"""

from pathlib import Path
from typing import Annotated

import typer

from ml.src.parser import AccelLogParser

# Simple instantiation of Typer app
app = typer.Typer(help="Clean raw accelerometer log files into training CSVs.")


@app.command()
def main(
    raw_dir: Annotated[Path, typer.Option(help="Directory containing raw .txt logs")] = Path(
        "data/raw"
    ),
    clean_dir: Annotated[Path, typer.Option(help="Output directory for clean CSVs")] = Path(
        "data/clean"
    ),
    mapping_path: Annotated[Path, typer.Option(help="Path to JSON handle mapping file")] = Path(
        "data/log_mapping.json"
    ),
):
    """
    Cleans raw accelerometer log files into standardized CSVs for training.
    """
    # 1. Path validations
    if not raw_dir.exists():
        print(f"❌ Raw directory not found: {raw_dir}")
        raise typer.Exit(1)

    if not mapping_path.exists():
        print(f"ℹ️ Warning: Mapping file not found at {mapping_path}. Default parser used.")
        mapping = None
    else:
        mapping = str(mapping_path)

    clean_dir.mkdir(parents=True, exist_ok=True)

    # 2. Scanning and Processing
    raw_files = list(raw_dir.glob("*.txt"))
    if not raw_files:
        print(f"ℹ️ No logs found in {raw_dir}")
        return

    print(f"🧹 Preprocessing {len(raw_files)} files using Typer engine...")
    parser = AccelLogParser(mapping=mapping)

    for rf in raw_files:
        source_path = str(rf)
        target_path = str(clean_dir / rf.with_suffix(".csv").name)

        try:
            parser.parse_to_csv(source_path, target_path)
            print(f"  ✅ {rf.name} -> CSV")
        except Exception as e:
            print(f"  ❌ Failed {rf.name}: {e}")

    print(f"🚀 Complete. Data located in: {clean_dir}")


if __name__ == "__main__":
    app()
