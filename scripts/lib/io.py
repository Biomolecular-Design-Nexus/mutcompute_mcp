"""
Shared I/O functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""
from pathlib import Path
from typing import Union, Any, List
import json
import csv

def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)

def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_csv(file_path: Union[str, Path]) -> List[dict]:
    """Load CSV file as list of dictionaries."""
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_csv(data: List[dict], file_path: Union[str, Path]) -> None:
    """Save list of dictionaries to CSV file."""
    if not data:
        return

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', newline='') as f:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def list_pdb_files(directory: Union[str, Path]) -> List[Path]:
    """Find all PDB files in a directory."""
    directory = Path(directory)
    pdb_files = list(directory.glob("*.pdb")) + list(directory.glob("*.PDB"))
    return sorted(pdb_files)

def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path