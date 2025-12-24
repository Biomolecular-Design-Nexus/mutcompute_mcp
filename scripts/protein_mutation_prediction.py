#!/usr/bin/env python3
"""
Script: protein_mutation_prediction.py
Description: Predict mutation effects for a protein using MutCompute ensemble model

Original Use Case: examples/use_case_1_protein_mutation_prediction.py
Dependencies Removed: datetime (unused), simplified environment setup
Repo Dependencies: Still requires repo/mutcompute for model inference (lazy loaded)

Usage:
    python scripts/protein_mutation_prediction.py --input <pdb_file> --output <output_file>

Example:
    python scripts/protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --output results/predictions.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Union, Optional, Dict, Any
import json

# Essential scientific packages
import numpy as np
import pandas as pd

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "use_cpu": False,
    "theano_flags_cpu": "device=cpu,floatX=float32",
    "theano_flags_gpu": "device=cuda,floatX=float32",
    "output_suffix": "_mutcompute.csv"
}

# ==============================================================================
# Path Management
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MCP_ROOT = SCRIPT_DIR.parent
REPO_DIR = MCP_ROOT / 'repo' / 'mutcompute'

# ==============================================================================
# Utility Functions (simplified from repo)
# ==============================================================================
def configure_theano(use_cpu: bool = False) -> None:
    """Configure Theano device settings before importing."""
    if 'THEANO_FLAGS' not in os.environ:
        if use_cpu:
            os.environ['THEANO_FLAGS'] = DEFAULT_CONFIG["theano_flags_cpu"]
        else:
            os.environ['THEANO_FLAGS'] = DEFAULT_CONFIG["theano_flags_gpu"]

def setup_mutcompute_environment() -> str:
    """Setup environment for MutCompute execution. Returns original CWD."""
    # Set PYTHONPATH to include the repo directory
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = f"{REPO_DIR}:{current_pythonpath}"

    # Change to repo directory for relative imports
    original_cwd = os.getcwd()
    os.chdir(REPO_DIR)
    return original_cwd

def validate_pdb_file(pdb_path: Path) -> bool:
    """Validate that the PDB file exists and has basic structure."""
    if not pdb_path.exists():
        return False

    # Basic validation - check if file contains ATOM records
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                return True
    return False

def generate_output_path(input_file: Path, output_file: Optional[Path] = None) -> Path:
    """Generate output file path if not provided."""
    if output_file:
        return Path(output_file)

    # Auto-generate output path
    output_name = input_file.stem + DEFAULT_CONFIG["output_suffix"]
    return input_file.parent / output_name

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_protein_mutation_prediction(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict mutation effects for a protein using MutCompute.

    Args:
        input_file: Path to input PDB file
        output_file: Path to save output CSV (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with mutation predictions
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_protein_mutation_prediction("input.pdb", "output.csv")
        >>> print(f"Predicted {len(result['result'])} mutations")
    """
    # Setup
    input_file = Path(input_file).resolve()
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validation
    if not validate_pdb_file(input_file):
        raise FileNotFoundError(f"Invalid or missing PDB file: {input_file}")

    # Configure Theano before importing MutCompute
    configure_theano(use_cpu=config["use_cpu"])

    # Setup MutCompute environment
    original_cwd = setup_mutcompute_environment()

    try:
        # Lazy import of MutCompute (only when needed)
        sys.path.insert(0, str(REPO_DIR))
        from run import gen_ensemble_inference

        print(f"Processing PDB file: {input_file}")
        print(f"Using {'CPU' if config['use_cpu'] else 'GPU'} for inference")

        # Generate output path
        output_path = generate_output_path(input_file, output_file) if output_file else None

        # Run prediction
        start_time = time.time()
        predictions_df = gen_ensemble_inference(str(input_file), str(output_path) if output_path else None)
        elapsed_time = time.time() - start_time

        print(f"Prediction completed in {elapsed_time:.2f} seconds")
        print(f"Generated predictions for {len(predictions_df)} residues")

        # Calculate summary statistics
        summary_stats = {
            "total_residues": len(predictions_df),
            "unique_chains": predictions_df['chain_id'].nunique() if 'chain_id' in predictions_df.columns else 1,
            "avg_wt_prob": float(predictions_df['wt_prob'].mean()) if 'wt_prob' in predictions_df.columns else None,
            "avg_pred_prob": float(predictions_df['pred_prob'].mean()) if 'pred_prob' in predictions_df.columns else None,
            "processing_time": elapsed_time
        }

        return {
            "result": predictions_df,
            "output_file": str(output_path) if output_path else None,
            "metadata": {
                "input_file": str(input_file),
                "config": config,
                "summary_stats": summary_stats
            }
        }

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

# ==============================================================================
# Batch Processing Support
# ==============================================================================
def run_batch_protein_mutation_prediction(
    input_files: list,
    output_dir: Union[str, Path],
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process multiple PDB files in batch mode.

    Args:
        input_files: List of PDB file paths
        output_dir: Directory to save output files
        config: Configuration dict
        **kwargs: Override specific config parameters

    Returns:
        Dict with batch processing results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    results = []
    total_start_time = time.time()

    for i, input_file in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] Processing {Path(input_file).name}")

        try:
            # Generate output file path
            input_path = Path(input_file)
            output_file = output_dir / f"{input_path.stem}{config['output_suffix']}"

            # Process single file
            result = run_protein_mutation_prediction(
                input_file=input_file,
                output_file=output_file,
                config=config
            )

            result_summary = {
                "file": input_path.name,
                "status": "success",
                "output_file": result["output_file"],
                **result["metadata"]["summary_stats"]
            }

        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
            result_summary = {
                "file": Path(input_file).name,
                "status": f"failed: {str(e)}",
                "output_file": None,
                "total_residues": 0,
                "processing_time": 0
            }

        results.append(result_summary)

    total_time = time.time() - total_start_time

    # Save batch summary
    summary_df = pd.DataFrame(results)
    summary_file = output_dir / "batch_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    successful = summary_df[summary_df['status'] == 'success']

    batch_metadata = {
        "total_files": len(input_files),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_time": total_time,
        "summary_file": str(summary_file)
    }

    return {
        "results": results,
        "summary": summary_df,
        "metadata": batch_metadata
    }

# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input PDB file path')
    parser.add_argument('--output', '-o',
                       help='Output CSV file path (auto-generated if not specified)')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU for inference')
    parser.add_argument('--batch', action='store_true',
                       help='Treat input as directory for batch processing')
    parser.add_argument('--output_dir',
                       help='Output directory for batch mode')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI arguments
    cli_overrides = {}
    if args.cpu:
        cli_overrides["use_cpu"] = True

    try:
        if args.batch:
            # Batch mode
            if not args.output_dir:
                print("Error: --output_dir required for batch mode")
                sys.exit(1)

            input_dir = Path(args.input)
            if not input_dir.is_dir():
                print(f"Error: Input directory not found: {input_dir}")
                sys.exit(1)

            # Find PDB files
            pdb_files = list(input_dir.glob("*.pdb")) + list(input_dir.glob("*.PDB"))
            if not pdb_files:
                print(f"No PDB files found in: {input_dir}")
                sys.exit(1)

            print(f"Found {len(pdb_files)} PDB files for batch processing")

            result = run_batch_protein_mutation_prediction(
                input_files=[str(f) for f in pdb_files],
                output_dir=args.output_dir,
                config=config,
                **cli_overrides
            )

            print(f"\nBatch processing completed:")
            print(f"  Successful: {result['metadata']['successful']}")
            print(f"  Failed: {result['metadata']['failed']}")
            print(f"  Summary: {result['metadata']['summary_file']}")

        else:
            # Single file mode
            result = run_protein_mutation_prediction(
                input_file=args.input,
                output_file=args.output,
                config=config,
                **cli_overrides
            )

            stats = result["metadata"]["summary_stats"]
            print(f"\nPrediction Summary:")
            print(f"  Total residues: {stats['total_residues']}")
            print(f"  Unique chains: {stats['unique_chains']}")
            if stats['avg_wt_prob']:
                print(f"  Average wild-type probability: {stats['avg_wt_prob']:.3f}")
            if stats['avg_pred_prob']:
                print(f"  Average predicted probability: {stats['avg_pred_prob']:.3f}")
            print(f"  Processing time: {stats['processing_time']:.2f}s")

            if result["output_file"]:
                print(f"✅ Output saved: {result['output_file']}")

        return result

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()