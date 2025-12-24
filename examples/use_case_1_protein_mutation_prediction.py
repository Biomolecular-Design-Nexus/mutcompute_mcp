#!/usr/bin/env python3
"""
MutCompute Use Case 1: Protein Mutation Effect Prediction

This script predicts the effect of point mutations on protein stability using
the MutCompute ensemble model. It processes a PDB file and generates mutation
probability matrices for all residues in the protein structure.

Usage:
    python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --output predictions.csv
    python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --output predictions.csv --cpu
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# Add the repo to Python path to import MutCompute modules
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent / 'repo' / 'mutcompute'
sys.path.insert(0, str(REPO_DIR))

def configure_theano(use_cpu=False):
    """Configure Theano device settings."""
    if 'THEANO_FLAGS' not in os.environ:
        if use_cpu:
            os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'
        else:
            os.environ['THEANO_FLAGS'] = 'device=cuda,floatX=float32'

def setup_environment():
    """Setup the environment for MutCompute execution."""
    # Set PYTHONPATH to include the repo directory
    os.environ['PYTHONPATH'] = str(REPO_DIR) + ':' + os.environ.get('PYTHONPATH', '')

    # Change to the repo directory for relative imports to work
    original_cwd = os.getcwd()
    os.chdir(REPO_DIR)
    return original_cwd

def predict_mutations(pdb_path, output_path=None, use_cpu=False):
    """
    Predict mutation effects for a given PDB file.

    Args:
        pdb_path (str): Path to the input PDB file
        output_path (str, optional): Path for output CSV. If None, auto-generated
        use_cpu (bool): Whether to use CPU instead of GPU

    Returns:
        pandas.DataFrame: Predictions with mutation probabilities
    """
    # Configure Theano before importing
    configure_theano(use_cpu=use_cpu)

    # Setup environment and change directory
    original_cwd = setup_environment()

    try:
        # Import MutCompute modules after environment setup
        from run import gen_ensemble_inference

        # Convert to absolute path
        pdb_path = Path(pdb_path).resolve()

        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        print(f"Processing PDB file: {pdb_path}")
        print(f"Using {'CPU' if use_cpu else 'GPU'} for inference")

        # Generate predictions
        start_time = time.time()
        predictions_df = gen_ensemble_inference(pdb_path, output_path)
        elapsed_time = time.time() - start_time

        print(f"Prediction completed in {elapsed_time:.2f} seconds")
        print(f"Generated predictions for {len(predictions_df)} residues")

        return predictions_df

    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def main():
    parser = argparse.ArgumentParser(
        description='Predict the effect of point mutations on protein stability using MutCompute',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default output
    python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb

    # Specify custom output file
    python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --output my_predictions.csv

    # Use CPU instead of GPU
    python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --cpu

    # Process your own PDB file
    python use_case_1_protein_mutation_prediction.py --input /path/to/your/protein.pdb --output results.csv
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to the input PDB file')

    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file path. If not specified, saves as {input_name}_mutcompute.csv')

    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU for inference (default: use GPU)')

    args = parser.parse_args()

    try:
        predictions = predict_mutations(
            pdb_path=args.input,
            output_path=args.output,
            use_cpu=args.cpu
        )

        print(f"\nPrediction Summary:")
        print(f"- Total residues: {len(predictions)}")
        print(f"- Unique chains: {predictions['chain_id'].nunique()}")
        print(f"- Average wild-type probability: {predictions['wt_prob'].mean():.3f}")
        print(f"- Average predicted probability: {predictions['pred_prob'].mean():.3f}")

        # Show top 5 most confident predictions
        print(f"\nTop 5 most confident predictions:")
        top_predictions = predictions.nlargest(5, 'pred_prob')[['aa_id', 'pos', 'wtAA', 'prAA', 'pred_prob']]
        print(top_predictions.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()