#!/usr/bin/env python3
"""
MutCompute Use Case 2: Batch Mutation Analysis

This script processes multiple PDB files in batch mode and generates mutation
predictions for all of them. It's useful for analyzing entire protein datasets
or comparing mutation effects across different protein structures.

Usage:
    python use_case_2_batch_mutation_analysis.py --input_dir /path/to/pdb/files --output_dir results
    python use_case_2_batch_mutation_analysis.py --input_list protein_list.txt --output_dir results
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import glob

# Add the repo to Python path
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
    os.environ['PYTHONPATH'] = str(REPO_DIR) + ':' + os.environ.get('PYTHONPATH', '')
    original_cwd = os.getcwd()
    os.chdir(REPO_DIR)
    return original_cwd

def process_single_pdb(pdb_path, output_dir, use_cpu=False):
    """Process a single PDB file and return results summary."""
    try:
        from run import gen_ensemble_inference

        pdb_path = Path(pdb_path)
        output_path = Path(output_dir) / f"{pdb_path.stem}_mutcompute.csv"

        print(f"Processing: {pdb_path.name}")
        start_time = time.time()

        predictions_df = gen_ensemble_inference(str(pdb_path), str(output_path))
        elapsed_time = time.time() - start_time

        # Calculate summary statistics
        summary = {
            'pdb_file': pdb_path.name,
            'num_residues': len(predictions_df),
            'num_chains': predictions_df['chain_id'].nunique(),
            'avg_wt_prob': predictions_df['wt_prob'].mean(),
            'avg_pred_prob': predictions_df['pred_prob'].mean(),
            'avg_log_ratio': predictions_df['avg_log_ratio'].mean(),
            'processing_time': elapsed_time,
            'output_file': output_path.name,
            'status': 'Success'
        }

        print(f"  ✓ Completed in {elapsed_time:.2f}s - {len(predictions_df)} residues")
        return summary

    except Exception as e:
        print(f"  ✗ Failed: {str(e)}")
        return {
            'pdb_file': pdb_path.name,
            'num_residues': 0,
            'num_chains': 0,
            'avg_wt_prob': None,
            'avg_pred_prob': None,
            'avg_log_ratio': None,
            'processing_time': 0,
            'output_file': None,
            'status': f'Failed: {str(e)}'
        }

def batch_process(pdb_files, output_dir, use_cpu=False):
    """Process multiple PDB files in batch mode."""
    # Configure Theano before processing
    configure_theano(use_cpu=use_cpu)

    # Setup environment
    original_cwd = setup_environment()

    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting batch processing of {len(pdb_files)} PDB files")
        print(f"Output directory: {output_dir.resolve()}")
        print(f"Using {'CPU' if use_cpu else 'GPU'} for inference")
        print("-" * 60)

        summaries = []
        total_start_time = time.time()

        for i, pdb_file in enumerate(pdb_files, 1):
            print(f"[{i}/{len(pdb_files)}]", end=" ")
            summary = process_single_pdb(pdb_file, output_dir, use_cpu)
            summaries.append(summary)

        total_time = time.time() - total_start_time

        # Create summary report
        summary_df = pd.DataFrame(summaries)
        summary_file = output_dir / "batch_summary.csv"
        summary_df.to_csv(summary_file, index=False)

        print("-" * 60)
        print(f"Batch processing completed in {total_time:.2f} seconds")
        print(f"Summary report saved to: {summary_file}")

        # Print summary statistics
        successful = summary_df[summary_df['status'] == 'Success']
        failed = summary_df[summary_df['status'] != 'Success']

        print(f"\nResults Summary:")
        print(f"  Successful: {len(successful)}/{len(summary_df)}")
        print(f"  Failed: {len(failed)}/{len(summary_df)}")

        if len(successful) > 0:
            print(f"  Total residues processed: {successful['num_residues'].sum()}")
            print(f"  Average processing time: {successful['processing_time'].mean():.2f}s per file")
            print(f"  Average wild-type probability: {successful['avg_wt_prob'].mean():.3f}")

        if len(failed) > 0:
            print(f"\nFailed files:")
            for _, row in failed.iterrows():
                print(f"  - {row['pdb_file']}: {row['status']}")

        return summary_df

    finally:
        os.chdir(original_cwd)

def get_pdb_files_from_directory(directory):
    """Get all PDB files from a directory."""
    directory = Path(directory)
    pdb_files = list(directory.glob("*.pdb")) + list(directory.glob("*.PDB"))
    return sorted([str(f) for f in pdb_files])

def get_pdb_files_from_list(file_list):
    """Get PDB files from a text file list."""
    pdb_files = []
    with open(file_list, 'r') as f:
        for line in f:
            pdb_path = line.strip()
            if pdb_path and Path(pdb_path).exists():
                pdb_files.append(pdb_path)
            elif pdb_path:
                print(f"Warning: File not found: {pdb_path}")
    return pdb_files

def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple PDB files for mutation effect prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all PDB files in a directory
    python use_case_2_batch_mutation_analysis.py --input_dir examples/data --output_dir batch_results

    # Process files listed in a text file
    python use_case_2_batch_mutation_analysis.py --input_list my_proteins.txt --output_dir results

    # Use CPU for processing (slower but doesn't require GPU)
    python use_case_2_batch_mutation_analysis.py --input_dir examples/data --output_dir results --cpu

Note: Input list should be a text file with one PDB file path per line.
        """
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input_dir', '-d',
                            help='Directory containing PDB files to process')
    input_group.add_argument('--input_list', '-l',
                            help='Text file containing list of PDB file paths (one per line)')

    parser.add_argument('--output_dir', '-o', required=True,
                        help='Output directory for results and summary')

    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU for inference')

    args = parser.parse_args()

    try:
        # Get list of PDB files to process
        if args.input_dir:
            pdb_files = get_pdb_files_from_directory(args.input_dir)
            if not pdb_files:
                print(f"No PDB files found in directory: {args.input_dir}")
                sys.exit(1)
        else:
            pdb_files = get_pdb_files_from_list(args.input_list)
            if not pdb_files:
                print(f"No valid PDB files found in list: {args.input_list}")
                sys.exit(1)

        print(f"Found {len(pdb_files)} PDB files to process")

        # Process files in batch
        summary_df = batch_process(pdb_files, args.output_dir, args.cpu)

    except KeyboardInterrupt:
        print("\nBatch processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()