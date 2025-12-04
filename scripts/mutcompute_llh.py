#!/usr/bin/env python3
"""
Script to calculate log-likelihood changes for mutations using MutCompute probabilities.
This script calculates how likely mutations are compared to wild-type based on MutCompute predictions.
Mutations are derived by comparing variant sequences to the wild-type sequence from wt.fasta.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from loguru import logger
from scipy.stats import spearmanr, pearsonr

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# Amino acid three-letter to one-letter code mapping
AA_THREE2ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# Valid amino acid one-letter codes
VALID_AA = set(AA_THREE2ONE.values())


def load_fasta(fasta_path):
    """
    Load a FASTA file and return the sequence.

    Args:
        fasta_path (str or Path): Path to FASTA file

    Returns:
        str: The protein sequence
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    sequence = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            sequence.append(line)

    return ''.join(sequence)


def get_mutations_from_sequence(wt_seq, variant_seq, start_pos=1):
    """
    Derive mutations by comparing a variant sequence to the wild-type.

    Args:
        wt_seq (str): Wild-type sequence
        variant_seq (str): Variant sequence
        start_pos (int): Starting position for numbering (1-indexed)

    Returns:
        str: Mutation string (e.g., "V39Y/A52G") or empty string if identical
    """
    if not isinstance(variant_seq, str) or pd.isna(variant_seq) or variant_seq == '':
        return ''

    # Clean the sequences
    wt_seq = wt_seq.strip().upper()
    variant_seq = variant_seq.strip().upper()

    if len(wt_seq) != len(variant_seq):
        logger.warning(f"Sequence length mismatch: WT={len(wt_seq)}, variant={len(variant_seq)}")
        return None

    mutations = []
    for i, (wt_aa, var_aa) in enumerate(zip(wt_seq, variant_seq)):
        if wt_aa != var_aa:
            if wt_aa not in VALID_AA or var_aa not in VALID_AA:
                logger.warning(f"Invalid amino acid at position {i + start_pos}: {wt_aa} -> {var_aa}")
                continue
            pos = i + start_pos
            mutations.append(f"{wt_aa}{pos}{var_aa}")

    return '/'.join(mutations) if mutations else ''


def cal_loglikelihood(mutant, df_mut_probs):
    """
    Calculate the log-likelihood ratio for mutations w.r.t. the wild-type.

    Args:
        mutant (str): Mutation string, e.g. "V39Y" or "V39Y/V54S" for multiple mutations
        df_mut_probs (pd.DataFrame): DataFrame containing MutCompute mutation probabilities

    Returns:
        float: Log-likelihood ratio of the mutation(s), or None if invalid
    """
    if not isinstance(mutant, str) or mutant is None or mutant == '':
        return 0

    muts = mutant.split('/')
    likelihood = 0

    for mut in muts:
        ref_AA, pos, mut_AA = mut[0], int(mut[1:-1]), mut[-1]

        if ref_AA not in df_mut_probs.columns or mut_AA not in df_mut_probs.columns:
            logger.warning(f"AA {ref_AA} or {mut_AA} not in df_mut_probs columns")
            return None

        if pos not in df_mut_probs['pos'].values:
            logger.warning(f"Position {pos} not in df_mut_probs")
            return None

        prob_ref = df_mut_probs.loc[df_mut_probs['pos'] == pos, ref_AA].values[0]
        prob_mut = df_mut_probs.loc[df_mut_probs['pos'] == pos, mut_AA].values[0]
        likelihood += np.log(prob_mut / prob_ref)

    return likelihood


def cal_llh_wrapper(args):
    """Wrapper function for multiprocessing."""
    mutation, df_mut_probs = args
    return cal_loglikelihood(mutation, df_mut_probs)


def cal_llh_batch(mutations, df_mut_probs, n_proc=4):
    """
    Calculate log-likelihoods for a batch of mutations using multiprocessing.

    Args:
        mutations (list): List of mutation strings
        df_mut_probs (pd.DataFrame): DataFrame containing mutation probabilities
        n_proc (int): Number of processes to use

    Returns:
        np.array: Array of log-likelihood values
    """
    with mp.Pool(n_proc) as pool:
        args_for_cal_llh = zip(mutations, [df_mut_probs] * len(mutations))
        llhs = list(tqdm(pool.imap(cal_llh_wrapper, args_for_cal_llh),
                        total=len(mutations), ncols=80, desc="Calculating LLH"))
    return np.array(llhs)


def load_mutcompute_probs(mutcompute_path):
    """
    Load and process MutCompute probability file.

    Args:
        mutcompute_path (str or Path): Path to mutcompute.csv file

    Returns:
        pd.DataFrame: Processed DataFrame with one-letter AA codes as columns
    """
    df_mut_probs = pd.read_csv(mutcompute_path, index_col=0)

    # Rename columns from three-letter to one-letter codes
    for three_letter, one_letter in AA_THREE2ONE.items():
        df_mut_probs = df_mut_probs.rename(columns={'pr' + three_letter: one_letter})

    return df_mut_probs


def run_mutcompute_llh(seq_path, wt_fasta_path=None, mutcompute_path=None,
                       seq_col='sequence', start_pos=1, n_proc=4, output_path=None,
                       fitness_col=None):
    """
    Calculate log-likelihood ratios for variant sequences in a CSV file.

    Args:
        seq_path: Path to input CSV file containing variant sequences
        wt_fasta_path: Path to wild-type FASTA file
        mutcompute_path: Path to MutCompute probability CSV file
        seq_col: Column name containing variant sequences
        start_pos: Starting position for mutation numbering (1-indexed)
        n_proc: Number of processes for parallel computation
        output_path: Optional output file path
        fitness_col: Column name containing fitness values for correlation evaluation

    Returns:
        pd.DataFrame: DataFrame with added 'mutations' and 'mc' columns
    """
    seq_path = Path(seq_path).resolve()
    if not seq_path.exists():
        logger.error(f"Input file not found: {seq_path}")
        raise FileNotFoundError(f"Input file not found: {seq_path}")

    # Determine wt.fasta path
    if wt_fasta_path is None:
        wt_fasta_path = seq_path.parent / "wt.fasta"
    else:
        wt_fasta_path = Path(wt_fasta_path).resolve()

    if not wt_fasta_path.exists():
        logger.error(f"Wild-type FASTA not found: {wt_fasta_path}")
        raise FileNotFoundError(f"Wild-type FASTA not found: {wt_fasta_path}")

    # Determine mutcompute.csv path
    if mutcompute_path is None:
        mutcompute_path = seq_path.parent / "mutcompute.csv"
    else:
        mutcompute_path = Path(mutcompute_path).resolve()

    if not mutcompute_path.exists():
        logger.error(f"MutCompute data not found: {mutcompute_path}")
        raise FileNotFoundError(f"MutCompute data not found: {mutcompute_path}")

    # Load wild-type sequence
    logger.info(f"Loading wild-type sequence from: {wt_fasta_path}")
    wt_seq = load_fasta(wt_fasta_path)
    logger.info(f"Wild-type sequence length: {len(wt_seq)}")

    # Load input data
    logger.info(f"Loading input data from: {seq_path}")
    df = pd.read_csv(seq_path)

    if seq_col not in df.columns:
        logger.error(f"Column '{seq_col}' not found in input file")
        raise ValueError(f"Column '{seq_col}' not found in input file")

    variant_seqs = df[seq_col].tolist()
    logger.info(f"Found {len(variant_seqs)} variant sequences to process")

    # Derive mutations from sequences
    logger.info("Deriving mutations from variant sequences...")
    mutations = []
    for variant_seq in tqdm(variant_seqs, desc="Deriving mutations", ncols=80):
        mut = get_mutations_from_sequence(wt_seq, variant_seq, start_pos=start_pos)
        mutations.append(mut)

    df['mutations'] = mutations
    n_with_mutations = sum(1 for m in mutations if m and m != '')
    logger.info(f"Found {n_with_mutations} variants with mutations")

    # Load MutCompute probabilities
    logger.info(f"Loading MutCompute probabilities from: {mutcompute_path}")
    df_mut_probs = load_mutcompute_probs(mutcompute_path)
    logger.info(f"Loaded probabilities for {len(df_mut_probs)} positions")

    # Calculate log-likelihoods
    logger.info(f"Calculating log-likelihoods using {n_proc} processes")
    df['mc'] = cal_llh_batch(mutations, df_mut_probs, n_proc=n_proc)

    # Evaluate correlation with fitness if available
    fitness_col_found = None
    if fitness_col:
        # User specified fitness column
        if fitness_col in df.columns:
            fitness_col_found = fitness_col
        else:
            logger.warning(f"Specified fitness column '{fitness_col}' not found in CSV")
    else:
        # Auto-detect fitness column
        for col in ['fitness', 'log_fitness']:
            if col in df.columns:
                fitness_col_found = col
                break

    if fitness_col_found:
        logger.info(f"Evaluating with fitness column: '{fitness_col_found}'")

        # Convert columns to numeric, coercing errors to NaN
        llh_series = pd.to_numeric(df['mc'], errors='coerce')
        fitness_series = pd.to_numeric(df[fitness_col_found], errors='coerce')

        # Get valid pairs (both LLH and fitness are not NaN and not None)
        valid_mask = ~(pd.isna(llh_series) | pd.isna(fitness_series))
        llh_values = llh_series[valid_mask].values
        fitness_values = fitness_series[valid_mask].values

        if len(llh_values) > 2:
            spearman_r, spearman_p = spearmanr(llh_values, fitness_values)
            pearson_r, pearson_p = pearsonr(llh_values, fitness_values)

            logger.info("=" * 60)
            logger.info("Correlation with Fitness")
            logger.info("=" * 60)
            logger.info(f"  Valid pairs: {len(llh_values)} / {len(df)}")
            logger.info(f"  Spearman correlation: {spearman_r:.4f} (p={spearman_p:.2e})")
            logger.info(f"  Pearson correlation:  {pearson_r:.4f} (p={pearson_p:.2e})")
            logger.info("=" * 60)
        else:
            logger.warning(f"Not enough valid pairs for correlation: {len(llh_values)}")
    else:
        logger.info("No fitness column found for correlation evaluation. Skipping.")

    # Determine output path
    if output_path is None:
        output_path = seq_path.parent / f"{seq_path.stem}_mc.csv"
    else:
        output_path = Path(output_path).resolve()

    # Save results
    df.to_csv(output_path, index=False)
    logger.success(f"Results saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate log-likelihood ratios for mutations using MutCompute probabilities.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_mutcompute_llh.py -i data.csv
  python scripts/run_mutcompute_llh.py -i data.csv -w wt.fasta -m mutcompute.csv
  python scripts/run_mutcompute_llh.py -i variants.csv --seq_col aa_seq --start_pos 2 -o results.csv

Note:
  - The wt.fasta file should contain the wild-type protein sequence
  - The mutcompute.csv file should be generated by running MutCompute on the wild-type
    PDB structure (either via the MutCompute server or using run_mutcompute.py)
  - Mutations are derived by comparing each variant sequence to the wild-type
"""
    )

    parser.add_argument('-i', '--input', required=True,
                        help='REQUIRED. Path to input CSV file containing variant sequences.')

    parser.add_argument('-w', '--wt_fasta', required=False, default=None,
                        help='OPTIONAL. Path to wild-type FASTA file. '
                             'Default: wt.fasta in the same directory as input.')

    parser.add_argument('-m', '--mutcompute', required=False, default=None,
                        help='OPTIONAL. Path to MutCompute probability CSV file. '
                             'Default: mutcompute.csv in the same directory as input.')

    parser.add_argument('-o', '--output', required=False, default=None,
                        help='OPTIONAL. Output CSV file path. '
                             'Default: {input_name}_mc.csv in the same directory.')

    parser.add_argument('--seq_col', type=str, default='sequence',
                        help='OPTIONAL. Column name containing variant sequences. Default: sequence')

    parser.add_argument('--start_pos', type=int, default=1,
                        help='OPTIONAL. Starting position for mutation numbering (1-indexed). Default: 1')

    parser.add_argument('-n', '--n_proc', type=int, default=4,
                        help='OPTIONAL. Number of processes for parallel computation. Default: 4')

    parser.add_argument('--fitness_col', type=str, default=None,
                        help='OPTIONAL. Column name containing fitness values for correlation evaluation. '
                             'If not specified, will auto-detect "fitness" or "log_fitness" columns.')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Starting MutCompute Log-Likelihood Calculation")
    logger.info("=" * 80)

    logger.info(f"Fitness column: {args.fitness_col if args.fitness_col else 'auto-detect'}")

    try:
        run_mutcompute_llh(
            seq_path=args.input,
            wt_fasta_path=args.wt_fasta,
            mutcompute_path=args.mutcompute,
            seq_col=args.seq_col,
            start_pos=args.start_pos,
            n_proc=args.n_proc,
            output_path=args.output,
            fitness_col=args.fitness_col
        )
        logger.info("=" * 80)
        logger.info("Log-likelihood calculation completed")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"Log-likelihood calculation failed: {e}")
        logger.error("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
