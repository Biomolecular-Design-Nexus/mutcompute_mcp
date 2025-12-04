"""
MutCompute log-likelihood calculation tools for protein fitness prediction.

This MCP Server provides 1 tool:
1. mutcompute_calculate_llh: Calculate log-likelihood ratios for mutations using MutCompute probabilities

The tool reads a CSV file with variant sequences, compares to wild-type sequence,
and calculates log-likelihood values for mutations using MutCompute predictions.
"""

# Standard imports
from typing import Annotated
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
from fastmcp import FastMCP
from scipy.stats import spearmanr, pearsonr
from loguru import logger

# MCP server instance
mutcompute_llh_mcp = FastMCP(name="mutcompute_llh")

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
        logger.error(f"FASTA file not found: {fasta_path}")
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    logger.debug(f"Loading FASTA file: {fasta_path}")
    sequence = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('>'):
                continue
            sequence.append(line)

    result = ''.join(sequence)
    logger.debug(f"Loaded sequence of length {len(result)}")
    return result


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
        return None

    mutations = []
    for i, (wt_aa, var_aa) in enumerate(zip(wt_seq, variant_seq)):
        if wt_aa != var_aa:
            if wt_aa not in VALID_AA or var_aa not in VALID_AA:
                continue
            pos = i + start_pos
            mutations.append(f"{wt_aa}{pos}{var_aa}")

    return '/'.join(mutations) if mutations else ''


def calculate_loglikelihood(mutant, df_mut_probs):
    """
    Calculate the log-likelihood ratio for mutations w.r.t. the wild-type.

    Args:
        mutant (str): Mutation string, e.g. "V39Y" or "V39Y/V54S" for multiple mutations
        df_mut_probs (pd.DataFrame): DataFrame containing MutCompute mutation probabilities

    Returns:
        float: Log-likelihood ratio of the mutation(s), or None if invalid
    """
    if not isinstance(mutant, str) or mutant is None or mutant == '':
        return 0.0

    muts = mutant.split('/')
    likelihood = 0.0

    for mut in muts:
        if len(mut) < 3:
            return None

        ref_AA, pos, mut_AA = mut[0], int(mut[1:-1]), mut[-1]

        if ref_AA not in df_mut_probs.columns or mut_AA not in df_mut_probs.columns:
            return None

        if pos not in df_mut_probs['pos'].values:
            return None

        prob_ref = df_mut_probs.loc[df_mut_probs['pos'] == pos, ref_AA].values[0]
        prob_mut = df_mut_probs.loc[df_mut_probs['pos'] == pos, mut_AA].values[0]

        if prob_ref > 0:
            likelihood += np.log(prob_mut / prob_ref)
        else:
            return None

    return likelihood


def calculate_llh_wrapper(args):
    """Wrapper function for multiprocessing."""
    mutation, df_mut_probs = args
    return calculate_loglikelihood(mutation, df_mut_probs)


def calculate_llh_batch(mutations, df_mut_probs, n_proc=4):
    """
    Calculate log-likelihoods for a batch of mutations using multiprocessing.

    Args:
        mutations (list): List of mutation strings
        df_mut_probs (pd.DataFrame): DataFrame containing mutation probabilities
        n_proc (int): Number of processes to use

    Returns:
        np.array: Array of log-likelihood values
    """
    if n_proc == 1:
        llhs = [calculate_loglikelihood(mut, df_mut_probs) for mut in tqdm(mutations, ncols=80)]
    else:
        with mp.Pool(n_proc) as pool:
            args_for_llh = zip(mutations, [df_mut_probs] * len(mutations))
            llhs = list(tqdm(pool.imap(calculate_llh_wrapper, args_for_llh),
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
    logger.debug(f"Loading MutCompute probabilities from: {mutcompute_path}")
    df_mut_probs = pd.read_csv(mutcompute_path, index_col=0)

    # Rename columns from three-letter to one-letter codes
    for three_letter, one_letter in AA_THREE2ONE.items():
        df_mut_probs = df_mut_probs.rename(columns={'pr' + three_letter: one_letter})

    logger.debug(f"Loaded MutCompute probabilities with {len(df_mut_probs)} positions")
    return df_mut_probs


@mutcompute_llh_mcp.tool
def mutcompute_calculate_llh(
    data_csv: Annotated[str, "Path to CSV file containing variant sequences"],
    wt_fasta: Annotated[str | None, "Path to wild-type FASTA file. If None, uses wt.fasta in same directory as data_csv"] = None,
    mutcompute_csv: Annotated[str | None, "Path to MutCompute probability CSV file. If None, uses mutcompute.csv in same directory as data_csv"] = None,
    seq_col: Annotated[str, "Column name containing variant sequences"] = "sequence",
    start_pos: Annotated[int, "Starting position for mutation numbering (1-indexed)"] = 1,
    n_proc: Annotated[int, "Number of processes for parallel computation"] = 4,
    output_col: Annotated[str, "Name for output column"] = "mc_llh",
    output_csv: Annotated[str | None, "Output CSV file path. If None, uses <input_csv>_mc.csv"] = None,
    fitness_col: Annotated[str | None, "Column name containing fitness values for correlation evaluation"] = None,
) -> dict:
    """
    Calculate log-likelihood ratios for mutations using MutCompute probabilities.

    This tool:
    1. Reads CSV file with variant sequences and wild-type FASTA
    2. Derives mutations by comparing sequences to wild-type
    3. Loads MutCompute mutation probabilities from CSV
    4. Calculates log-likelihood ratio for each variant
    5. Returns results with optional correlation statistics

    The MutCompute probabilities should be pre-computed using the mutcompute_predict tool.

    Input: CSV with sequences, wild-type FASTA, MutCompute probabilities CSV
    Output: Dictionary with results path, statistics, and correlation metrics
    """
    logger.info(f"mutcompute_calculate_llh called with data_csv={data_csv}, seq_col={seq_col}, n_proc={n_proc}")

    try:
        data_csv_path = Path(data_csv).resolve()
        if not data_csv_path.exists():
            logger.error(f"Input file not found: {data_csv_path}")
            raise FileNotFoundError(f"Input file not found: {data_csv_path}")

        data_dir = data_csv_path.parent

        # Determine wt.fasta path
        if wt_fasta is None:
            wt_fasta_path = data_dir / "wt.fasta"
        else:
            wt_fasta_path = Path(wt_fasta).resolve()

        if not wt_fasta_path.exists():
            logger.error(f"Wild-type FASTA not found: {wt_fasta_path}")
            raise FileNotFoundError(f"Wild-type FASTA not found: {wt_fasta_path}")

        logger.info(f"Using wild-type FASTA: {wt_fasta_path}")

        # Determine mutcompute.csv path
        if mutcompute_csv is None:
            mutcompute_csv_path = data_dir / "mutcompute.csv"
        else:
            mutcompute_csv_path = Path(mutcompute_csv).resolve()

        if not mutcompute_csv_path.exists():
            logger.error(f"MutCompute data not found: {mutcompute_csv_path}")
            raise FileNotFoundError(f"MutCompute data not found: {mutcompute_csv_path}")

        logger.info(f"Using MutCompute probabilities: {mutcompute_csv_path}")

        # Load wild-type sequence
        wt_seq = load_fasta(wt_fasta_path)
        logger.info(f"Loaded wild-type sequence of length {len(wt_seq)}")

        # Load input data
        df = pd.read_csv(data_csv_path)
        logger.info(f"Loaded input data with {len(df)} rows")

        if seq_col not in df.columns:
            logger.error(f"Column '{seq_col}' not found in input file. Available columns: {df.columns.tolist()}")
            raise ValueError(f"Column '{seq_col}' not found in input file. Available columns: {df.columns.tolist()}")

        variant_seqs = df[seq_col].tolist()

        # Derive mutations from sequences
        logger.info("Deriving mutations from sequences")
        mutations = []
        for variant_seq in tqdm(variant_seqs, desc="Deriving mutations", ncols=80):
            mut = get_mutations_from_sequence(wt_seq, variant_seq, start_pos=start_pos)
            mutations.append(mut)

        df['_derived_mutations'] = mutations
        n_with_mutations = sum(1 for m in mutations if m and m != '')
        logger.info(f"Found {n_with_mutations} variants with mutations out of {len(mutations)} total")

        # Load MutCompute probabilities
        df_mut_probs = load_mutcompute_probs(mutcompute_csv_path)

        # Calculate log-likelihoods
        logger.info(f"Calculating log-likelihoods using {n_proc} processes")
        llhs = calculate_llh_batch(mutations, df_mut_probs, n_proc=n_proc)

        # Add to dataframe
        df_results = df.copy()
        df_results[output_col] = llhs

        # Report statistics
        valid_llhs = llhs[~pd.isna(llhs)]
        stats = {
            "valid_llh_count": int(len(valid_llhs)),
            "total_count": int(len(llhs)),
            "variants_with_mutations": n_with_mutations,
        }

        logger.info(f"Computed {len(valid_llhs)} valid log-likelihoods out of {len(llhs)} total")

        if len(valid_llhs) > 0:
            stats.update({
                "mean": float(np.mean(valid_llhs)),
                "std": float(np.std(valid_llhs)),
                "min": float(np.min(valid_llhs)),
                "max": float(np.max(valid_llhs)),
            })
            logger.info(f"LLH statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

        # Evaluate correlation with fitness if available
        correlation_results = None
        fitness_col_found = None

        if fitness_col:
            if fitness_col in df_results.columns:
                fitness_col_found = fitness_col
        else:
            # Auto-detect fitness column
            for col in ['fitness', 'log_fitness']:
                if col in df_results.columns:
                    fitness_col_found = col
                    break

        if fitness_col_found:
            logger.info(f"Evaluating correlation with fitness column: {fitness_col_found}")
            llh_series = pd.to_numeric(df_results[output_col], errors='coerce')
            fitness_series = pd.to_numeric(df_results[fitness_col_found], errors='coerce')

            valid_mask = ~(pd.isna(llh_series) | pd.isna(fitness_series))
            llh_values = llh_series[valid_mask].values
            fitness_values = fitness_series[valid_mask].values

            if len(llh_values) > 2:
                spearman_r, spearman_p = spearmanr(llh_values, fitness_values)
                pearson_r, pearson_p = pearsonr(llh_values, fitness_values)

                correlation_results = {
                    "fitness_column": fitness_col_found,
                    "valid_pairs": int(len(llh_values)),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                }
                logger.info(f"Correlation results: Spearman r={spearman_r:.4f}, Pearson r={pearson_r:.4f}")

        # Determine output path
        if output_csv:
            output_path = Path(output_csv).resolve()
        else:
            output_path = data_dir / f"{data_csv_path.stem}_mc.csv"

        # Save results
        df_results.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

        logger.info(f"mutcompute_calculate_llh completed successfully")
        return {
            "status": "success",
            "output_csv": str(output_path),
            "wt_sequence_length": len(wt_seq),
            "total_variants": len(df_results),
            "statistics": stats,
            "correlation": correlation_results,
            "mutcompute_probs_file": str(mutcompute_csv_path),
        }

    except Exception as e:
        logger.exception(f"Unexpected error in mutcompute_calculate_llh: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "data_csv": str(data_csv),
        }
