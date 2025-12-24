#!/usr/bin/env python3
"""
Script: mutation_hotspot_analysis.py
Description: Analyze mutation hotspots from MutCompute prediction results

Original Use Case: examples/use_case_3_mutation_hotspot_analysis.py
Dependencies Removed: seaborn (using matplotlib only), simplified visualization code
Repo Dependencies: None - fully self-contained

Usage:
    python scripts/mutation_hotspot_analysis.py --input <predictions.csv> --output_dir <output_dir>

Example:
    python scripts/mutation_hotspot_analysis.py --input examples/data/1y4a_BPN_mutcompute.csv --output_dir results/hotspots
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple, List

# Essential scientific packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "percentile_threshold": 10,
    "amino_acids": ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'],
    "plot_dpi": 300,
    "plot_format": "png"
}

# ==============================================================================
# Core Analysis Functions (extracted from use case)
# ==============================================================================
def analyze_mutation_tolerance(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Analyze mutation tolerance for each residue.

    Args:
        df: DataFrame with MutCompute predictions
        config: Configuration parameters

    Returns:
        DataFrame with tolerance analysis results
    """
    amino_acids = config["amino_acids"]
    prob_columns = [f'pr{aa}' for aa in amino_acids]
    tolerance_data = []

    for _, row in df.iterrows():
        residue_probs = row[prob_columns].values

        # Get basic residue information
        wt_aa = row['wtAA']
        wt_prob = row['wt_prob']
        pred_prob = row['pred_prob']

        # Calculate Shannon entropy (mutation tolerance)
        residue_probs_array = np.array(residue_probs, dtype=np.float64)
        entropy = -np.sum(residue_probs_array * np.log2(np.maximum(residue_probs_array, 1e-10)))

        # Find best alternative amino acid
        non_wt_probs = residue_probs_array.copy()
        wt_index = amino_acids.index(wt_aa)
        non_wt_probs[wt_index] = 0
        max_alternative_prob = np.max(non_wt_probs)
        max_alternative_aa = amino_acids[np.argmax(non_wt_probs)]

        # Calculate top 3 alternative probabilities
        top3_alternatives = np.sort(non_wt_probs)[-3:].sum()

        # Conservation score (1 - normalized entropy)
        max_entropy = np.log2(20)  # Maximum possible entropy for 20 amino acids
        conservation_score = 1 - (entropy / max_entropy)

        tolerance_data.append({
            'aa_id': row['aa_id'],
            'pdb_id': row.get('pdb_id', 'unknown'),
            'chain_id': row.get('chain_id', 'unknown'),
            'pos': row['pos'],
            'wtAA': wt_aa,
            'wt_prob': wt_prob,
            'pred_prob': pred_prob,
            'avg_log_ratio': row.get('avg_log_ratio', 0),
            'entropy': entropy,
            'conservation_score': conservation_score,
            'mutation_sensitivity': wt_prob,
            'max_alternative_prob': max_alternative_prob,
            'max_alternative_aa': max_alternative_aa,
            'top3_alternatives_prob': top3_alternatives
        })

    return pd.DataFrame(tolerance_data)

def identify_hotspots(tolerance_df: pd.DataFrame, percentile_threshold: float = 10) -> pd.DataFrame:
    """
    Identify mutation hotspots based on multiple criteria.

    Args:
        tolerance_df: DataFrame from analyze_mutation_tolerance
        percentile_threshold: Percentile threshold for hotspot identification

    Returns:
        DataFrame with hotspot classifications added
    """
    # Calculate percentile thresholds
    low_entropy_threshold = np.percentile(tolerance_df['entropy'], percentile_threshold)
    high_conservation_threshold = np.percentile(tolerance_df['conservation_score'], 100 - percentile_threshold)
    high_sensitivity_threshold = np.percentile(tolerance_df['mutation_sensitivity'], 100 - percentile_threshold)

    # Classify positions
    tolerance_df['is_conserved'] = tolerance_df['entropy'] <= low_entropy_threshold
    tolerance_df['is_highly_conserved'] = tolerance_df['conservation_score'] >= high_conservation_threshold
    tolerance_df['is_mutation_sensitive'] = tolerance_df['mutation_sensitivity'] >= high_sensitivity_threshold

    # Compute hotspot score
    tolerance_df['hotspot_score'] = (
        tolerance_df['is_conserved'].astype(int) +
        tolerance_df['is_highly_conserved'].astype(int) +
        tolerance_df['is_mutation_sensitive'].astype(int)
    )

    tolerance_df['is_hotspot'] = tolerance_df['hotspot_score'] >= 2

    # Identify flexible positions
    high_entropy_threshold = np.percentile(tolerance_df['entropy'], 100 - percentile_threshold)
    tolerance_df['is_flexible'] = tolerance_df['entropy'] >= high_entropy_threshold

    return tolerance_df

def generate_summary_report(tolerance_df: pd.DataFrame) -> str:
    """Generate a comprehensive summary report text."""
    total_residues = len(tolerance_df)
    hotspots = tolerance_df[tolerance_df['is_hotspot']]
    flexible_positions = tolerance_df[tolerance_df['is_flexible']]

    report = []
    report.append("=== MUTATION HOTSPOT ANALYSIS REPORT ===\n")

    report.append(f"Total residues analyzed: {total_residues}")
    report.append(f"Mutation hotspots identified: {len(hotspots)} ({len(hotspots)/total_residues*100:.1f}%)")
    report.append(f"Flexible positions identified: {len(flexible_positions)} ({len(flexible_positions)/total_residues*100:.1f}%)\n")

    # Overall statistics
    report.append("=== OVERALL STATISTICS ===")
    report.append(f"Average entropy: {tolerance_df['entropy'].mean():.3f} ± {tolerance_df['entropy'].std():.3f}")
    report.append(f"Average conservation score: {tolerance_df['conservation_score'].mean():.3f} ± {tolerance_df['conservation_score'].std():.3f}")
    report.append(f"Average wild-type probability: {tolerance_df['wt_prob'].mean():.3f} ± {tolerance_df['wt_prob'].std():.3f}")

    if 'avg_log_ratio' in tolerance_df.columns:
        report.append(f"Average log ratio: {tolerance_df['avg_log_ratio'].mean():.3f} ± {tolerance_df['avg_log_ratio'].std():.3f}")
    report.append("")

    # Hotspot details
    if len(hotspots) > 0:
        report.append("=== MUTATION HOTSPOTS ===")
        report.append("(Residues with low mutation tolerance)")
        hotspots_sorted = hotspots.sort_values('hotspot_score', ascending=False)
        for _, row in hotspots_sorted.head(10).iterrows():
            report.append(f"  {row['aa_id']}: {row['wtAA']}{row['pos']} (score: {row['hotspot_score']}, conservation: {row['conservation_score']:.3f})")

    # Flexible positions
    if len(flexible_positions) > 0:
        report.append("\n=== FLEXIBLE POSITIONS ===")
        report.append("(Residues with high mutation tolerance)")
        flexible_sorted = flexible_positions.sort_values('entropy', ascending=False)
        for _, row in flexible_sorted.head(10).iterrows():
            report.append(f"  {row['aa_id']}: {row['wtAA']}{row['pos']} (entropy: {row['entropy']:.3f}, best alt: {row['max_alternative_aa']})")

    # Chain-wise analysis if available
    if 'chain_id' in tolerance_df.columns and tolerance_df['chain_id'].nunique() > 1:
        report.append("\n=== CHAIN-WISE ANALYSIS ===")
        chain_stats = tolerance_df.groupby('chain_id').agg({
            'is_hotspot': 'sum',
            'is_flexible': 'sum',
            'entropy': 'mean',
            'conservation_score': 'mean'
        }).round(3)

        for chain_id, stats in chain_stats.iterrows():
            chain_total = len(tolerance_df[tolerance_df['chain_id'] == chain_id])
            report.append(f"  Chain {chain_id}: {chain_total} residues, {stats['is_hotspot']} hotspots, {stats['is_flexible']} flexible")

    return '\n'.join(report)

def create_visualizations(tolerance_df: pd.DataFrame, output_dir: Path, config: Dict[str, Any]) -> List[Path]:
    """
    Create visualizations of the mutation analysis.

    Returns:
        List of created plot file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []

    # Set matplotlib style
    plt.style.use('default')

    # 1. Entropy vs Conservation scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tolerance_df['entropy'], tolerance_df['conservation_score'],
                         c=tolerance_df['hotspot_score'], cmap='viridis', alpha=0.6)
    plt.xlabel('Entropy (Mutation Tolerance)')
    plt.ylabel('Conservation Score')
    plt.title('Mutation Tolerance vs Conservation')
    plt.colorbar(scatter, label='Hotspot Score')

    plot_file = output_dir / f'entropy_vs_conservation.{config["plot_format"]}'
    plt.savefig(plot_file, dpi=config["plot_dpi"], bbox_inches='tight')
    plt.close()
    plot_files.append(plot_file)

    # 2. Position-wise conservation
    plt.figure(figsize=(12, 6))
    colors = ['red' if x else 'lightblue' for x in tolerance_df['is_hotspot']]
    plt.scatter(tolerance_df['pos'], tolerance_df['conservation_score'], c=colors, alpha=0.7)
    plt.xlabel('Residue Position')
    plt.ylabel('Conservation Score')
    plt.title('Conservation Score by Position (Red = Hotspots)')

    plot_file = output_dir / f'position_conservation.{config["plot_format"]}'
    plt.savefig(plot_file, dpi=config["plot_dpi"], bbox_inches='tight')
    plt.close()
    plot_files.append(plot_file)

    # 3. Entropy distribution
    plt.figure(figsize=(10, 6))
    plt.hist(tolerance_df['entropy'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    mean_entropy = tolerance_df['entropy'].mean()
    plt.axvline(mean_entropy, color='red', linestyle='--', label=f'Mean: {mean_entropy:.3f}')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Distribution of Mutation Tolerance (Entropy)')
    plt.legend()

    plot_file = output_dir / f'entropy_distribution.{config["plot_format"]}'
    plt.savefig(plot_file, dpi=config["plot_dpi"], bbox_inches='tight')
    plt.close()
    plot_files.append(plot_file)

    return plot_files

# ==============================================================================
# Main Function (extracted and simplified from use case)
# ==============================================================================
def run_mutation_hotspot_analysis(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze mutation hotspots from MutCompute prediction results.

    Args:
        input_file: Path to CSV file with MutCompute predictions
        output_dir: Directory to save analysis files (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - tolerance_df: DataFrame with detailed tolerance analysis
            - hotspots: DataFrame with just hotspot residues
            - flexible: DataFrame with just flexible residues
            - report_text: Summary report as text
            - output_files: Dict of created output files
            - metadata: Analysis metadata

    Example:
        >>> result = run_mutation_hotspot_analysis("predictions.csv", "hotspot_analysis")
        >>> print(f"Found {len(result['hotspots'])} hotspots")
    """
    # Setup
    input_file = Path(input_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Validation
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load data
    print(f"Loading MutCompute predictions from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Analyzing {len(df)} residues...")

    # Validate required columns
    required_cols = ['wtAA', 'wt_prob', 'pred_prob', 'pos'] + [f'pr{aa}' for aa in config["amino_acids"]]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Perform analysis
    tolerance_df = analyze_mutation_tolerance(df, config)
    tolerance_df = identify_hotspots(tolerance_df, config["percentile_threshold"])

    # Extract results
    hotspots = tolerance_df[tolerance_df['is_hotspot']].copy()
    flexible_positions = tolerance_df[tolerance_df['is_flexible']].copy()

    # Generate report
    report_text = generate_summary_report(tolerance_df)

    output_files = {}

    # Save outputs if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed analysis
        tolerance_file = output_dir / "tolerance_analysis.csv"
        tolerance_df.to_csv(tolerance_file, index=False)
        output_files["tolerance_analysis"] = str(tolerance_file)

        # Save report
        report_file = output_dir / "hotspot_analysis_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)
        output_files["report"] = str(report_file)

        # Create visualizations
        print("Generating visualizations...")
        plot_files = create_visualizations(tolerance_df, output_dir, config)
        output_files["plots"] = [str(f) for f in plot_files]

        print(f"Analysis complete. Results saved to: {output_dir}")

    # Calculate summary statistics
    summary_stats = {
        "total_residues": len(tolerance_df),
        "hotspots_count": len(hotspots),
        "flexible_count": len(flexible_positions),
        "hotspots_percentage": len(hotspots) / len(tolerance_df) * 100,
        "flexible_percentage": len(flexible_positions) / len(tolerance_df) * 100,
        "avg_entropy": float(tolerance_df['entropy'].mean()),
        "avg_conservation": float(tolerance_df['conservation_score'].mean())
    }

    return {
        "tolerance_df": tolerance_df,
        "hotspots": hotspots,
        "flexible": flexible_positions,
        "report_text": report_text,
        "output_files": output_files,
        "metadata": {
            "input_file": str(input_file),
            "config": config,
            "summary_stats": summary_stats
        }
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
                       help='Input CSV file from MutCompute predictions')
    parser.add_argument('--output_dir', '-o',
                       help='Output directory for analysis files and plots')
    parser.add_argument('--config', '-c',
                       help='Config file (JSON)')
    parser.add_argument('--percentile', '-p', type=float, default=10,
                       help='Percentile threshold for hotspot identification (default: 10)')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip generating plot files')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI arguments
    cli_overrides = {}
    if args.percentile != 10:
        cli_overrides["percentile_threshold"] = args.percentile

    try:
        result = run_mutation_hotspot_analysis(
            input_file=args.input,
            output_dir=args.output_dir,
            config=config,
            **cli_overrides
        )

        # Print report
        print("\n" + result["report_text"])

        # Print key findings
        hotspots = result["hotspots"]
        flexible = result["flexible"]
        stats = result["metadata"]["summary_stats"]

        print(f"\n=== KEY FINDINGS ===")
        print(f"Identified {len(hotspots)} mutation hotspots and {len(flexible)} flexible positions")
        print(f"Hotspots: {stats['hotspots_percentage']:.1f}% of residues")
        print(f"Flexible: {stats['flexible_percentage']:.1f}% of residues")

        if len(hotspots) > 0:
            most_critical = hotspots.loc[hotspots['conservation_score'].idxmax()]
            print(f"Most critical hotspot: {most_critical['wtAA']}{most_critical['pos']} (conservation: {most_critical['conservation_score']:.3f})")

        if len(flexible) > 0:
            most_flexible = flexible.loc[flexible['entropy'].idxmax()]
            print(f"Most flexible position: {most_flexible['wtAA']}{most_flexible['pos']} (entropy: {most_flexible['entropy']:.3f}, best alt: {most_flexible['max_alternative_aa']})")

        if result["output_files"]:
            print(f"\n✅ Output files saved:")
            for key, path in result["output_files"].items():
                if key == "plots":
                    print(f"  Plots: {len(path)} files")
                else:
                    print(f"  {key}: {path}")

        return result

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()