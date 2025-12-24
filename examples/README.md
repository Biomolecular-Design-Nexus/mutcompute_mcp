# MutCompute Examples and Use Cases

This directory contains standalone Python scripts demonstrating different use cases for the MutCompute protein mutation prediction tool.

## Directory Structure

```
examples/
├── README.md                                    # This file
├── use_case_1_protein_mutation_prediction.py   # Single protein analysis
├── use_case_2_batch_mutation_analysis.py       # Batch processing multiple proteins
├── use_case_3_mutation_hotspot_analysis.py     # Hotspot analysis from predictions
├── data/                                        # Demo data files
│   ├── 1y4a_BPN.pdb                            # Sample protein structure
│   └── 1y4a_BPN_mutcompute.csv                 # Pre-computed predictions
└── models/                                      # Model files
    ├── mean_matrix.npy                          # Feature normalization
    ├── stdev_matrix.npy                         # Feature scaling
    └── weights/                                 # Model weights
        ├── weights1/
        ├── weights2/
        └── weights3/
```

## Use Cases Overview

### Use Case 1: Single Protein Mutation Prediction
**File**: `use_case_1_protein_mutation_prediction.py`

Predicts the effect of point mutations on protein stability for a single PDB file.

**What it does**:
- Takes a PDB structure as input
- Generates mutation probability matrices for all residues
- Predicts which amino acid substitutions are most/least favorable
- Outputs comprehensive CSV with probabilities for all 20 amino acids

**Usage**:
```bash
# Basic usage
python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb

# Specify output file
python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --output my_predictions.csv

# Use CPU instead of GPU
python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --cpu
```

**Environment**: `./env_py36` (requires Python 3.6 for Theano/legacy dependencies)

### Use Case 2: Batch Mutation Analysis
**File**: `use_case_2_batch_mutation_analysis.py`

Processes multiple PDB files in batch mode for large-scale analysis.

**What it does**:
- Processes entire directories of PDB files
- Generates summary statistics across all proteins
- Creates batch processing reports
- Handles errors gracefully and continues processing

**Usage**:
```bash
# Process all PDB files in a directory
python use_case_2_batch_mutation_analysis.py --input_dir /path/to/pdb/files --output_dir batch_results

# Process files from a list
python use_case_2_batch_mutation_analysis.py --input_list protein_list.txt --output_dir results

# Use CPU for processing
python use_case_2_batch_mutation_analysis.py --input_dir examples/data --output_dir results --cpu
```

**Environment**: `./env_py36`

### Use Case 3: Mutation Hotspot Analysis
**File**: `use_case_3_mutation_hotspot_analysis.py`

Analyzes mutation predictions to identify critical and flexible regions.

**What it does**:
- Calculates mutation tolerance metrics (entropy, conservation scores)
- Identifies mutation hotspots (critical for protein stability)
- Finds flexible positions (tolerant to mutations)
- Generates visualizations and detailed reports

**Usage**:
```bash
# Analyze existing predictions
python use_case_3_mutation_hotspot_analysis.py --input examples/data/1y4a_BPN_mutcompute.csv

# Generate detailed analysis with plots
python use_case_3_mutation_hotspot_analysis.py --input predictions.csv --output_dir hotspot_analysis

# Adjust sensitivity
python use_case_3_mutation_hotspot_analysis.py --input predictions.csv --percentile 5
```

**Environment**: `./env` (uses standard data science libraries)

## Demo Data

### 1y4a_BPN.pdb
- **Source**: Protein Data Bank structure 1Y4A (chain B)
- **Description**: Bacterial serine protease (subtilisin BPN')
- **Size**: 265 residues
- **Use**: Reference structure for testing mutation predictions

### 1y4a_BPN_mutcompute.csv
- **Description**: Pre-computed MutCompute predictions for 1Y4A chain B
- **Columns**: Residue info, wild-type/predicted amino acids, probabilities for all 20 AAs
- **Use**: Testing hotspot analysis without running full prediction

## Getting Started

1. **Set up environments** (if not already done):
   ```bash
   # Main environment for data analysis
   mamba activate ./env

   # Legacy environment for MutCompute predictions
   mamba activate ./env_py36
   ```

2. **Test with demo data**:
   ```bash
   # Quick test using pre-computed predictions
   mamba activate ./env
   python use_case_3_mutation_hotspot_analysis.py --input examples/data/1y4a_BPN_mutcompute.csv

   # Full prediction pipeline test
   mamba activate ./env_py36
   python use_case_1_protein_mutation_prediction.py --input examples/data/1y4a_BPN.pdb --cpu
   ```

3. **Use your own data**:
   - Place PDB files in a convenient location
   - Run predictions using use case 1 or 2
   - Analyze results using use case 3

## Output Files

### Prediction CSV Format
Each prediction generates a CSV with these columns:
- `aa_id`: Unique residue identifier (PDB_Chain_Position)
- `pdb_id`, `chain_id`, `pos`: Structure identifiers
- `wtAA`, `prAA`: Wild-type and predicted amino acids
- `wt_prob`, `pred_prob`: Probabilities for wild-type and predicted AAs
- `avg_log_ratio`: Average log ratio across ensemble models
- `prALA`, `prARG`, ..., `prVAL`: Probabilities for all 20 amino acids

### Hotspot Analysis Output
- `tolerance_analysis.csv`: Detailed metrics for each residue
- `hotspot_analysis_report.txt`: Human-readable summary
- Visualization plots (PNG files)

## Troubleshooting

### Common Issues

1. **Environment activation fails**:
   ```bash
   # Use full path if needed
   source ./env_py36/bin/activate
   ```

2. **CUDA/GPU errors**:
   ```bash
   # Use CPU mode
   python script.py --cpu
   ```

3. **Import errors**:
   ```bash
   # Ensure you're in the correct environment
   mamba activate ./env_py36
   ```

4. **Missing dependencies**:
   ```bash
   # Install missing packages
   pip install matplotlib seaborn  # for visualization
   ```

## Performance Notes

- **GPU vs CPU**: GPU is ~10-50x faster but requires CUDA setup
- **Memory requirements**: ~4-8GB RAM per protein structure
- **Processing time**: 30-180 seconds per protein (GPU), 10-30 minutes (CPU)
- **Batch processing**: Use case 2 for processing >10 proteins

## Citation

If you use MutCompute in your research, please cite:
[Add appropriate citation when available]