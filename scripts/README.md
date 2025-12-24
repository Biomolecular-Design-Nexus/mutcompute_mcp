# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config | Independent |
|--------|-------------|----------------|--------|-------------|
| `protein_mutation_prediction.py` | Predict mutation effects from PDB | Yes (MutCompute model) | `configs/protein_mutation_prediction_config.json` | No |
| `mutation_hotspot_analysis.py` | Analyze mutation hotspots from predictions | No | `configs/mutation_hotspot_analysis_config.json` | Yes |

## Dependencies Summary

### protein_mutation_prediction.py
- **Essential**: numpy, pandas, pathlib, json
- **Repo Required**: MutCompute neural network models (lazy loaded)
- **Environment**: Python 3.6 (`./env_py36`) for neural network compatibility
- **GPU/CPU**: Configurable via Theano flags

### mutation_hotspot_analysis.py
- **Essential**: numpy, pandas, matplotlib, pathlib, json
- **Repo Required**: None - fully self-contained
- **Environment**: Python 3.10+ (`./env`)
- **Visualization**: matplotlib only (seaborn removed)

## Usage

### Environment Setup

```bash
# For mutation prediction (requires older Python for neural networks)
mamba run -p ./env_py36 python scripts/protein_mutation_prediction.py [options]

# For hotspot analysis (modern Python)
mamba run -p ./env python scripts/mutation_hotspot_analysis.py [options]
```

### Single Protein Prediction

```bash
# Basic prediction with auto-generated output
mamba run -p ./env_py36 python scripts/protein_mutation_prediction.py \
  --input examples/data/1y4a_BPN.pdb \
  --output results/predictions.csv \
  --cpu

# With custom config
mamba run -p ./env_py36 python scripts/protein_mutation_prediction.py \
  --input examples/data/1y4a_BPN.pdb \
  --output results/predictions.csv \
  --config configs/protein_mutation_prediction_config.json
```

### Batch Processing

```bash
# Process all PDB files in a directory
mamba run -p ./env_py36 python scripts/protein_mutation_prediction.py \
  --input examples/data \
  --batch \
  --output_dir results/batch \
  --cpu
```

### Hotspot Analysis

```bash
# Analyze hotspots with visualization
mamba run -p ./env python scripts/mutation_hotspot_analysis.py \
  --input examples/data/1y4a_BPN_mutcompute.csv \
  --output_dir results/hotspots

# Custom sensitivity threshold
mamba run -p ./env python scripts/mutation_hotspot_analysis.py \
  --input examples/data/1y4a_BPN_mutcompute.csv \
  --output_dir results/hotspots \
  --percentile 5
```

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving, PDB file discovery

## Configuration Files

Each script can be configured via JSON files in `configs/`:

### protein_mutation_prediction_config.json
```json
{
  "computation": {
    "use_cpu": false,
    "theano_flags_cpu": "device=cpu,floatX=float32",
    "theano_flags_gpu": "device=cuda,floatX=float32"
  },
  "output": {
    "suffix": "_mutcompute.csv",
    "include_metadata": true
  }
}
```

### mutation_hotspot_analysis_config.json
```json
{
  "analysis": {
    "percentile_threshold": 10,
    "amino_acids": ["ALA", "ARG", ...]
  },
  "visualization": {
    "plot_dpi": 300,
    "plot_format": "png"
  }
}
```

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from scripts.protein_mutation_prediction import run_protein_mutation_prediction
from scripts.mutation_hotspot_analysis import run_mutation_hotspot_analysis

# In MCP tool:
@mcp.tool()
def predict_mutations(input_file: str, output_file: str = None):
    return run_protein_mutation_prediction(input_file, output_file)

@mcp.tool()
def analyze_hotspots(input_file: str, output_dir: str = None):
    return run_mutation_hotspot_analysis(input_file, output_dir)
```

## Script Return Format

All scripts return consistent dictionaries:

```python
{
    "result": data,           # Primary result (DataFrame or dict)
    "output_file": "path",    # Path to saved file(s)
    "metadata": {
        "input_file": "path",
        "config": {...},
        "summary_stats": {...}
    }
}
```

## Input/Output Formats

### Inputs
- **PDB files**: Standard Protein Data Bank format
- **CSV files**: MutCompute prediction output format

### Outputs
- **Prediction CSV**: Mutation probability matrix for all residues
- **Analysis CSV**: Detailed tolerance metrics per residue
- **Report TXT**: Human-readable summary
- **PNG plots**: Visualization of results

## Testing

Scripts have been verified to work with the demo data:

```bash
# Test hotspot analysis (fully independent)
mamba run -p ./env python scripts/mutation_hotspot_analysis.py \
  --input examples/data/1y4a_BPN_mutcompute.csv \
  --output_dir results/test_hotspot

# Test protein prediction (requires repo models)
mamba run -p ./env_py36 python scripts/protein_mutation_prediction.py \
  --input examples/data/1y4a_BPN.pdb \
  --output results/test_prediction.csv \
  --cpu
```

## Simplifications Made

### From Original Use Cases
1. **Removed unused imports**: datetime, sys (where not needed)
2. **Inlined simple functions**: Path validation, output generation
3. **Unified error handling**: Consistent exception patterns
4. **Consolidated configuration**: External JSON configs vs hardcoded values
5. **Simplified visualization**: matplotlib only, removed seaborn dependency
6. **Added batch support**: Both scripts support single/batch modes
7. **Lazy loading**: Heavy imports only when needed
8. **Absolute paths**: Fixed path resolution issues from original scripts

### Repo Dependencies
- **protein_mutation_prediction.py**: Still requires repo/mutcompute for neural network models (cannot be easily extracted)
- **mutation_hotspot_analysis.py**: Fully independent, no repo dependencies

The scripts maintain full compatibility with the original functionality while being cleaner and more suitable for MCP tool wrapping.