# mutcompute MCP

> A Model Context Protocol (MCP) server for protein mutation effect prediction using neural networks

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The MutCompute MCP server provides both synchronous and asynchronous APIs for protein mutation analysis. It predicts the effect of point mutations on protein stability by analyzing 3D protein structures and generating comprehensive mutation probability matrices using neural network ensemble models.

### Features
- **Neural Network Prediction**: 3D CNN ensemble for mutation effect prediction
- **Fast Analysis**: Hotspot identification and log-likelihood calculations
- **Batch Processing**: Handle multiple protein structures efficiently
- **Background Jobs**: Long-running tasks with progress tracking and job management
- **Dual Environment**: Supports both legacy Python 3.6 (neural networks) and modern Python 3.10 (analysis)

### Directory Structure
```
./
├── README.md                   # This file
├── env/                        # Conda environment (Python 3.10)
├── env_py36/                   # Legacy environment (Python 3.6 for neural networks)
├── src/
│   └── server.py               # MCP server
├── scripts/
│   ├── protein_mutation_prediction.py    # Neural network prediction
│   ├── mutation_hotspot_analysis.py      # Hotspot analysis
│   ├── mutcompute_llh.py                # Log-likelihood calculation
│   ├── run_mutcompute.py                # Full pipeline
│   └── lib/                             # Shared utilities
├── examples/
│   └── data/                   # Demo data
│       ├── 1y4a_BPN.pdb       # Sample protein structure
│       ├── 1y4a_BPN_mutcompute.csv  # Pre-computed predictions
│       └── wt.fasta           # Wild-type sequence
├── configs/                    # Configuration files
│   ├── protein_mutation_prediction_config.json
│   └── mutation_hotspot_analysis_config.json
└── repo/                       # Original repository with neural network models
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+ for MCP server
- Python 3.6.12 for neural network compatibility
- CUDA-compatible GPU (optional, CPU mode available)

### Create Environment
The environments are already set up in this installation. To recreate them:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/mutcompute_mcp

# Create main MCP environment (Python 3.10)
mamba create -p ./env python=3.10 pip -y
# or: conda create -p ./env python=3.10 pip -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install MCP dependencies
pip install fastmcp loguru pandas numpy matplotlib seaborn scipy tqdm

# Create legacy environment for neural networks (Python 3.6)
mamba env create -f repo/mutcompute/environment.yaml -p ./env_py36 -y
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/protein_mutation_prediction.py` | Predict mutation effects from PDB structures | See below |
| `scripts/mutation_hotspot_analysis.py` | Analyze hotspots from prediction results | See below |
| `scripts/mutcompute_llh.py` | Calculate log-likelihood scores | See below |
| `scripts/run_mutcompute.py` | Run complete MutCompute pipeline | See below |

### Script Examples

#### Protein Mutation Prediction

```bash
# Activate legacy environment for neural networks
mamba activate ./env_py36

# Single protein prediction
python scripts/protein_mutation_prediction.py \
  --input examples/data/1y4a_BPN.pdb \
  --output results/predictions.csv \
  --cpu

# Batch processing
python scripts/protein_mutation_prediction.py \
  --input examples/data \
  --batch \
  --output_dir results/batch \
  --cpu
```

**Parameters:**
- `--input, -i`: PDB file or directory (required)
- `--output, -o`: Output CSV file path (default: auto-generated)
- `--batch`: Enable batch processing mode
- `--output_dir`: Output directory for batch mode (default: results/)
- `--cpu`: Use CPU instead of GPU
- `--config`: Configuration file path (optional)

#### Mutation Hotspot Analysis

```bash
# Activate modern environment
mamba activate ./env

# Analyze hotspots with visualizations
python scripts/mutation_hotspot_analysis.py \
  --input examples/data/1y4a_BPN_mutcompute.csv \
  --output_dir results/hotspots \
  --percentile 10
```

**Parameters:**
- `--input, -i`: CSV file with MutCompute predictions (required)
- `--output_dir, -o`: Output directory (default: current directory)
- `--percentile, -p`: Percentile threshold for hotspot detection (default: 10.0)
- `--config`: Configuration file path (optional)

#### Log-Likelihood Calculation

```bash
# Calculate mutation likelihoods
python scripts/mutcompute_llh.py \
  --input examples/data/variants.csv \
  --mutcompute examples/data/1y4a_BPN_mutcompute.csv \
  --wt_fasta examples/data/wt.fasta
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name mutcompute
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add mutcompute -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "mutcompute": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/mutcompute_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/mutcompute_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from mutcompute?
```

#### Basic Validation
```
Use validate_pdb_file with pdb_file "@examples/data/1y4a_BPN.pdb"
```

#### Fast Analysis (Sync)
```
Use analyze_mutation_hotspots with input_file "@examples/data/1y4a_BPN_mutcompute.csv"
```

#### Long-Running Prediction (Async)
```
Submit mutation prediction for @examples/data/1y4a_BPN.pdb
Then check the job status
```

#### Batch Processing
```
Submit batch processing for these files:
- @examples/data/1y4a_BPN.pdb
```

#### Job Management
```
List all my jobs
Check status of job "abc123"
Get results for job "abc123"
View logs for job "abc123" with tail 20
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/1y4a_BPN.pdb` | Reference a specific PDB file |
| `@configs/protein_mutation_prediction_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "mutcompute": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/mutcompute_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/mutcompute_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use analyze_mutation_hotspots with input_file "examples/data/1y4a_BPN_mutcompute.csv"
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `analyze_mutation_hotspots` | Identify critical/flexible protein regions | `input_file`, `output_dir`, `percentile`, `config_file` |
| `calculate_mutation_llh` | Calculate log-likelihood scores for mutations | `input_file`, `wt_fasta_path`, `mutcompute_path`, `output_file` |
| `validate_pdb_file` | Validate PDB file compatibility | `pdb_file` |
| `list_example_data` | List available example files | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_mutation_prediction` | Predict mutation effects for PDB | `input_file`, `output_file`, `use_cpu`, `config_file`, `job_name` |
| `submit_batch_mutation_prediction` | Batch processing multiple PDBs | `input_files`, `output_dir`, `use_cpu`, `config_file`, `job_name` |
| `submit_mutcompute_pipeline` | Full MutCompute pipeline | `input_file`, `output_path`, `use_cpu`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs with optional tail |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs with optional status filter |

---

## Examples

### Example 1: Mutation Hotspot Analysis

**Goal:** Identify critical and flexible residues in a protein

**Using Script:**
```bash
mamba activate ./env
python scripts/mutation_hotspot_analysis.py \
  --input examples/data/1y4a_BPN_mutcompute.csv \
  --output_dir results/example1/
```

**Using MCP (in Claude Code):**
```
Use analyze_mutation_hotspots to process @examples/data/1y4a_BPN_mutcompute.csv and save results to results/example1/
```

**Expected Output:**
- `tolerance_analysis.csv`: Detailed tolerance metrics for 275 residues
- `hotspot_analysis_report.txt`: Human-readable summary identifying ~28 hotspots and 28 flexible positions
- `*.png`: Visualization plots (entropy distribution, conservation analysis, position plots)

### Example 2: Protein Mutation Prediction

**Goal:** Generate mutation probability matrix for a protein structure

**Using Script:**
```bash
mamba activate ./env_py36
python scripts/protein_mutation_prediction.py \
  --input examples/data/1y4a_BPN.pdb \
  --output results/example2/predictions.csv \
  --cpu
```

**Using MCP (in Claude Code):**
```
Submit mutation prediction for @examples/data/1y4a_BPN.pdb with use_cpu true
```

**Expected Output:**
- CSV file with 20 amino acid probabilities for each of 275 residues
- Processing time: ~71 seconds (CPU mode)
- File size: ~129KB

### Example 3: Batch Processing

**Goal:** Process multiple PDB files at once

**Using Script:**
```bash
mamba activate ./env_py36
python scripts/protein_mutation_prediction.py \
  --input examples/data \
  --batch \
  --output_dir results/batch/ \
  --cpu
```

**Using MCP (in Claude Code):**
```
Submit batch processing for all PDB files in @examples/data/ with output_dir "results/batch/"
```

### Example 4: Log-Likelihood Calculation

**Goal:** Calculate mutation likelihoods for specific variants

**Using Script:**
```bash
mamba activate ./env
python scripts/mutcompute_llh.py \
  --input variants.csv \
  --mutcompute examples/data/1y4a_BPN_mutcompute.csv \
  --wt_fasta examples/data/wt.fasta
```

**Using MCP (in Claude Code):**
```
Use calculate_mutation_llh with input_file "variants.csv" and mutcompute_path "@examples/data/1y4a_BPN_mutcompute.csv"
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `1y4a_BPN.pdb` | Bacterial serine protease structure (275 residues) | Mutation prediction tools |
| `1y4a_BPN_mutcompute.csv` | Pre-computed mutation predictions | Hotspot analysis, LLH calculation |
| `wt.fasta` | Wild-type protein sequence | Log-likelihood calculation |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `protein_mutation_prediction_config.json` | Neural network prediction settings | `use_cpu`, `theano_flags`, `output_suffix` |
| `mutation_hotspot_analysis_config.json` | Hotspot analysis parameters | `percentile_threshold`, `amino_acids`, `plot_settings` |

### Config Example

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

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
pip install fastmcp loguru pandas numpy matplotlib
```

**Problem:** Import errors
```bash
# Verify installation
python -c "from src.server import mcp; print('Server OK')"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove mutcompute
claude mcp add mutcompute -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
python src/server.py
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

### Performance Issues

**Problem:** Prediction takes too long
- Use CPU mode for development (`use_cpu: true`)
- GPU mode requires CUDA-compatible hardware
- Large proteins (>500 residues) may take >30 minutes

**Problem:** Out of memory
- Neural network prediction requires ~6GB RAM
- Close other applications during processing
- Use batch processing for multiple small proteins instead of large ones

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test server startup
python src/server.py

# Test individual scripts
python scripts/mutation_hotspot_analysis.py --input examples/data/1y4a_BPN_mutcompute.csv
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
fastmcp dev src/server.py
```

### Environment Requirements

#### Main Environment (./env - Python 3.10)
- FastMCP, loguru for MCP server
- pandas, numpy, matplotlib for analysis
- Standard data science libraries

#### Legacy Environment (./env_py36 - Python 3.6)
- Theano, Keras for neural networks
- BioPython for structure processing
- MutCompute model dependencies

---

## Performance Benchmarks

Tested with 1Y4A protein (275 residues):
- **Neural Network Prediction**: ~71 seconds (CPU mode)
- **Hotspot Analysis**: ~2 seconds
- **Memory usage**: ~6GB RAM during neural network inference
- **Output size**: ~50KB CSV per protein
- **Success rate**: 100% (all test cases passed)

---

## License

Based on the MutCompute repository for protein mutation effect prediction.

## Credits

Based on [MutCompute](repo/mutcompute/) - Neural network ensemble for protein mutation analysis