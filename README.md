# MutCompute MCP

MutCompute MCP server for protein modeling.

## Overview
This MutCompute MCP server support two tools: 1. Mutation recommendations for each site; 2. Likelihood calculation given variants.

## Installation
### Create MCP environment
```bash
# Create and activate virtual environment
mamba env create -p ./env python=3.10 pip -y
mamba activate ./env
pip install loguru sniffio pandas numpy tqdm scipy

pip install --ignore-installed fastmcp
```
### Create MutCompute environment
```bash
mamba env create -f repo/mutcompute/environment.yaml -p ./env_py36 -y
mamba activate ./env_py36
mamba install -c conda-forge cudnn -y

# build box builder
cd repo/mutcompute/src/box_builder
python3 setup.py build_ext --inplace

# Install free sasa
wget https://freesasa.github.io/freesasa-2.1.2.zip
unzip freesasa-2.1.2.zip
cd freesasa-2.1.2
./configure  --disable-json --disable-xml
sudo apt install libc++-dev clang libjson-c-dev
mamba install -c conda-forge libcxx -y
export LD_LIBRARY_PATH=/usr/lib/llvm-14/lib/:$LD_LIBRARY_PATH
export PATH=/usr/lib/llvm-14/lib/:$PATH
sed -i 's/-lc++/-lstdc++/g' src/Makefile
CC=/usr/bin/gcc CXX=/usr/bin/g++ ./configure --prefix=/usr/local
make
sudo make install
```
## Local usage
### Run MutCompute to obtain the mutational probability file given a PDB file
```shell
# Activate the main environment
mamba activate ./env
python scripts/run_mutcompute.py -p example/wt_struct.pdb
```

The script will:
1. Use the main `env` environment to call MutCompute
2. Execute MutCompute in the `env_py36` environment via subprocess
3. Generate PQR and SASA files automatically
4. Run ensemble inference with 3 model weights
5. Save predictions to CSV (default: `{pdb_name}_mutcompute.csv`)
6. Clean up temporary files automatically
7. Display detailed logs using loguru

### Calculate the MutCompute likelihood based on the mutational probabilities
```shell
python scripts/mutcompute_llh.py -i example/data.csv -m example/wt_struct_mutcompute.csv --seq_col seq
```

## MCP usage

### Install MCP Server
```shell
fastmcp install claude-code mcp-servers/mutcompute_mcp/src/mutcompute_mcp.py --python mcp-servers/mutcompute_mcp/env/bin/python
fastmcp install gemini-cli mcp-servers/mutcompute_mcp/src/mutcompute_mcp.py --python mcp-servers/mutcompute_mcp/env/bin/python
```

### Call MCP service
### 1. Calculate Mutation Probability Matrix
```markdown
Can you help run MutCompute data @examples/case2.1_subtilisin/wt_struct.pdb using the mutcompute server.

Please convert the relative path to absolution path before calling the MCP servers. 
```

### 2. Estimate MutCompute likelihoods
```markdown
Can you estimate the MutCompute likelihoods for data @examples/case2.1_subtilisin/data.pdb with precalculated probability matrix @examples/case2.1_subtilisin/wt_struct_mutcompute.csv using the mutcompute mcp server.

Please convert the relative path to absolution path before calling the MCP servers. 
```
