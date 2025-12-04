"""
Model Context Protocol (MCP) for MutCompute

This MCP server provides structure-based protein mutation analysis tools using MutCompute.
MutCompute uses a 3D convolutional neural network trained on protein structures to predict
which amino acid substitutions are likely to be tolerated or beneficial at each position.

This MCP Server contains tools extracted from the following scripts:
1. mutcompute_predict
    - mutcompute_predict: Run MutCompute ensemble inference on a PDB file to get mutation probabilities
2. mutcompute_llh
    - mutcompute_calculate_llh: Calculate log-likelihood ratios for mutations using MutCompute probabilities
"""

from loguru import logger
from fastmcp import FastMCP
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Import tool modules
from tools.mutcompute_predict import mutcompute_predict_mcp
from tools.mutcompute_llh import mutcompute_llh_mcp

# Server definition and mounting
mcp = FastMCP(name="mutcompute")
logger.info("Mounting mutcompute_predict tool")
mcp.mount(mutcompute_predict_mcp)
logger.info("Mounting mutcompute_llh tool")
mcp.mount(mutcompute_llh_mcp)

if __name__ == "__main__":
    logger.info("Starting MutCompute MCP server")
    mcp.run()
