"""
MutCompute structure-based mutation prediction tools.

This MCP Server provides 1 tool:
1. mutcompute_predict: Run MutCompute ensemble inference on a PDB file

MutCompute predicts mutation effects using a 3D-CNN trained on protein structures,
providing per-residue mutation probabilities for all 20 amino acids.
"""

# Standard imports
from typing import Annotated
import subprocess
from pathlib import Path
from fastmcp import FastMCP
from loguru import logger

# Get paths relative to this file
TOOLS_DIR = Path(__file__).resolve().parent
SRC_DIR = TOOLS_DIR.parent
PROJECT_DIR = SRC_DIR.parent

# Paths to MutCompute environment and script
ENV_PY36 = PROJECT_DIR / "env_py36"
MUTCOMPUTE_RUN = PROJECT_DIR / "repo" / "mutcompute" / "run.py"

# MCP server instance
mutcompute_predict_mcp = FastMCP(name="mutcompute_predict")


def run_mutcompute_subprocess(pdb_path: str, output_path: str = None,
                               use_cpu: bool = False, name: str = "run") -> dict:
    """
    Run MutCompute ensemble inference on a PDB file via subprocess.

    Args:
        pdb_path: Path to the input PDB file
        output_path: Optional output CSV file path
        use_cpu: If True, use CPU instead of GPU
        name: Name of run for logging

    Returns:
        dict with status and results
    """
    logger.info(f"Starting MutCompute subprocess for PDB: {pdb_path}")

    # Verify paths exist
    if not ENV_PY36.exists():
        logger.error(f"MutCompute Python 3.6 environment not found: {ENV_PY36}")
        raise FileNotFoundError(f"MutCompute Python 3.6 environment not found: {ENV_PY36}")

    if not MUTCOMPUTE_RUN.exists():
        logger.error(f"MutCompute run.py not found: {MUTCOMPUTE_RUN}")
        raise FileNotFoundError(f"MutCompute run.py not found: {MUTCOMPUTE_RUN}")

    pdb_file = Path(pdb_path).resolve()
    if not pdb_file.exists():
        logger.error(f"PDB file not found: {pdb_file}")
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    # Build command
    python_exe = ENV_PY36 / "bin" / "python"
    cmd = [str(python_exe), str(MUTCOMPUTE_RUN), "-p", str(pdb_file), "-n", name]

    if output_path:
        cmd.extend(["-o", str(output_path)])

    if use_cpu:
        cmd.append("--cpu")

    logger.debug(f"Executing command: {' '.join(cmd)}")

    # Run the subprocess and capture output
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    logger.info(f"MutCompute subprocess completed with return code: {result.returncode}")

    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


@mutcompute_predict_mcp.tool
def mutcompute_predict(
    pdb_path: Annotated[str, "Path to the input PDB file"],
    output_path: Annotated[str | None, "Output CSV file path. If None, uses <pdb_name>_mutcompute.csv"] = None,
    use_cpu: Annotated[bool, "Use CPU instead of GPU for inference"] = False,
    name: Annotated[str, "Name of run for logging"] = "run",
) -> dict:
    """
    Run MutCompute ensemble inference on a PDB file.

    This tool:
    1. Runs MutCompute 3D-CNN ensemble on a protein structure
    2. Predicts mutation probabilities for all 20 amino acids at each position
    3. Outputs results as a CSV file

    MutCompute uses a deep learning model trained on protein structures to predict
    which amino acid substitutions are likely to be tolerated or beneficial.

    Input: PDB structure file
    Output: Dictionary with status, output path, and execution details
    """
    logger.info(f"mutcompute_predict called with pdb_path={pdb_path}, use_cpu={use_cpu}, name={name}")

    try:
        pdb_file = Path(pdb_path).resolve()

        # Determine output path
        if output_path is None:
            output_file = pdb_file.parent / f"{pdb_file.stem}_mutcompute.csv"
        else:
            output_file = Path(output_path).resolve()

        logger.info(f"Output will be written to: {output_file}")

        # Run MutCompute
        result = run_mutcompute_subprocess(
            pdb_path=str(pdb_file),
            output_path=str(output_file),
            use_cpu=use_cpu,
            name=name
        )

        # Check if output file was created
        if output_file.exists():
            import pandas as pd
            df = pd.read_csv(output_file)
            num_positions = len(df)
            logger.info(f"Output file created with {num_positions} positions")
        else:
            num_positions = None
            logger.warning("Output file was not created")

        logger.info(f"mutcompute_predict completed successfully for {pdb_file}")

        return {
            "status": "success",
            "pdb_file": str(pdb_file),
            "output_csv": str(output_file),
            "num_positions": num_positions,
            "device": "CPU" if use_cpu else "GPU",
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"MutCompute failed with return code {e.returncode}: {e.stderr}")
        return {
            "status": "error",
            "error_message": f"MutCompute failed with return code {e.returncode}",
            "pdb_path": str(pdb_path),
            "stderr": e.stderr,
        }

    except Exception as e:
        logger.exception(f"Unexpected error in mutcompute_predict: {e}")
        return {
            "status": "error",
            "error_message": str(e),
            "pdb_path": str(pdb_path),
        }
