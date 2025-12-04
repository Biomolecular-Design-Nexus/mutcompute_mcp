#!/usr/bin/env python3
"""
Script to run MutCompute as a local service using subprocess.
This script uses the main environment (env) to call MutCompute which runs in env_py36.
"""

import subprocess
import argparse
import sys
from pathlib import Path
from loguru import logger

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Paths to environments
ENV_PY36 = PROJECT_DIR / "env_py36"
MUTCOMPUTE_RUN = PROJECT_DIR / "repo" / "mutcompute" / "run.py"

# Configure loguru logger
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def run_mutcompute(pdb_path, output_path=None, use_cpu=False, name="run"):
    """
    Run MutCompute ensemble inference on a PDB file.

    Args:
        pdb_path: Path to the input PDB file
        output_path: Optional output CSV file path
        use_cpu: If True, use CPU instead of GPU
        name: Name of run for logging

    Returns:
        subprocess.CompletedProcess object
    """
    # Verify paths exist
    if not ENV_PY36.exists():
        logger.error(f"Environment not found: {ENV_PY36}")
        raise FileNotFoundError(f"Environment not found: {ENV_PY36}")

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

    logger.info(f"Running MutCompute with command: {' '.join(cmd)}")
    logger.info(f"PDB file: {pdb_file}")
    logger.info(f"Output: {output_path if output_path else f'{pdb_file.stem}_mutcompute.csv'}")
    logger.info(f"Device: {'CPU' if use_cpu else 'GPU'}")

    try:
        # Run the subprocess and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        # Log stdout
        if result.stdout:
            logger.info("MutCompute stdout:")
            for line in result.stdout.splitlines():
                logger.info(f"  {line}")

        # Log stderr (warnings, info from MutCompute)
        if result.stderr:
            logger.info("MutCompute stderr:")
            for line in result.stderr.splitlines():
                logger.info(f"  {line}")

        logger.success("MutCompute completed successfully")
        return result

    except subprocess.CalledProcessError as e:
        logger.error(f"MutCompute failed with return code {e.returncode}")

        if e.stdout:
            logger.error("MutCompute stdout:")
            for line in e.stdout.splitlines():
                logger.error(f"  {line}")

        if e.stderr:
            logger.error("MutCompute stderr:")
            for line in e.stderr.splitlines():
                logger.error(f"  {line}")

        raise

    except Exception as e:
        logger.error(f"Unexpected error running MutCompute: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run MutCompute ensemble inference using subprocess.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_mutcompute.py -p protein.pdb
  python scripts/run_mutcompute.py -p ./example/my_protein.pdb -o ./results/predictions.csv
  python scripts/run_mutcompute.py -p ../structures/1abc.pdb --cpu
"""
    )

    parser.add_argument('-p', '--pdb', required=True,
                        help='REQUIRED. Path to the input PDB file (relative or absolute).')

    parser.add_argument('-o', '--output', required=False, default=None, type=str,
                        help='OPTIONAL. Output CSV file path. '
                             'Default: {pdb_name}_mutcompute.csv in the same directory as input PDB')

    parser.add_argument('-n', '--name', required=False, default='run', type=str,
                        help='OPTIONAL. Name of run for logging. Default: run')

    parser.add_argument('--cpu', action='store_true',
                        help='OPTIONAL. Use CPU instead of GPU for inference. Default: Use CUDA GPU')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("Starting MutCompute subprocess execution")
    logger.info("="*80)

    try:
        run_mutcompute(
            pdb_path=args.pdb,
            output_path=args.output,
            use_cpu=args.cpu,
            name=args.name
        )
        logger.info("="*80)
        logger.info("MutCompute execution completed")
        logger.info("="*80)

    except Exception as e:
        logger.error("="*80)
        logger.error(f"MutCompute execution failed: {e}")
        logger.error("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
