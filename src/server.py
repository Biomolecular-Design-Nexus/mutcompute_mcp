"""MCP Server for mutcompute

Provides both synchronous and asynchronous (submit) APIs for protein mutation analysis tools.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

try:
    from fastmcp import FastMCP
    from jobs.manager import job_manager
except ImportError as e:
    print(f"Error: {e}")
    print("Please install required dependencies:")
    print("mamba run -p ./env pip install fastmcp loguru")
    sys.exit(1)

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Create MCP server
mcp = FastMCP("mutcompute")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def analyze_mutation_hotspots(
    input_file: str,
    output_dir: Optional[str] = None,
    percentile: float = 10.0,
    config_file: Optional[str] = None
) -> dict:
    """
    Analyze mutation hotspots from MutCompute prediction results.

    This is a fast operation that identifies critical and flexible regions
    in proteins based on mutation tolerance analysis.

    Args:
        input_file: Path to CSV file with MutCompute predictions
        output_dir: Directory to save analysis outputs (optional)
        percentile: Percentile threshold for hotspot detection (default: 10.0)
        config_file: Optional path to config JSON file

    Returns:
        Dictionary with analysis results and output file paths
    """
    try:
        # Import the analysis function
        from mutation_hotspot_analysis import run_mutation_hotspot_analysis

        result = run_mutation_hotspot_analysis(
            input_file=input_file,
            output_dir=output_dir,
            percentile=percentile,
            config_file=config_file
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"Hotspot analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def calculate_mutation_llh(
    input_file: str,
    wt_fasta_path: Optional[str] = None,
    mutcompute_path: Optional[str] = None,
    output_file: Optional[str] = None
) -> dict:
    """
    Calculate log-likelihood scores for mutations based on MutCompute predictions.

    This is a fast operation that computes likelihood ratios for observed
    mutations compared to expected mutation patterns.

    Args:
        input_file: Path to variant sequences CSV file
        wt_fasta_path: Path to wild-type FASTA file (optional, auto-detected)
        mutcompute_path: Path to MutCompute predictions CSV (optional, auto-detected)
        output_file: Path to save LLH results (optional, auto-generated)

    Returns:
        Dictionary with LLH calculations and output file path
    """
    try:
        # Import the LLH calculation function
        from mutcompute_llh import run_mutcompute_llh

        result = run_mutcompute_llh(
            seq_path=input_file,
            wt_fasta_path=wt_fasta_path,
            mutcompute_path=mutcompute_path,
            output_path=output_file
        )
        return {"status": "success", **result}
    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"LLH calculation failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_mutation_prediction(
    input_file: str,
    output_file: Optional[str] = None,
    use_cpu: bool = True,
    config_file: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit protein mutation prediction for background processing.

    This operation uses neural networks and may take >10 minutes. Returns a job_id for tracking.
    Uses the env_py36 environment for Theano/neural network compatibility.

    Args:
        input_file: Path to PDB file for mutation prediction
        output_file: Path to save prediction CSV (optional, auto-generated)
        use_cpu: Use CPU instead of GPU (default: True for compatibility)
        config_file: Optional path to config JSON file
        job_name: Optional name for tracking this job

    Returns:
        Dictionary with job_id. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "protein_mutation_prediction.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": input_file,
            "output": output_file,
            "cpu": use_cpu,
            "config": config_file
        },
        job_name=job_name or f"predict_{Path(input_file).stem}",
        environment="env_py36"  # Use Python 3.6 environment for neural networks
    )

@mcp.tool()
def submit_batch_mutation_prediction(
    input_files: List[str],
    output_dir: Optional[str] = None,
    use_cpu: bool = True,
    config_file: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch mutation prediction for multiple PDB files.

    Processes multiple protein structures in a single job. Suitable for:
    - Processing many PDB files at once
    - Large-scale mutation analysis
    - Batch processing with consistent parameters

    Args:
        input_files: List of PDB file paths to process
        output_dir: Directory to save all prediction outputs
        use_cpu: Use CPU instead of GPU (default: True)
        config_file: Optional path to config JSON file
        job_name: Optional name for tracking this batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    script_path = str(SCRIPTS_DIR / "protein_mutation_prediction.py")

    # Convert list to comma-separated string for CLI
    inputs_str = ",".join(input_files)

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input_files": inputs_str,
            "batch": True,
            "output_dir": output_dir,
            "cpu": use_cpu,
            "config": config_file
        },
        job_name=job_name or f"batch_{len(input_files)}_files",
        environment="env_py36"
    )

@mcp.tool()
def submit_mutcompute_pipeline(
    input_file: str,
    output_path: Optional[str] = None,
    use_cpu: bool = True,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit full MutCompute pipeline for background processing.

    Runs the complete mutation analysis pipeline including structure processing
    and neural network prediction. May take >10 minutes depending on protein size.

    Args:
        input_file: Path to PDB file for complete analysis
        output_path: Path to save pipeline results (optional, auto-generated)
        use_cpu: Use CPU instead of GPU (default: True)
        job_name: Optional name for tracking this job

    Returns:
        Dictionary with job_id for tracking the pipeline execution
    """
    script_path = str(SCRIPTS_DIR / "run_mutcompute.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "pdb_path": input_file,
            "output_path": output_path,
            "use_cpu": use_cpu
        },
        job_name=job_name or f"pipeline_{Path(input_file).stem}",
        environment="env_py36"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_pdb_file(pdb_file: str) -> dict:
    """
    Validate a PDB file for compatibility with MutCompute.

    Args:
        pdb_file: Path to PDB file to validate

    Returns:
        Dictionary with validation results
    """
    try:
        pdb_path = Path(pdb_file)
        if not pdb_path.exists():
            return {"status": "error", "error": f"PDB file not found: {pdb_file}"}

        # Basic validation
        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        atom_lines = [line for line in lines if line.startswith('ATOM')]
        if not atom_lines:
            return {"status": "error", "error": "No ATOM records found in PDB file"}

        # Count residues
        residues = set()
        for line in atom_lines:
            if len(line) >= 26:
                res_num = line[22:26].strip()
                chain = line[21].strip()
                residues.add((chain, res_num))

        return {
            "status": "success",
            "file_size": pdb_path.stat().st_size,
            "total_atoms": len(atom_lines),
            "unique_residues": len(residues),
            "chains": len(set(res[0] for res in residues)),
            "message": f"Valid PDB file with {len(residues)} residues"
        }

    except Exception as e:
        return {"status": "error", "error": f"Validation failed: {str(e)}"}

@mcp.tool()
def list_example_data() -> dict:
    """
    List available example data files for testing.

    Returns:
        Dictionary with available example files and their descriptions
    """
    try:
        examples_dir = MCP_ROOT / "examples" / "data"

        if not examples_dir.exists():
            return {"status": "error", "error": "Examples directory not found"}

        files = []
        for file_path in examples_dir.iterdir():
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "type": "PDB structure" if file_path.suffix == ".pdb" else
                           "CSV predictions" if file_path.suffix == ".csv" else
                           "Other"
                })

        return {
            "status": "success",
            "example_files": files,
            "examples_dir": str(examples_dir)
        }

    except Exception as e:
        return {"status": "error", "error": f"Failed to list examples: {str(e)}"}

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()