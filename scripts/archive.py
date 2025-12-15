"""
Archive management module for keyword network analysis runs
Handles UUID generation, directory creation, and run metadata tracking
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


class RunArchive:
    """Manages archiving of analysis runs with UUID-based directories"""

    def __init__(self, base_archive_dir: str = "exports/runs"):
        """
        Initialize the archive manager

        Parameters:
        -----------
        base_archive_dir : str
            Base directory for storing archived runs
        """
        self.base_archive_dir = Path(base_archive_dir)
        self.run_uuid = None
        self.run_id = None
        self.run_dir = None
        self.manifest = None
        self.start_time = None
        self.errors = []
        self.warnings = []

    def initialize_run(self) -> str:
        """
        Initialize a new run with UUID and create directory structure

        Returns:
        --------
        str : Path to the run directory
        """
        # Generate UUID and timestamp
        self.run_uuid = str(uuid.uuid4())
        short_uuid = self.run_uuid.split('-')[0]
        
        timestamp = datetime.now(timezone.utc)
        self.start_time = timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create run ID with format: YYYY-MM-DD_HH-MM-SS_<short-uuid>
        self.run_id = f"{timestamp_str}_{short_uuid}"
        
        # Create run directory
        self.run_dir = self.base_archive_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize manifest
        self.manifest = {
            "run_id": self.run_id,
            "uuid": self.run_uuid,
            "timestamp_start": timestamp.isoformat(),
            "timestamp_end": None,
            "duration_seconds": None,
            "configuration": {},
            "results_summary": {
                "total_posts_collected": 0,
                "analyses": {}
            },
            "files_generated": [],
            "errors": [],
            "warnings": []
        }
        
        # Save initial manifest
        self._save_manifest()
        
        print(f"\n{'='*80}")
        print(f"ARCHIVE INITIALIZED")
        print(f"{'='*80}")
        print(f"Run ID: {self.run_id}")
        print(f"UUID: {self.run_uuid}")
        print(f"Directory: {self.run_dir}")
        print(f"{'='*80}\n")
        
        return str(self.run_dir)

    def get_analysis_dir(self, analysis_name: str) -> str:
        """
        Get the directory path for a specific analysis within the run

        Parameters:
        -----------
        analysis_name : str
            Name of the analysis (e.g., 'main_keywords')

        Returns:
        --------
        str : Path to the analysis directory
        """
        if not self.run_dir:
            raise RuntimeError("Run not initialized. Call initialize_run() first.")
        
        analysis_dir = self.run_dir / analysis_name
        analysis_dir.mkdir(parents=True, exist_ok=True)
        return str(analysis_dir)

    def update_configuration(self, config: Dict[str, Any]):
        """
        Update the configuration section of the manifest

        Parameters:
        -----------
        config : dict
            Configuration dictionary to store
        """
        self.manifest["configuration"].update(config)
        self._save_manifest()

    def update_results_summary(self, analysis_name: str, summary: Dict[str, Any]):
        """
        Update the results summary for a specific analysis

        Parameters:
        -----------
        analysis_name : str
            Name of the analysis
        summary : dict
            Summary statistics for the analysis
        """
        self.manifest["results_summary"]["analyses"][analysis_name] = summary
        self._save_manifest()

    def set_total_posts(self, count: int):
        """
        Set the total number of posts collected

        Parameters:
        -----------
        count : int
            Total posts collected
        """
        self.manifest["results_summary"]["total_posts_collected"] = count
        self._save_manifest()

    def add_file(self, file_path: str):
        """
        Register a generated file in the manifest

        Parameters:
        -----------
        file_path : str
            Path to the generated file (relative to run directory)
        """
        if file_path not in self.manifest["files_generated"]:
            self.manifest["files_generated"].append(file_path)
            self._save_manifest()

    def add_error(self, error: str):
        """
        Register an error in the manifest

        Parameters:
        -----------
        error : str
            Error message
        """
        self.errors.append(error)
        self.manifest["errors"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": error
        })
        self._save_manifest()

    def add_warning(self, warning: str):
        """
        Register a warning in the manifest

        Parameters:
        -----------
        warning : str
            Warning message
        """
        self.warnings.append(warning)
        self.manifest["warnings"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": warning
        })
        self._save_manifest()

    def finalize_run(self):
        """
        Finalize the run by computing duration and updating the index
        """
        if not self.start_time:
            raise RuntimeError("Run not initialized")
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()
        
        self.manifest["timestamp_end"] = end_time.isoformat()
        self.manifest["duration_seconds"] = round(duration, 2)
        
        self._save_manifest()
        self._update_index()
        
        print(f"\n{'='*80}")
        print(f"ARCHIVE FINALIZED")
        print(f"{'='*80}")
        print(f"Run ID: {self.run_id}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Files generated: {len(self.manifest['files_generated'])}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"{'='*80}\n")

    def _save_manifest(self):
        """Save the manifest to the run directory"""
        if not self.run_dir:
            return
        
        manifest_path = self.run_dir / "manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2)

    def _update_index(self):
        """Update the global index of all runs"""
        index_path = self.base_archive_dir / "index.json"
        
        # Load existing index or create new one
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {"runs": []}
        
        # Create index entry for this run
        run_entry = {
            "run_id": self.run_id,
            "uuid": self.run_uuid,
            "timestamp": self.manifest["timestamp_start"],
            "duration_seconds": self.manifest["duration_seconds"],
            "total_posts": self.manifest["results_summary"]["total_posts_collected"],
            "analyses": list(self.manifest["results_summary"]["analyses"].keys()),
            "files_count": len(self.manifest["files_generated"]),
            "has_errors": len(self.errors) > 0,
            "has_warnings": len(self.warnings) > 0
        }
        
        # Add to index (newest first)
        index["runs"].insert(0, run_entry)
        
        # Save updated index
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)


# Utility functions for archive management

def list_runs(base_archive_dir: str = "exports/runs") -> List[Dict[str, Any]]:
    """
    List all archived runs

    Parameters:
    -----------
    base_archive_dir : str
        Base directory for archived runs

    Returns:
    --------
    list : List of run metadata dictionaries
    """
    index_path = Path(base_archive_dir) / "index.json"
    
    if not index_path.exists():
        return []
    
    with open(index_path, 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    return index.get("runs", [])


def get_run_manifest(run_id: str, base_archive_dir: str = "exports/runs") -> Optional[Dict[str, Any]]:
    """
    Get the manifest for a specific run

    Parameters:
    -----------
    run_id : str
        Run ID or UUID
    base_archive_dir : str
        Base directory for archived runs

    Returns:
    --------
    dict : Manifest dictionary or None if not found
    """
    base_path = Path(base_archive_dir)
    
    # Try as run_id first
    manifest_path = base_path / run_id / "manifest.json"
    
    if not manifest_path.exists():
        # Try to find by UUID
        for run_dir in base_path.iterdir():
            if run_dir.is_dir():
                manifest_path = run_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                        if manifest.get("uuid") == run_id or manifest.get("run_id") == run_id:
                            return manifest
        return None
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_latest_run(base_archive_dir: str = "exports/runs") -> Optional[Dict[str, Any]]:
    """
    Get the most recent run

    Parameters:
    -----------
    base_archive_dir : str
        Base directory for archived runs

    Returns:
    --------
    dict : Latest run metadata or None if no runs exist
    """
    runs = list_runs(base_archive_dir)
    return runs[0] if runs else None


def print_runs_table(limit: int = 10, base_archive_dir: str = "exports/runs"):
    """
    Print a formatted table of runs

    Parameters:
    -----------
    limit : int
        Maximum number of runs to display
    base_archive_dir : str
        Base directory for archived runs
    """
    runs = list_runs(base_archive_dir)
    
    if not runs:
        print("No archived runs found.")
        return
    
    print(f"\n{'='*120}")
    print(f"ARCHIVED RUNS (showing {min(limit, len(runs))} of {len(runs)})")
    print(f"{'='*120}")
    print(f"{'Run ID':<30} {'Date':<20} {'Posts':<8} {'Analyses':<12} {'Duration':<12} {'Status':<10}")
    print(f"{'-'*120}")
    
    for run in runs[:limit]:
        run_id = run["run_id"][:28] + ".." if len(run["run_id"]) > 30 else run["run_id"]
        timestamp = datetime.fromisoformat(run["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        posts = run["total_posts"]
        analyses = len(run["analyses"])
        duration = f"{run['duration_seconds']:.1f}s"
        
        status = "✓"
        if run["has_errors"]:
            status = "✗ Error"
        elif run["has_warnings"]:
            status = "⚠ Warning"
        
        print(f"{run_id:<30} {timestamp:<20} {posts:<8} {analyses:<12} {duration:<12} {status:<10}")
    
    print(f"{'='*120}\n")


def cleanup_old_runs(keep_last_n: int = 10, base_archive_dir: str = "exports/runs"):
    """
    Remove old runs, keeping only the most recent N

    Parameters:
    -----------
    keep_last_n : int
        Number of recent runs to keep
    base_archive_dir : str
        Base directory for archived runs
    """
    import shutil
    
    runs = list_runs(base_archive_dir)
    
    if len(runs) <= keep_last_n:
        print(f"No cleanup needed. {len(runs)} runs exist, keeping last {keep_last_n}.")
        return
    
    runs_to_delete = runs[keep_last_n:]
    base_path = Path(base_archive_dir)
    
    print(f"\nCleaning up {len(runs_to_delete)} old runs...")
    
    for run in runs_to_delete:
        run_dir = base_path / run["run_id"]
        if run_dir.exists():
            shutil.rmtree(run_dir)
            print(f"  Deleted: {run['run_id']}")
    
    # Update index
    index_path = base_path / "index.json"
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump({"runs": runs[:keep_last_n]}, f, indent=2)
    
    print(f"Cleanup complete. Kept {keep_last_n} most recent runs.\n")


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python archive.py list [limit]           - List archived runs")
        print("  python archive.py show <run_id|uuid>     - Show run details")
        print("  python archive.py latest                 - Show latest run")
        print("  python archive.py cleanup --keep N       - Keep only last N runs")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "list":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        print_runs_table(limit=limit)
    
    elif command == "show":
        if len(sys.argv) < 3:
            print("Error: Please provide a run ID or UUID")
            sys.exit(1)
        
        run_id = sys.argv[2]
        manifest = get_run_manifest(run_id)
        
        if not manifest:
            print(f"Error: Run '{run_id}' not found")
            sys.exit(1)
        
        print(json.dumps(manifest, indent=2))
    
    elif command == "latest":
        latest = get_latest_run()
        if latest:
            print(f"Latest run: {latest['run_id']}")
            print(f"Timestamp: {latest['timestamp']}")
            print(f"Posts: {latest['total_posts']}")
            print(f"Analyses: {', '.join(latest['analyses'])}")
        else:
            print("No runs found")
    
    elif command == "cleanup":
        keep_n = 10
        if len(sys.argv) > 2 and sys.argv[2] == "--keep":
            keep_n = int(sys.argv[3])
        cleanup_old_runs(keep_last_n=keep_n)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
