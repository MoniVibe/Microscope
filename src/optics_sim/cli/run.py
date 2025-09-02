"""
CLI runner for Microscope optical simulation pipeline
Enforces FP32 precision and produces standardized outputs
"""

import argparse
import json
import yaml
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
import subprocess
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.optics_sim.core.pipeline import Pipeline
from src.optics_sim.prop import (
    AngularSpectrumSolver,
    SplitStepBPMSolver, 
    WideAngleBPMSolver
)


def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def get_git_info():
    """Get current git commit info"""
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()[:8]
        
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        
        return {'commit': commit, 'branch': branch}
    except:
        return {'commit': 'unknown', 'branch': 'unknown'}


def validate_gpu(device: str):
    """Validate GPU availability and properties"""
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        
        # Get GPU info
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'capability': torch.cuda.get_device_capability(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
        
        # Check memory
        if gpu_info['memory_gb'] < 4.0:
            logging.warning(f"GPU memory {gpu_info['memory_gb']:.1f} GB may be insufficient")
        
        return gpu_info
    
    return None


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML or JSON"""
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def augment_config(config: dict, args) -> dict:
    """Augment config with CLI arguments"""
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Ensure FP32 precision
    config['mixed_fft'] = False  # Enforce strict FP32
    config['enforce_fp32'] = True
    
    # Set deterministic mode
    config['deterministic'] = True
    if 'seed' not in config:
        config['seed'] = 1337
    
    # Set output directory
    if args.output:
        config.setdefault('output', {})['directory'] = args.output
    
    return config


def simulate(config: dict) -> dict:
    """
    Run simulation with given configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Results dictionary with metrics and performance stats
    """
    # Create pipeline
    pipeline = Pipeline(config)
    
    # Run simulation
    results = pipeline.run()
    
    return results


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Run Microscope optical simulation on GPU with FP32 precision'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to configuration file (YAML or JSON)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Output directory (default: from config or ./output)'
    )
    
    parser.add_argument(
        '--device', '-d',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Compute device (default: cuda)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate configuration without running'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Augment with CLI args
    config = augment_config(config, args)
    
    # Validate GPU
    logger.info(f"Validating {args.device} device")
    gpu_info = validate_gpu(args.device)
    
    if gpu_info:
        logger.info(f"GPU: {gpu_info['name']}")
        logger.info(f"Memory: {gpu_info['memory_gb']:.1f} GB")
    
    # Set output directory
    output_dir = Path(config.get('output', {}).get('directory', 'output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    if args.validate_only:
        logger.info("Configuration validated successfully")
        return 0
    
    # Run simulation
    logger.info("Starting simulation")
    start_time = time.time()
    
    try:
        # Run pipeline
        results = simulate(config)
        
        # Log results
        wall_time = time.time() - start_time
        logger.info(f"Simulation completed in {wall_time:.2f} seconds")
        
        if 'metrics' in results:
            metrics = results['metrics']
            logger.info("Metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.6f}")
        
        if 'performance' in results:
            perf = results['performance']
            logger.info("Performance:")
            logger.info(f"  Wall time: {perf['wall_time_s']:.2f} s")
            logger.info(f"  Peak VRAM: {perf['peak_vram_gb']:.3f} GB")
            
            # Convert to expected format for output
            perf_output = {
                'peak_vram_bytes': int(perf['peak_vram_gb'] * 1024**3),
                'wall_time_sec': perf['wall_time_s']
            }
            
            # Save performance metrics
            with open(output_dir / 'perf.json', 'w') as f:
                json.dump(perf_output, f, indent=2)
        
        # Validate metrics gates
        gates_passed = validate_metrics_gates(results.get('metrics', {}))
        
        if gates_passed:
            logger.info("All metric gates PASSED")
        else:
            logger.warning("Some metric gates FAILED")
        
        return 0
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def validate_metrics_gates(metrics: dict) -> bool:
    """
    Validate metrics against acceptance gates
    
    Gates:
    - L2 error ≤ 3%
    - Energy error ≤ 1%
    - First zero error ≤ 2%
    - Strehl ratio ≥ 0.95
    - MTF cutoff error ≤ 5%
    """
    gates = {
        'l2_error': 0.03,
        'energy_error': 0.01,
        'airy_first_zero_error': 0.02,
        'strehl_ratio': 0.95,  # minimum
        'mtf_cutoff_error': 0.05
    }
    
    passed = True
    logger = logging.getLogger(__name__)
    
    for metric, threshold in gates.items():
        if metric in metrics:
            value = metrics[metric]
            if metric == 'strehl_ratio':
                # Strehl should be above threshold
                if value < threshold:
                    logger.warning(f"  {metric}: {value:.4f} < {threshold} (FAILED)")
                    passed = False
                else:
                    logger.info(f"  {metric}: {value:.4f} ≥ {threshold} (PASSED)")
            else:
                # Errors should be below threshold
                if value > threshold:
                    logger.warning(f"  {metric}: {value:.4f} > {threshold} (FAILED)")
                    passed = False
                else:
                    logger.info(f"  {metric}: {value:.4f} ≤ {threshold} (PASSED)")
    
    return passed


if __name__ == '__main__':
    sys.exit(main())
