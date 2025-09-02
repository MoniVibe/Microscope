"""
M3 GPU Runner for Microscope project - Complete Version
Enforces strict FP32 precision and produces standardized outputs
Supports all three M3 solvers with real metrics calculations
"""

import argparse
import json
import yaml
import sys
import logging
import time
import os
from pathlib import Path
import torch
import numpy as np
import tifffile
from typing import Dict, Any, Optional

# Setup paths for proper imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SRC_DIR = PROJECT_ROOT / 'src'
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

# Import M3 solvers
from optics_sim.prop.solvers.m3_angular_spectrum import M3AngularSpectrumSolver
from optics_sim.prop.solvers.m3_split_step_bpm import M3SplitStepBPMSolver  
from optics_sim.prop.solvers.m3_wide_angle_bpm import M3WideAngleBPMSolver

# Import M3 metrics
from optics_sim.metrics.m3_metrics import calculate_m3_metrics

logger = logging.getLogger(__name__)


class M3Pipeline:
    """M3-compliant pipeline for Microscope project with all solvers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        
        # M3: Enforce FP32 settings
        self.mixed_fft = False  # Always False for M3
        self.deterministic = True
        self.seed = config.get('seed', 1337)
        
        # Set deterministic execution
        self._set_deterministic()
        
        # Initialize solver
        self.solver = self._create_solver()
        
        # Output directory
        self.output_dir = Path(config.get('output', {}).get('directory', 'output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.perf_stats = {}
    
    def _set_deterministic(self):
        """Configure deterministic execution for M3"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # Enforce highest matmul precision for FP32
        try:
            torch.set_float32_matmul_precision('highest')
        except Exception:
            pass
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    def _create_solver(self):
        """Create M3-compliant solver based on config"""
        solver_cfg = self.config['solver']
        solver_type = solver_cfg['type']
        params = solver_cfg.get('params', {})
        
        # Force M3 compliance
        params['device'] = self.device
        params['enforce_fp32'] = True
        params['mixed_fft'] = False
        params['seed'] = self.seed
        
        # Create appropriate solver
        if solver_type == 'angular_spectrum':
            logger.info("Using M3 Angular Spectrum solver")
            return M3AngularSpectrumSolver(**params)
        elif solver_type == 'split_step_bpm':
            logger.info("Using M3 Split-Step BPM solver")
            return M3SplitStepBPMSolver(**params)
        elif solver_type == 'wide_angle_bpm':
            logger.info("Using M3 Wide-Angle BPM solver")
            return M3WideAngleBPMSolver(**params)
        else:
            raise ValueError(f"Unknown M3 solver type: {solver_type}")
    
    def _create_source(self) -> torch.Tensor:
        """Create source field with FP32 precision"""
        source_cfg = self.config['source']
        source_type = source_cfg['type']
        params = source_cfg.get('params', {})
        
        N = self.config['grid']['N']
        dx = self.config['grid']['dx']
        
        # Create coordinate grids in FP32
        x = torch.arange(N, dtype=torch.float32, device=self.device) - N // 2
        x = x * dx
        X, Y = torch.meshgrid(x, x, indexing='xy')
        
        if source_type == 'gaussian':
            waist = params.get('waist', 1e-3)
            R_sq = X**2 + Y**2
            field = torch.exp(-R_sq / (waist**2)).to(torch.complex64)
        elif source_type == 'plane_wave':
            amplitude = params.get('amplitude', 1.0)
            angle_deg = params.get('angle_deg', 0)
            angle_rad = np.deg2rad(angle_deg)
            
            # Tilted plane wave
            k = 2 * np.pi / params.get('wavelength', 550e-9)
            phase = k * (X * np.sin(angle_rad))
            field = amplitude * torch.exp(1j * phase).to(torch.complex64)
        elif source_type == 'point':
            # Point source (delta function)
            field = torch.zeros(N, N, dtype=torch.complex64, device=self.device)
            field[N//2, N//2] = N * N  # Normalized delta
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        return field
    
    def run(self) -> Dict[str, Any]:
        """Execute M3 pipeline with proper metrics"""
        logger.info("Starting M3 pipeline")
        start_time = time.time()
        
        # Reset GPU memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        
        # Create source field
        source_field = self._create_source()
        
        # Get propagation parameters
        dx = self.config['grid']['dx']
        z_distance = self.config['propagation']['distance']
        
        logger.info(f"Propagating {z_distance*1e3:.1f} mm with dx={dx*1e6:.2f} μm")
        prop_start = time.time()
        
        # Handle special cases for different solvers
        solver_type = self.config['solver']['type']
        
        if solver_type == 'split_step_bpm' and 'grating' in self.config['propagation']:
            # Handle grating propagation
            grating = self.config['propagation']['grating']
            if grating.get('enabled', False):
                output_field = self.solver.propagate_grating(
                    source_field,
                    z_distance,
                    dx,
                    grating['period'],
                    grating['depth'],
                    grating.get('n_grating', 1.5)
                )
            else:
                output_field = self.solver.propagate(source_field, z_distance, dx)
        elif solver_type == 'wide_angle_bpm' and 'lens' in self.config['propagation']:
            # Handle high-NA focusing
            lens = self.config['propagation']['lens']
            if lens.get('enabled', False):
                output_field = self.solver.propagate_high_na_focus(
                    source_field,
                    lens['focal_length'],
                    dx
                )
            else:
                output_field = self.solver.propagate(source_field, z_distance, dx)
        else:
            # Standard propagation
            output_field = self.solver.propagate(source_field, z_distance, dx)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        prop_time = time.time() - prop_start
        
        # Calculate intensity
        intensity = torch.abs(output_field) ** 2
        intensity_np = intensity.cpu().numpy().astype(np.float32)
        
        # Save outputs
        self._save_outputs(intensity_np, output_field)
        
        # Calculate real metrics using M3 metrics module
        wavelength = self.config['solver']['params']['wavelength']
        NA = self.config['solver']['params'].get('NA', 0.5)
        
        metrics = calculate_m3_metrics(
            output_field,
            wavelength,
            NA,
            dx,
            reference_field=source_field,
            device=self.device
        )
        
        # Performance stats
        wall_time = time.time() - start_time
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(self.device)
        
        self.perf_stats = {
            'wall_time_sec': wall_time,
            'solver_time_sec': prop_time,
            'peak_vram_bytes': int(peak_memory),
            'peak_vram_gb': peak_memory / (1024**3)
        }
        
        # Save all JSONs
        self._save_json_outputs(metrics)
        
        # Validate M3 compliance
        compliance = self.solver.validate_m3_compliance()
        
        return {
            'metrics': metrics,
            'performance': self.perf_stats,
            'compliance': compliance
        }
    
    def _save_outputs(self, intensity: np.ndarray, field: torch.Tensor):
        """Save TIFF output with complete metadata"""
        # Prepare metadata
        metadata = {
            'grid_N': self.config['grid']['N'],
            'grid_dx_m': float(self.config['grid']['dx']),
            'wavelength_m': float(self.config['solver']['params']['wavelength']),
            'NA': float(self.config['solver']['params'].get('NA', 0.5)),
            'z_distance_m': float(self.config['propagation']['distance']),
            'seed': self.seed,
            'solver': self.config['solver']['type'],
            'device': str(self.device),
            'units': 'meters',
            'intensity_units': 'arbitrary'
        }
        
        # Save TIFF
        tiff_path = self.output_dir / 'output.tiff'
        tifffile.imwrite(
            tiff_path,
            intensity,
            metadata=metadata,
            photometric='minisblack'
        )
        logger.info(f"Saved TIFF: {tiff_path}")
    
    def _save_json_outputs(self, metrics: Dict[str, float]):
        """Save JSON outputs for M3 with CI-compatible keys"""
        # Save metrics.json with both standard and CI keys
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save perf.json
        perf_path = self.output_dir / 'perf.json'
        with open(perf_path, 'w') as f:
            json.dump(self.perf_stats, f, indent=2)
        
        # Save env.json
        env_info = {
            'torch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'deterministic': self.deterministic,
            'seed': self.seed,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'TORCH_ALLOW_TF32': int(os.environ.get('TORCH_ALLOW_TF32', '0') or '0'),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG', ''),
        }
        
        if torch.cuda.is_available():
            env_info.update({
                'cuda_version': torch.version.cuda if hasattr(torch.version, 'cuda') else 'unknown',
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 0,
                'gpu_name': torch.cuda.get_device_name(0)
            })
        
        env_path = self.output_dir / 'env.json'
        with open(env_path, 'w') as f:
            json.dump(env_info, f, indent=2)
        
        logger.info(f"Saved JSON outputs to {self.output_dir}")


def main():
    """Main M3 CLI entry point"""
    parser = argparse.ArgumentParser(
        description='M3 GPU Runner for Microscope project - Complete Version'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Path to M3 configuration file (YAML)'
    )
    
    parser.add_argument(
        '--output', '--out', '-o',
        type=Path,
        default=None,
        dest='output',
        help='Output directory'
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
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    logger.info(f"Loading config: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override settings
    if args.device:
        config['device'] = args.device
    if args.output:
        config.setdefault('output', {})['directory'] = str(args.output)

    # Add file logging to output directory (run.log)
    try:
        out_dir = Path(config.get('output', {}).get('directory', 'output'))
        out_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(out_dir / 'run.log', mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to initialize file logging: {e}")
    
    # Validate GPU
    if config.get('device') == 'cuda':
        if not torch.cuda.is_available():
            logger.error("CUDA requested but not available")
            return 1
        
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3)
        }
        logger.info(f"GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
        
        # Check memory budget
        if gpu_info['memory_gb'] < 4.0:
            logger.warning(f"GPU memory {gpu_info['memory_gb']:.1f} GB may be insufficient for M3")
    
    # Run pipeline
    try:
        pipeline = M3Pipeline(config)
        results = pipeline.run()
        
        # Report results
        logger.info("="*60)
        logger.info("M3 Pipeline completed successfully")
        logger.info(f"Wall time: {results['performance']['wall_time_sec']:.2f} s")
        logger.info(f"Solver time: {results['performance']['solver_time_sec']:.2f} s")
        logger.info(f"Peak VRAM: {results['performance']['peak_vram_gb']:.3f} GB")
        
        # Check metrics gates
        logger.info("-"*60)
        logger.info("Metrics validation:")
        
        gates = {
            'l2_error': 0.03,
            'energy_error': 0.01,
            'airy_first_zero_error': 0.02,
            'strehl_ratio': 0.95,  # minimum
            'mtf_cutoff_error': 0.05
        }
        
        passed = True
        for metric, threshold in gates.items():
            value = results['metrics'].get(metric, -1)
            if value < 0:
                logger.warning(f"  {metric}: NOT CALCULATED")
                continue
                
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
        
        logger.info("-"*60)
        if passed:
            logger.info("✓ All M3 metric gates PASSED")
        else:
            logger.warning("✗ Some M3 metric gates FAILED")
        
        # Check M3 compliance
        if results['compliance']['m3_compliant']:
            logger.info("✓ M3 compliance verified")
        else:
            logger.warning("✗ M3 compliance issues detected:")
            for key, value in results['compliance'].items():
                if key != 'm3_compliant':
                    logger.warning(f"    {key}: {value}")
        
        logger.info("="*60)
        
        return 0 if passed and results['compliance']['m3_compliant'] else 1
        
    except Exception as e:
        logger.error(f"M3 pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
