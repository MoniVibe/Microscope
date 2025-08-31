"""TIFF file I/O for optical field data.

Writes 32-bit TIFF stacks with complex fields as real/imaginary planes
and comprehensive metadata embedding.
"""

from __future__ import annotations

import json
import struct
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

# Try to import tifffile for more robust TIFF writing
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def write_tiff(filename: Union[str, Path],
               data: Union[torch.Tensor, np.ndarray],
               metadata: Optional[Dict] = None,
               dx_um: float = 1.0,
               dy_um: float = 1.0,
               dz_um: float = 1.0,
               wavelengths_um: Optional[np.ndarray] = None,
               na: Optional[float] = None) -> None:
    """Write data to 32-bit TIFF file with metadata.
    
    Complex data is written as two planes (real, imaginary).
    Metadata is embedded in ImageDescription tag.
    
    Args:
        filename: Output filename
        data: Data tensor (real or complex), shape (ny, nx) or (nz, ny, nx)
        metadata: Additional metadata dictionary
        dx_um, dy_um, dz_um: Voxel dimensions in micrometers
        wavelengths_um: Array of wavelengths for spectral data
        na: Numerical aperture
    """
    filename = Path(filename)
    
    # Convert torch to numpy if needed
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    
    # Prepare comprehensive metadata
    meta_dict = _prepare_metadata(
        data, metadata, dx_um, dy_um, dz_um, wavelengths_um, na
    )
    
    # Handle complex data
    if np.iscomplexobj(data):
        data_stack = _prepare_complex_stack(data)
        meta_dict['is_complex'] = True
        meta_dict['data_type'] = 'complex64'
    else:
        data_stack = data.astype(np.float32)
        if data_stack.ndim == 2:
            data_stack = data_stack[np.newaxis, ...]
        meta_dict['is_complex'] = False
        meta_dict['data_type'] = 'float32'
    
    # Write using tifffile if available, otherwise use basic writer
    if HAS_TIFFFILE:
        _write_with_tifffile(filename, data_stack, meta_dict)
    else:
        _write_basic_tiff(filename, data_stack, meta_dict)


def _prepare_metadata(data: np.ndarray,
                     user_metadata: Optional[Dict],
                     dx_um: float, dy_um: float, dz_um: float,
                     wavelengths_um: Optional[np.ndarray],
                     na: Optional[float]) -> Dict:
    """Prepare comprehensive metadata dictionary.
    
    Args:
        data: Data array
        user_metadata: User-provided metadata
        dx_um, dy_um, dz_um: Voxel dimensions
        wavelengths_um: Wavelengths array
        na: Numerical aperture
        
    Returns:
        Complete metadata dictionary
    """
    import platform
    import torch
    
    # Basic metadata
    meta = {
        'units': 'micrometers',
        'dx_um': dx_um,
        'dy_um': dy_um,
        'dz_um': dz_um,
        'shape': list(data.shape),
        'dtype': str(data.dtype),
        'timestamp': datetime.now().isoformat(),
        'coordinate_frame': 'right-handed, Z-Y-X Euler',
    }
    
    # Add optical parameters
    if wavelengths_um is not None:
        meta['wavelengths_um'] = wavelengths_um.tolist() if hasattr(wavelengths_um, 'tolist') else list(wavelengths_um)
        meta['lambda_min_um'] = float(np.min(wavelengths_um))
        meta['lambda_max_um'] = float(np.max(wavelengths_um))
    
    if na is not None:
        meta['NA'] = float(na)
    
    # Add system info
    meta['system'] = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__ if torch else 'N/A',
        'cuda_available': torch.cuda.is_available() if torch else False,
    }
    
    # Add user metadata
    if user_metadata:
        # Seeds and config hash
        if 'seeds' in user_metadata:
            meta['seeds'] = user_metadata['seeds']
        if 'config_hash' in user_metadata:
            meta['config_hash'] = user_metadata['config_hash']
        
        # Merge remaining
        for key, value in user_metadata.items():
            if key not in meta:
                meta[key] = value
    
    return meta


def _prepare_complex_stack(data: np.ndarray) -> np.ndarray:
    """Prepare complex data as interleaved real/imaginary stack.
    
    Args:
        data: Complex data array
        
    Returns:
        Float32 stack with real/imag interleaved
    """
    data_real = np.real(data).astype(np.float32)
    data_imag = np.imag(data).astype(np.float32)
    
    if data.ndim == 2:
        # Single frame: stack real and imag
        return np.stack([data_real, data_imag], axis=0)
    elif data.ndim == 3:
        # Multiple frames: interleave real/imag
        n_frames = data.shape[0]
        stack = np.zeros((2 * n_frames, *data.shape[1:]), dtype=np.float32)
        stack[0::2] = data_real  # Even indices: real
        stack[1::2] = data_imag  # Odd indices: imaginary
        return stack
    else:
        raise ValueError(f"Unsupported data dimensions: {data.ndim}")


def _write_with_tifffile(filename: Path, data: np.ndarray, metadata: Dict) -> None:
    """Write TIFF using tifffile library.
    
    Args:
        filename: Output filename
        data: Data stack
        metadata: Metadata dictionary
    """
    # Convert metadata to JSON string for ImageDescription
    meta_json = json.dumps(metadata, indent=2)
    
    # Resolution in pixels per centimeter
    resolution = (10000.0 / metadata['dx_um'], 10000.0 / metadata['dy_um'])
    
    # Write TIFF
    tifffile.imwrite(
        filename,
        data,
        dtype=np.float32,
        resolution=resolution,
        resolutionunit='CENTIMETER',
        metadata={'axes': 'ZYX', 'unit': 'um'},
        description=meta_json,
    )


def _write_basic_tiff(filename: Path, data: np.ndarray, metadata: Dict) -> None:
    """Basic TIFF writer without external dependencies.
    
    Writes a simplified TIFF that can be read by most software.
    
    Args:
        filename: Output filename
        data: Data stack (frames, height, width)
        metadata: Metadata dictionary
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    n_frames, height, width = data.shape
    
    # Ensure data is float32 and C-contiguous
    data = np.ascontiguousarray(data, dtype=np.float32)
    
    # Convert metadata to JSON
    meta_json = json.dumps(metadata, indent=2)
    meta_bytes = meta_json.encode('ascii', errors='ignore')
    
    with open(filename, 'wb') as f:
        # Write simple single-frame TIFF for first frame
        # (Full multi-frame support would be more complex)
        
        # TIFF header
        f.write(b'II')  # Little-endian
        f.write(struct.pack('<H', 42))  # Magic number
        f.write(struct.pack('<I', 8))  # First IFD offset
        
        # IFD (Image File Directory)
        # Using minimal tags for compatibility
        n_tags = 10
        f.write(struct.pack('<H', n_tags))
        
        # Calculate data offset (after IFD and metadata)
        data_offset = 8 + 2 + n_tags * 12 + 4 + len(meta_bytes) + 1
        if data_offset % 2:
            data_offset += 1
        
        # ImageWidth (256)
        f.write(struct.pack('<HHII', 256, 4, 1, width))
        
        # ImageLength (257)
        f.write(struct.pack('<HHII', 257, 4, 1, height))
        
        # BitsPerSample (258)
        f.write(struct.pack('<HHIHH', 258, 3, 1, 32, 0))
        
        # Compression (259) - no compression
        f.write(struct.pack('<HHIHH', 259, 3, 1, 1, 0))
        
        # PhotometricInterpretation (262)
        f.write(struct.pack('<HHIHH', 262, 3, 1, 1, 0))
        
        # StripOffsets (273)
        f.write(struct.pack('<HHII', 273, 4, 1, data_offset))
        
        # SamplesPerPixel (277)
        f.write(struct.pack('<HHIHH', 277, 3, 1, 1, 0))
        
        # RowsPerStrip (278)
        f.write(struct.pack('<HHIHH', 278, 3, 1, height, 0))
        
        # StripByteCounts (279)
        f.write(struct.pack('<HHII', 279, 4, 1, width * height * 4))
        
        # SampleFormat (339) - floating point
        f.write(struct.pack('<HHIHH', 339, 3, 1, 3, 0))
        
        # Next IFD offset (0 = last)
        f.write(struct.pack('<I', 0))
        
        # Write metadata as comment
        f.write(meta_bytes)
        f.write(b'\x00')
        
        # Pad to data offset
        current = f.tell()
        if current < data_offset:
            f.write(b'\x00' * (data_offset - current))
        
        # Write first frame data
        f.write(data[0].tobytes())
        
        # Note: This simplified writer only saves the first frame
        # Full multi-frame support would require more complex IFD chaining
        if n_frames > 1:
            print(f"Warning: Basic TIFF writer only saved first frame of {n_frames}")


def read_tiff(filename: Union[str, Path]) -> tuple[np.ndarray, Dict]:
    """Read TIFF file with metadata.
    
    Args:
        filename: Input filename
        
    Returns:
        Tuple of (data, metadata)
    """
    filename = Path(filename)
    
    if HAS_TIFFFILE:
        return _read_with_tifffile(filename)
    else:
        return _read_basic_tiff(filename)


def _read_with_tifffile(filename: Path) -> tuple[np.ndarray, Dict]:
    """Read TIFF using tifffile library."""
    with tifffile.TiffFile(filename) as tif:
        data = tif.asarray()
        
        # Extract metadata from ImageDescription
        metadata = {}
        if tif.pages[0].description:
            try:
                metadata = json.loads(tif.pages[0].description)
            except json.JSONDecodeError:
                metadata = {'description': tif.pages[0].description}
        
        # Reconstruct complex data if needed
        if metadata.get('is_complex', False):
            if data.ndim == 3 and data.shape[0] % 2 == 0:
                # Interleaved real/imaginary
                data_real = data[0::2]
                data_imag = data[1::2]
                data = data_real + 1j * data_imag
        
        return data, metadata


def _read_basic_tiff(filename: Path) -> tuple[np.ndarray, Dict]:
    """Basic TIFF reader (simplified, single frame only)."""
    with open(filename, 'rb') as f:
        # Read header
        byte_order = f.read(2)
        if byte_order == b'II':
            endian = '<'  # Little-endian
        elif byte_order == b'MM':
            endian = '>'  # Big-endian
        else:
            raise ValueError("Invalid TIFF file")
        
        magic = struct.unpack(endian + 'H', f.read(2))[0]
        if magic != 42:
            raise ValueError("Invalid TIFF magic number")
        
        ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
        
        # Seek to IFD
        f.seek(ifd_offset)
        n_tags = struct.unpack(endian + 'H', f.read(2))[0]
        
        # Parse tags (simplified)
        width = height = 0
        data_offset = 0
        bits_per_sample = 8
        
        for _ in range(n_tags):
            tag, dtype, count, value_offset = struct.unpack(endian + 'HHII', f.read(12))
            
            if tag == 256:  # ImageWidth
                width = value_offset
            elif tag == 257:  # ImageLength
                height = value_offset
            elif tag == 258:  # BitsPerSample
                bits_per_sample = value_offset & 0xFFFF
            elif tag == 273:  # StripOffsets
                data_offset = value_offset
        
        # Read data (assuming float32)
        if data_offset and width and height:
            f.seek(data_offset)
            
            if bits_per_sample == 32:
                dtype = np.float32
                bytes_per_pixel = 4
            else:
                dtype = np.uint8 if bits_per_sample == 8 else np.uint16
                bytes_per_pixel = bits_per_sample // 8
            
            data_bytes = f.read(width * height * bytes_per_pixel)
            data = np.frombuffer(data_bytes, dtype=dtype).reshape((height, width))
        else:
            raise ValueError("Could not parse TIFF structure")
    
    return data, {}


def write_field_stack(filename: Union[str, Path],
                     fields: Dict[str, torch.Tensor],
                     config: Dict) -> None:
    """Write multiple fields to a TIFF stack with full metadata.
    
    Args:
        filename: Output filename
        fields: Dictionary of named fields
        config: Configuration dictionary with metadata
    """
    # Stack all fields
    all_data = []
    field_info = []
    
    for name, field in fields.items():
        if field.dim() == 2:
            field = field.unsqueeze(0)
        
        all_data.append(field)
        field_info.append({
            'name': name,
            'shape': list(field.shape),
            'is_complex': field.is_complex()
        })
    
    # Concatenate along first axis
    combined = torch.cat(all_data, dim=0)
    
    # Extract metadata from config
    metadata = {
        'fields': field_info,
        'config': config,
    }
    
    # Write to TIFF
    write_tiff(
        filename,
        combined,
        metadata=metadata,
        dx_um=config.get('dx_um', 1.0),
        dy_um=config.get('dy_um', 1.0),
        dz_um=config.get('dz_um', 1.0),
        wavelengths_um=config.get('wavelengths_um'),
        na=config.get('NA_max')
    )
