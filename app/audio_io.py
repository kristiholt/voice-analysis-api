"""
Audio input/output operations.
Handles decoding of WAV/FLAC/MP3 files to mono float32 format.
"""

import io
import tempfile
import os
from typing import Tuple, Optional, Dict, Any
import soundfile as sf
import librosa
import numpy as np
import logging

from .utils import get_env_var

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio file processing and format conversion"""
    
    def __init__(self):
        self.max_duration = int(get_env_var("MAX_AUDIO_DURATION_SECONDS", "60"))
        self.decode_timeout = int(get_env_var("DECODE_TIMEOUT_SECONDS", "5"))
    
    def decode_audio_file(self, audio_bytes: bytes, filename: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Decode audio file to mono float32 format.
        
        Args:
            audio_bytes: Raw audio file bytes
            filename: Original filename for format detection
            
        Returns:
            Tuple of (audio_data, sample_rate, metadata)
            
        Raises:
            ValueError: If audio format is unsupported or file is invalid
            RuntimeError: If audio is too long or processing fails
        """
        try:
            # Create temporary file for librosa/soundfile processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(filename)) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            try:
                # Try to read with soundfile first (faster for supported formats)
                audio_data, sample_rate = self._read_with_soundfile(temp_file_path)
                detected_format = self._detect_format_soundfile(temp_file_path)
            except Exception:
                # Fallback to librosa for more format support
                audio_data, sample_rate = self._read_with_librosa(temp_file_path)
                detected_format = self._detect_format_from_extension(filename)
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data.T)
            
            # Validate audio duration
            duration_seconds = len(audio_data) / sample_rate
            if duration_seconds > self.max_duration:
                raise RuntimeError(f"Audio duration {duration_seconds:.2f}s exceeds maximum {self.max_duration}s")
            
            # Normalize to float32 range [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure audio is in valid range
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            metadata = {
                'format': detected_format,
                'sample_rate': sample_rate,
                'duration_seconds': duration_seconds,
                'duration_ms': int(duration_seconds * 1000),
                'channels': 1,  # Always mono after processing
                'samples': len(audio_data)
            }
            
            logger.info(f"Decoded audio: {metadata}")
            return audio_data, sample_rate, metadata
            
        except Exception as e:
            logger.error(f"Error decoding audio file {filename}: {e}")
            raise ValueError(f"Failed to decode audio file: {str(e)}")
    
    def _read_with_soundfile(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Read audio with soundfile library"""
        try:
            audio_data, sample_rate = sf.read(file_path, dtype='float32')
            return audio_data, sample_rate
        except Exception as e:
            raise ValueError(f"Soundfile failed to read audio: {e}")
    
    def _read_with_librosa(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Read audio with librosa library (more format support)"""
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=None, mono=False, dtype=np.float32)
            sample_rate = int(sample_rate)
            return audio_data, sample_rate
        except Exception as e:
            raise ValueError(f"Librosa failed to read audio: {e}")
    
    def _detect_format_soundfile(self, file_path: str) -> str:
        """Detect audio format using soundfile"""
        try:
            info = sf.info(file_path)
            return info.format.lower()
        except Exception:
            return self._detect_format_from_extension(file_path)
    
    def _detect_format_from_extension(self, filename: str) -> str:
        """Detect format from file extension"""
        extension = os.path.splitext(filename.lower())[1]
        format_map = {
            '.wav': 'wav',
            '.flac': 'flac', 
            '.mp3': 'mp3',
            '.m4a': 'm4a',
            '.ogg': 'ogg',
            '.aac': 'aac'
        }
        return format_map.get(extension, 'unknown')
    
    def _get_file_extension(self, filename: str) -> str:
        """Get file extension or default to .wav"""
        extension = os.path.splitext(filename.lower())[1]
        return extension if extension else '.wav'
    
    def validate_audio_format(self, filename: str) -> bool:
        """Validate if audio format is supported"""
        supported_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.aac'}
        extension = os.path.splitext(filename.lower())[1]
        return extension in supported_extensions


# Global audio processor instance
audio_processor = AudioProcessor()
