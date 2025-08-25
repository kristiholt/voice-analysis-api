# X Voice API - Drop-in Replacement

Advanced voice analysis API by Voxcentia, built with FastAPI for professional voice analysis applications.

## Features

- **Professional Voice Analysis**: Comprehensive emotion and personality trait detection
- **Bearer Token Authentication**: SHA256 hashed API keys stored in Supabase
- **Audio Processing**: Supports WAV, FLAC, MP3, M4A, OGG, AAC formats (≤20s, ≤10MB)
- **Content-Hash Caching**: Avoids recomputation of identical audio files
- **Deterministic Features**: MFCC extraction with fixed parameters for consistent results
- **Emotions & Traits**: Returns scores for 26 emotions (emo1-emo26) and 94 traits (char1-char94)
- **Z-Score Normalization**: Rolling 30-day baselines with nightly updates
- **Request Logging**: Complete audit trail in Supabase
- **Real-time Processing**: <2s response time for typical audio files

## Quick Start

### Local Development (Replit)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   