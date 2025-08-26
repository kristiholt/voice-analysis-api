# replit.md

## Overview

X Voice API is an advanced voice analysis platform built by Voxcentia using FastAPI. The system analyzes audio files (WAV, FLAC, MP3, M4A, OGG, AAC) and returns emotion scores (emo1-emo26) and personality trait scores (char1-char94). It features deterministic MFCC feature extraction, enhanced psychological model architecture with Big Five personality mapping, content-hash caching to avoid recomputation, z-score normalization with rolling 30-day baselines, and comprehensive request logging. The API provides a standardized interface for seamless voice analysis integration.

**Recent Completion**: Clean API Response Template (2025-08-26) - Implemented customer-requested clean response format with pillars as separate top-level category, eliminating measurement duplication, maintaining consistent 0-100 scaling across all values.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Core API Framework
The application is built on FastAPI with uvicorn as the ASGI server. The main application structure follows a modular design with separate modules for audio processing, feature extraction, machine learning models, post-processing, and data storage. CORS middleware is enabled for cross-origin requests.

### Audio Processing Pipeline
Audio files are processed through a patent-validated pipeline that converts various formats to mono float32 format using librosa and soundfile. The system extracts MFCC features, converts audio waveforms to visual representations for pattern recognition, and applies quantum physics-based vibrational frequency analysis. Features undergo binary classification using 80th percentile population thresholds before conversion to standardized probability scores. The complete pipeline: Audio → Waveform → Image → Pattern Recognition → Binary Classification → Probability Scores → Wellbeing Status.

### Authentication & Security
Bearer token authentication is implemented using SHA256 hashed API keys stored in the database. The system parses Authorization headers and validates tokens against the Supabase database. API keys are stored securely with active/inactive status tracking.

### Caching Strategy
Content-hash caching is implemented using SHA256 hashes of audio file content. This prevents recomputation of identical audio files, significantly improving performance for repeated requests. Cache lookups occur before expensive audio processing operations.

### Machine Learning Architecture
The system features patent-validated architecture with binary classification methodology and Enneagram personality integration. Using the original research approach, emotions are classified using 80th percentile thresholds and converted to probability scores (0-1 scale) for standardized comparison. The system predicts 38 emotion scores (emo1-emo38), 94 personality traits (char1-char94), and 9 Enneagram personality archetypes. Advanced audio-to-image conversion enables pattern recognition analysis combined with quantum physics and vibrational frequency principles.

### Normalization & Post-Processing
Z-score normalization is applied using rolling 30-day baselines that are updated nightly. The system maintains historical statistics for both emotions and traits, allowing for consistent normalization across time periods. Normalization metadata is included in API responses.

### Data Persistence & Learning Loop
All API requests, results, and metadata are logged to the database for audit trails and analytics. The system stores audio metadata, processing times, feature vectors, prediction results, and user feedback for adaptive learning. Advanced features include journaling analysis through NLP emotional topic extraction, action response surveys for supervised learning, and continuous algorithm updates based on user outcomes. The system maintains Struggling/OK/Thriving classification thresholds and Enneagram personality archetype mappings.

### Background Jobs
A nightly cron job computes rolling baseline statistics for normalization. This job runs independently of the main API service and updates the normalization baselines used for z-score calculations.

### Error Handling & Validation
Comprehensive error handling covers audio format validation, file size limits, processing timeouts, and database connection issues. The system returns detailed error responses with appropriate HTTP status codes while maintaining API compatibility.

## External Dependencies

### Database & Storage
- **Supabase**: Primary database for API key management, request logging, results storage, and normalization baselines. Provides both PostgreSQL database and authentication services.

### Audio Processing Libraries
- **librosa**: Core audio analysis library for loading audio files and extracting MFCC features
- **soundfile**: Audio file I/O operations for various formats (WAV, FLAC, MP3, etc.)
- **numpy**: Numerical computing for audio signal processing and feature manipulation

### Web Framework & HTTP
- **FastAPI**: Modern Python web framework providing automatic API documentation and request validation
- **uvicorn**: ASGI server for serving the FastAPI application
- **httpx**: HTTP client library for testing and external API calls
- **pydantic**: Data validation and serialization for API request/response models

### Deployment & Infrastructure
- **Railway**: Cloud platform for production deployment with scale-to-zero capabilities
- **Replit**: Development environment for coding, testing, and Git integration

### Environment & Configuration
The application relies on environment variables for configuration including Supabase credentials, normalization parameters, audio processing limits, and feature extraction versions. All sensitive configuration is externalized from the codebase.