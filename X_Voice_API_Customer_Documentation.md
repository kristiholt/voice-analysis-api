# X Voice API - Customer Documentation

**Advanced voice analysis API by Voxcentia**  
*Drop-in replacement for Vibeonix audio pipeline*

---

## Overview

The X Voice API provides professional voice analysis capabilities, analyzing audio files to predict emotions and personality traits using advanced machine learning models. Built with FastAPI for high performance and reliability.

## Key Features

- **26 Emotion Scores** (emo1-emo26) with probabilistic predictions
- **94 Personality Traits** (char1-char94) based on Big Five psychology model
- **Multiple Audio Formats**: WAV, FLAC, MP3, M4A, OGG, AAC
- **Real-time Processing**: <2 second response time for typical audio files
- **Secure Authentication**: Bearer token with SHA256 hashed API keys
- **Content Caching**: Automatic deduplication for identical audio files
- **Smart Normalization**: Z-score normalization with rolling 30-day baselines

---

## Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

Contact your account manager to obtain your API key.

---

## Audio Analysis Endpoint

### POST `/v1/voice/analyze`

Analyzes voice audio for emotions and personality traits.

#### Request Format

- **Method**: POST
- **Content-Type**: multipart/form-data
- **File Parameter**: `audio`

#### Audio Requirements

- **Formats**: WAV, FLAC, MP3, M4A, OGG, AAC
- **Duration**: Maximum 20 seconds
- **File Size**: Maximum 10MB
- **Content**: Should contain human speech for optimal results

#### Headers

| Header | Required | Description |
|--------|----------|-------------|
| `Authorization` | Yes | Bearer token authentication |
| `Idempotency-Key` | No | Optional key for request deduplication |
| `X-No-Store` | No | Set to "true" to bypass caching |

#### Example Request

```bash
curl -X POST "https://your-api-domain.com/v1/voice/analyze" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@path/to/your/audio.wav"
```

#### Response Format

```json
{
  "request_id": "req_1234567890abcdef",
  "version": "1.0.0",
  "processing_ms": 1250,
  "audio_ms": 5000,
  "scores": {
    "emotions": {
      "emo1": 0.75,
      "emo2": 0.23,
      "emo3": 0.45,
      // ... emo1 through emo26
    },
    "traits": {
      "char1": 0.68,
      "char2": 0.34,
      "char3": 0.91,
      // ... char1 through char94
    }
  },
  "normalization": {
    "scheme": "zscore_rolling30d",
    "window_days": 30,
    "baseline_date": "2025-08-24"
  },
  "meta": {
    "audio_format": "wav",
    "sample_rate": 44100,
    "channels": 1,
    "features_version": "v1.0",
    "model_version": "enhanced_v1.0"
  },
  "warnings": []
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string | Unique identifier for this request |
| `version` | string | API version used |
| `processing_ms` | integer | Time taken to process the audio (milliseconds) |
| `audio_ms` | integer | Duration of the audio file (milliseconds) |
| `scores.emotions` | object | Emotion scores (emo1-emo26), values 0.0-1.0 |
| `scores.traits` | object | Personality trait scores (char1-char94), values 0.0-1.0 |
| `normalization` | object | Information about score normalization applied |
| `meta` | object | Technical metadata about the processing |
| `warnings` | array | Any warnings encountered during processing |

---

## Emotion Codes (emo1-emo26)

Currently mapped emotions include:

| Code | Emotion | Description |
|------|---------|-------------|
| emo1 | Happiness | Joy, contentment, positive affect |
| emo4 | Anger | Frustration, irritation, hostility |
| emo5 | Calm | Tranquility, peace, relaxation |
| emo6 | Love | Affection, warmth, care |
| emo7 | Positive Stress | Excitement, anticipation, energized tension |
| emo9 | Sadness | Sorrow, melancholy, low mood |
| emo11 | Loneliness | Isolation, disconnection, solitude |

*Complete emotion mappings available upon request.*

---

## Personality Traits (char1-char94)

Based on the Big Five personality model with psychological grounding:

### Neuroticism-Related Traits
- **char1**: Loss of Self-worth
- **char11**: Worry-prone
- **char15**: Doubting
- **char24**: Restless
- **char25**: Anger-prone
- **char31**: Inhibited by Fear
- **char47**: Lonely

### Extraversion-Related Traits
- **char7**: Pleasant Authority
- **char10**: Leadership
- **char19**: Assertive
- **char58**: Natural Leadership
- **char61**: Influencer
- **char71**: Convincing
- **char91**: Talkative

### Openness-Related Traits
- **char9**: Groundbreaking
- **char13**: Creative Mind
- **char43**: Theoretically Trained
- **char45**: Developer
- **char70**: Breaking from Past
- **char74**: Innovative Teamworker

### Conscientiousness-Related Traits
- **char26**: Silent Hard Worker
- **char27**: Rigid Duty
- **char37**: Skilled
- **char46**: Performance Leader
- **char77**: Urge to Perform

### Agreeableness-Related Traits
- **char39**: Compassion
- **char60**: Caring Conservative
- **char66**: Diplomatic
- **char72**: Supportive

*Complete trait mappings and descriptions available upon request.*

---

## Error Responses

### 400 Bad Request
```json
{
  "detail": "Unsupported audio format. Supported formats: WAV, FLAC, MP3, M4A, OGG, AAC",
  "error_code": "INVALID_FORMAT",
  "request_id": "req_1234567890abcdef"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid or missing API key",
  "error_code": "AUTH_FAILED"
}
```

### 413 Payload Too Large
```json
{
  "detail": "File size exceeds maximum limit of 10MB",
  "error_code": "FILE_TOO_LARGE"
}
```

### 429 Too Many Requests
```json
{
  "detail": "Rate limit exceeded. Maximum 60 requests per minute.",
  "error_code": "RATE_LIMITED"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error",
  "error_code": "PROCESSING_FAILED",
  "request_id": "req_1234567890abcdef"
}
```

---

## Health Check Endpoint

### GET `/health`

Returns API health status:

```json
{
  "status": "healthy",
  "timestamp": 1693843200,
  "version": "1.0.0"
}
```

---

## Rate Limits

- **Default**: 60 requests per minute per IP address
- **Custom limits**: Available based on your subscription plan
- **Rate limit headers**: Included in all responses

---

## Migration from Vibeonix

### Key Differences

1. **Direct API calls** instead of S3 bucket uploads
2. **Real-time responses** instead of batch processing
3. **26 emotions** instead of 38 (see gap analysis below)
4. **No CSV metadata required** - just upload audio directly
5. **Bearer token authentication** instead of x-api-key header

### Migration Steps

1. **Update authentication**: Replace `x-api-key` header with `Authorization: Bearer TOKEN`
2. **Change endpoint**: Use `/v1/voice/analyze` instead of `/audio-pipeline`
3. **Update request format**: Send audio as multipart/form-data instead of S3 upload
4. **Handle real-time responses**: Process JSON response immediately instead of polling S3

---

## Support

For technical support, API key requests, or custom integration assistance:

- **Documentation**: [API Documentation Portal]
- **Support Email**: support@voxcentia.com
- **Status Page**: [Service Status]

---

## Changelog

### Version 1.0.0
- Initial release with 26 emotions and 94 personality traits
- Real-time processing with <2s response times
- Content-hash caching implementation
- Z-score normalization with rolling baselines