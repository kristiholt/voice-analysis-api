# Option B Upgrade Path - Lazy Loading Implementation

## Current State (Option A) ✅
- **Health App**: Port 80 - Instant response (~0.002s)
- **Main API**: Port 8000 - Full functionality with ML models
- **Deployment**: Passes health checks, allows ML models to load separately

## Option B Implementation Steps

### 1. Create Lazy Loading Architecture
```python
# production_package/app/lazy_models.py
class LazyModelManager:
    def __init__(self):
        self._emotion_model = None
        self._trait_model = None
        self._enhanced_manager = None
    
    @property
    def emotion_model(self):
        if self._emotion_model is None:
            from .enhanced_models import EnhancedEmotionModel
            self._emotion_model = EnhancedEmotionModel()
        return self._emotion_model
    
    # Similar for other models...
```

### 2. Modify main.py Imports
```python
# Change from:
from .enhanced_models import enhanced_model_manager

# To:
from .lazy_models import lazy_model_manager as model_manager
```

### 3. Update replit.toml
```toml
[deployment]
# Switch to single app with lazy loading:
run = "uvicorn production_package.app.main:app --host 0.0.0.0 --port ${PORT:-80} --workers 1"
```

### 4. Test Health Check Performance
- Ensure `/` responds in <2 seconds
- Verify ML models load on first API call
- Confirm deployment passes health checks

## Benefits of Option B
- **Single Application**: Cleaner architecture
- **On-Demand Loading**: Models load only when needed
- **Better Resource Usage**: No separate health app process
- **Simpler Monitoring**: One service to track

## Rollback Plan
If Option B fails, revert to:
```toml
[deployment]
run = "cd production_package && python3 start_health_and_main.py"
```

## Testing Checklist
- [ ] Health check responds in <2 seconds
- [ ] First API call loads models successfully
- [ ] Subsequent calls are fast (models cached)
- [ ] Deployment health checks pass
- [ ] No import-time model loading