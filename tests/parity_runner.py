"""
Parity test runner for golden audio inputs.
Tests API responses and measures performance.
"""

import os
import sys
import asyncio
import json
import time
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import get_env_var

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParityRunner:
    """Test runner for API parity validation"""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or get_env_var("API_BASE_URL", "http://localhost:8000")
        self.api_key = api_key or get_env_var("TEST_API_KEY", "")
        self.golden_inputs_dir = Path("tests/golden_inputs")
        self.results_dir = Path("tests/results")
        
        # Ensure results directory exists
        self.results_dir.mkdir(exist_ok=True)
        
        if not self.api_key:
            logger.error("TEST_API_KEY environment variable is required")
            sys.exit(1)
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run tests on all audio files in golden_inputs directory.
        
        Returns:
            Test results summary
        """
        if not self.golden_inputs_dir.exists():
            logger.error(f"Golden inputs directory not found: {self.golden_inputs_dir}")
            return {"error": "Golden inputs directory not found"}
        
        # Find all audio files
        audio_files = []
        for ext in ['.wav', '.flac', '.mp3', '.m4a', '.ogg']:
            audio_files.extend(self.golden_inputs_dir.glob(f"*{ext}"))
            audio_files.extend(self.golden_inputs_dir.glob(f"*{ext.upper()}"))
        
        if not audio_files:
            logger.warning(f"No audio files found in {self.golden_inputs_dir}")
            return {"error": "No audio files found"}
        
        logger.info(f"Found {len(audio_files)} audio files to test")
        
        # Run tests
        results = []
        total_start_time = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for audio_file in audio_files:
                try:
                    result = await self._test_single_file(client, audio_file)
                    results.append(result)
                    logger.info(f"✓ {audio_file.name}: {result['processing_ms']}ms, {result['request_id']}")
                except Exception as e:
                    error_result = {
                        'filename': audio_file.name,
                        'status': 'error',
                        'error': str(e),
                        'processing_ms': 0,
                        'request_id': None
                    }
                    results.append(error_result)
                    logger.error(f"✗ {audio_file.name}: {e}")
        
        total_time = time.time() - total_start_time
        
        # Compute summary statistics
        successful_tests = [r for r in results if r['status'] == 'success']
        failed_tests = [r for r in results if r['status'] == 'error']
        
        if successful_tests:
            processing_times = [r['processing_ms'] for r in successful_tests]
            avg_processing_time = sum(processing_times) / len(processing_times)
            max_processing_time = max(processing_times)
            min_processing_time = min(processing_times)
        else:
            avg_processing_time = max_processing_time = min_processing_time = 0
        
        summary = {
            'total_files': len(audio_files),
            'successful': len(successful_tests),
            'failed': len(failed_tests),
            'total_time_seconds': round(total_time, 2),
            'avg_processing_ms': round(avg_processing_time, 2),
            'min_processing_ms': min_processing_time,
            'max_processing_ms': max_processing_time,
            'results': results
        }
        
        # Save results
        results_file = self.results_dir / f"parity_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Test results saved to {results_file}")
        return summary
    
    async def _test_single_file(self, client: httpx.AsyncClient, audio_file: Path) -> Dict[str, Any]:
        """Test a single audio file"""
        start_time = time.time()
        
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_data = f.read()
        
        # Prepare request
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {"audio": (audio_file.name, audio_data, "audio/wav")}
        
        # Make request
        response = await client.post(
            f"{self.base_url}/v1/voice/analyze",
            headers=headers,
            files=files
        )
        
        end_time = time.time()
        total_time_ms = int((end_time - start_time) * 1000)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'filename': audio_file.name,
                'status': 'success',
                'request_id': data.get('request_id'),
                'processing_ms': data.get('processing_ms', 0),
                'total_time_ms': total_time_ms,
                'audio_ms': data.get('audio_ms', 0),
                'emotion_count': len(data.get('scores', {}).get('emotions', {})),
                'trait_count': len(data.get('scores', {}).get('traits', {})),
                'warnings': data.get('warnings', [])
            }
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.headers.get('content-type', '').startswith('application/json') else response.text
            return {
                'filename': audio_file.name,
                'status': 'error',
                'error': f"HTTP {response.status_code}: {error_detail}",
                'processing_ms': 0,
                'total_time_ms': total_time_ms,
                'request_id': None
            }
    
    async def test_health_endpoint(self) -> Dict[str, Any]:
        """Test the health endpoint"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/health")
                
                if response.status_code == 200:
                    return {
                        'status': 'success',
                        'data': response.json()
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f"HTTP {response.status_code}: {response.text}"
                    }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def test_authentication(self) -> Dict[str, Any]:
        """Test authentication with invalid key"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {"Authorization": "Bearer invalid_key"}
                
                # Try to access analyze endpoint with invalid key
                response = await client.post(
                    f"{self.base_url}/v1/voice/analyze",
                    headers=headers,
                    files={"audio": ("test.wav", b"dummy", "audio/wav")}
                )
                
                if response.status_code == 403:
                    return {
                        'status': 'success',
                        'message': 'Authentication correctly rejected invalid key'
                    }
                else:
                    return {
                        'status': 'error',
                        'error': f"Expected 403, got {response.status_code}"
                    }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print test results summary"""
        print("\n" + "="*60)
        print("PARITY TEST RESULTS")
        print("="*60)
        
        if 'error' in summary:
            print(f"ERROR: {summary['error']}")
            return
        
        print(f"Total files tested: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['successful']/summary['total_files']*100:.1f}%")
        print(f"Total test time: {summary['total_time_seconds']}s")
        
        if summary['successful'] > 0:
            print(f"\nProcessing times:")
            print(f"  Average: {summary['avg_processing_ms']}ms")
            print(f"  Min: {summary['min_processing_ms']}ms")
            print(f"  Max: {summary['max_processing_ms']}ms")
        
        # Print failed tests
        failed_results = [r for r in summary['results'] if r['status'] == 'error']
        if failed_results:
            print(f"\nFailed tests:")
            for result in failed_results:
                print(f"  {result['filename']}: {result['error']}")
        
        print("="*60)


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run parity tests for X Voice API")
    parser.add_argument("--base-url", help="API base URL", default="http://localhost:8000")
    parser.add_argument("--api-key", help="Test API key")
    parser.add_argument("--health-only", action="store_true", help="Test health endpoint only")
    parser.add_argument("--auth-only", action="store_true", help="Test authentication only")
    
    args = parser.parse_args()
    
    runner = ParityRunner(base_url=args.base_url, api_key=args.api_key)
    
    if args.health_only:
        print("Testing health endpoint...")
        result = await runner.test_health_endpoint()
        print(f"Health test: {result}")
        return
    
    if args.auth_only:
        print("Testing authentication...")
        result = await runner.test_authentication()
        print(f"Auth test: {result}")
        return
    
    # Test health first
    print("Testing health endpoint...")
    health_result = await runner.test_health_endpoint()
    if health_result['status'] != 'success':
        print(f"Health check failed: {health_result}")
        return
    
    print("✓ Health endpoint OK")
    
    # Run full parity tests
    print("\nRunning parity tests...")
    summary = await runner.run_all_tests()
    runner.print_summary(summary)


if __name__ == "__main__":
    asyncio.run(main())
