#!/usr/bin/env python3
"""Demonstration script for NightScan optimized ML serving with connection pooling.

This script demonstrates:
1. Connection pooling for database and Redis
2. Model instance pooling for load balancing
3. Batch inference optimization
4. Performance monitoring and metrics
5. Integration with existing API

Usage:
    python demo_optimized_serving.py --mode [demo|benchmark|server]
"""

import asyncio
import argparse
import logging
import time
import numpy as np
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optimization modules
try:
    from model_serving_optimization import (
        get_optimized_serving_manager,
        ConnectionPoolConfig,
        OptimizedModelServingManager
    )
    from optimized_api_integration import (
        get_api_integration,
        register_optimized_endpoints,
        enhance_existing_api_server
    )
    from config import get_config
    
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Optimization modules not available: {e}")
    OPTIMIZATION_AVAILABLE = False


class OptimizedServingDemo:
    """Demonstration class for optimized ML serving capabilities."""
    
    def __init__(self):
        self.config = get_config() if OPTIMIZATION_AVAILABLE else None
        self.serving_manager = None
        self.demo_results = []
        
    async def initialize(self) -> bool:
        """Initialize the demo environment."""
        if not OPTIMIZATION_AVAILABLE:
            logger.error("Optimization modules not available")
            return False
            
        try:
            logger.info("🚀 Initializing NightScan Optimized ML Serving Demo")
            
            # Get optimized serving manager
            self.serving_manager = await get_optimized_serving_manager()
            
            # Deploy demo model
            success = await self._deploy_demo_model()
            
            if success:
                logger.info("✅ Demo environment initialized successfully")
                return True
            else:
                logger.error("❌ Failed to deploy demo model")
                return False
                
        except Exception as e:
            logger.error(f"❌ Demo initialization failed: {e}")
            return False
    
    async def _deploy_demo_model(self) -> bool:
        """Deploy a demo model for testing."""
        def create_demo_model():
            """Create a simple demo model for wildlife detection."""
            import torch
            import torch.nn as nn
            
            # Create a simple CNN model for demonstration
            model = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=64, stride=16),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(64),
                nn.Flatten(),
                nn.Linear(32 * 64, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 10)  # 10 wildlife classes
            )
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            logger.info(f"Created demo model on {device}")
            return model
        
        deployment_id = "wildlife_detector_demo"
        
        try:
            success = await self.serving_manager.deploy_optimized_model(
                deployment_id, create_demo_model
            )
            
            if success:
                logger.info(f"✅ Demo model deployed: {deployment_id}")
            else:
                logger.error(f"❌ Failed to deploy demo model: {deployment_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Demo model deployment failed: {e}")
            return False
    
    async def run_single_prediction_demo(self) -> Dict[str, Any]:
        """Demonstrate single optimized prediction."""
        logger.info("🔮 Running single prediction demo...")
        
        # Generate synthetic audio data
        audio_data = np.random.randn(44100).astype(np.float32)  # 1 second of audio at 44.1kHz
        request_id = "demo_single_001"
        
        start_time = time.time()
        
        result = await self.serving_manager.predict_async(
            deployment_id="wildlife_detector_demo",
            audio_data=audio_data,
            request_id=request_id,
            user_id="demo_user",
            priority=1
        )
        
        end_time = time.time()
        
        demo_result = {
            'type': 'single_prediction',
            'request_id': request_id,
            'latency_ms': (end_time - start_time) * 1000,
            'result': result,
            'audio_shape': audio_data.shape,
            'success': 'error' not in result
        }
        
        self.demo_results.append(demo_result)
        
        if demo_result['success']:
            logger.info(f"✅ Single prediction completed in {demo_result['latency_ms']:.2f}ms")
            logger.info(f"   Predicted class: {result.get('predicted_class', 'N/A')}")
            logger.info(f"   Confidence: {result.get('confidence', 0):.3f}")
            logger.info(f"   Cache hit: {result.get('cache_hit', False)}")
        else:
            logger.error(f"❌ Single prediction failed: {result.get('error', 'Unknown error')}")
        
        return demo_result
    
    async def run_batch_processing_demo(self, batch_size: int = 5) -> Dict[str, Any]:
        """Demonstrate batch processing optimization."""
        logger.info(f"📦 Running batch processing demo with {batch_size} requests...")
        
        # Generate multiple audio samples
        audio_samples = [
            np.random.randn(44100).astype(np.float32) for _ in range(batch_size)
        ]
        
        start_time = time.time()
        
        # Submit all requests concurrently
        tasks = []
        for i, audio_data in enumerate(audio_samples):
            task = self.serving_manager.predict_async(
                deployment_id="wildlife_detector_demo",
                audio_data=audio_data,
                request_id=f"demo_batch_{i:03d}",
                user_id="demo_user",
                priority=0  # Normal priority for batch
            )
            tasks.append(task)
        
        # Wait for all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Analyze results
        successful_results = [r for r in results if isinstance(r, dict) and 'error' not in r]
        failed_results = [r for r in results if not isinstance(r, dict) or 'error' in r]
        
        demo_result = {
            'type': 'batch_processing',
            'batch_size': batch_size,
            'total_latency_ms': (end_time - start_time) * 1000,
            'average_latency_ms': (end_time - start_time) * 1000 / batch_size,
            'successful_requests': len(successful_results),
            'failed_requests': len(failed_results),
            'success_rate': len(successful_results) / batch_size * 100,
            'results': results
        }
        
        self.demo_results.append(demo_result)
        
        logger.info(f"✅ Batch processing completed in {demo_result['total_latency_ms']:.2f}ms")
        logger.info(f"   Average per request: {demo_result['average_latency_ms']:.2f}ms")
        logger.info(f"   Success rate: {demo_result['success_rate']:.1f}%")
        logger.info(f"   Successful: {demo_result['successful_requests']}/{demo_result['batch_size']}")
        
        return demo_result
    
    async def run_connection_pool_demo(self) -> Dict[str, Any]:
        """Demonstrate connection pooling benefits."""
        logger.info("🔗 Running connection pool demo...")
        
        # Get optimization statistics
        stats = await self.serving_manager.get_optimization_stats()
        
        demo_result = {
            'type': 'connection_pooling',
            'database_pool': stats.get('connection_pools', {}).get('database', {}),
            'redis_pool': stats.get('connection_pools', {}).get('redis', {}),
            'model_pools': stats.get('model_pools', {}),
            'system_resources': stats.get('system_resources', {})
        }
        
        self.demo_results.append(demo_result)
        
        logger.info("📊 Connection Pool Status:")
        db_pool = demo_result['database_pool']
        redis_pool = demo_result['redis_pool']
        
        logger.info(f"   Database: {'✅ Active' if db_pool.get('initialized') else '❌ Inactive'}")
        logger.info(f"     - Min connections: {db_pool.get('min_connections', 'N/A')}")
        logger.info(f"     - Max connections: {db_pool.get('max_connections', 'N/A')}")
        
        logger.info(f"   Redis: {'✅ Active' if redis_pool.get('initialized') else '❌ Inactive'}")
        logger.info(f"     - Max connections: {redis_pool.get('max_connections', 'N/A')}")
        
        # Model pool information
        model_pools = demo_result['model_pools']
        for deployment_id, pool_stats in model_pools.items():
            logger.info(f"   Model Pool ({deployment_id}):")
            logger.info(f"     - Pool size: {pool_stats.get('pool_size', 'N/A')}")
            logger.info(f"     - Total usage: {pool_stats.get('total_usage', 'N/A')}")
            logger.info(f"     - Warmup complete: {'✅' if pool_stats.get('warmup_complete') else '⏳'}")
        
        return demo_result
    
    async def run_performance_benchmark(self, num_requests: int = 50, concurrency: int = 10) -> Dict[str, Any]:
        """Run a performance benchmark."""
        logger.info(f"⚡ Running performance benchmark: {num_requests} requests, {concurrency} concurrent...")
        
        # Generate test data
        test_audio = [np.random.randn(44100).astype(np.float32) for _ in range(num_requests)]
        
        start_time = time.time()
        latencies = []
        successful_requests = 0
        failed_requests = 0
        
        # Run requests in batches to control concurrency
        for i in range(0, num_requests, concurrency):
            batch_end = min(i + concurrency, num_requests)
            batch_audio = test_audio[i:batch_end]
            
            # Create batch of tasks
            tasks = []
            for j, audio_data in enumerate(batch_audio):
                task = self.serving_manager.predict_async(
                    deployment_id="wildlife_detector_demo",
                    audio_data=audio_data,
                    request_id=f"bench_{i+j:03d}",
                    user_id="benchmark_user",
                    timeout=15.0
                )
                tasks.append(task)
            
            # Execute batch
            batch_start = time.time()
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_end_time = time.time()
            
            # Process results
            for result in batch_results:
                if isinstance(result, dict) and 'error' not in result:
                    successful_requests += 1
                    latencies.append(result.get('latency', batch_end_time - batch_start))
                else:
                    failed_requests += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate statistics
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            min_latency = np.min(latencies)
            max_latency = np.max(latencies)
        else:
            avg_latency = p95_latency = p99_latency = min_latency = max_latency = 0
        
        throughput = num_requests / total_time
        success_rate = successful_requests / num_requests * 100
        
        benchmark_result = {
            'type': 'performance_benchmark',
            'num_requests': num_requests,
            'concurrency': concurrency,
            'total_time_s': total_time,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'throughput_rps': throughput,
            'latency_stats': {
                'average_ms': avg_latency * 1000,
                'p95_ms': p95_latency * 1000,
                'p99_ms': p99_latency * 1000,
                'min_ms': min_latency * 1000,
                'max_ms': max_latency * 1000
            }
        }
        
        self.demo_results.append(benchmark_result)
        
        logger.info(f"⚡ Benchmark completed in {total_time:.2f}s")
        logger.info(f"   Throughput: {throughput:.1f} requests/second")
        logger.info(f"   Success rate: {success_rate:.1f}%")
        logger.info(f"   Average latency: {avg_latency*1000:.2f}ms")
        logger.info(f"   P95 latency: {p95_latency*1000:.2f}ms")
        logger.info(f"   P99 latency: {p99_latency*1000:.2f}ms")
        
        return benchmark_result
    
    async def run_full_demo(self):
        """Run the complete demonstration."""
        logger.info("🎯 Starting NightScan Optimized ML Serving Full Demo")
        
        try:
            # Single prediction demo
            await self.run_single_prediction_demo()
            await asyncio.sleep(1)
            
            # Connection pool demo
            await self.run_connection_pool_demo()
            await asyncio.sleep(1)
            
            # Batch processing demo
            await self.run_batch_processing_demo(batch_size=8)
            await asyncio.sleep(1)
            
            # Performance benchmark
            await self.run_performance_benchmark(num_requests=25, concurrency=5)
            
            # Final statistics
            await self._show_final_statistics()
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _show_final_statistics(self):
        """Show final demo statistics."""
        logger.info("📊 Final Demo Statistics")
        
        # Get optimization stats
        stats = await self.serving_manager.get_optimization_stats()
        
        logger.info("\n🔧 System Configuration:")
        system_resources = stats.get('system_resources', {})
        logger.info(f"   Memory usage: {system_resources.get('memory_usage_mb', 0):.1f} MB")
        logger.info(f"   CPU usage: {system_resources.get('cpu_percent', 0):.1f}%")
        logger.info(f"   Available memory: {system_resources.get('available_memory_mb', 0):.1f} MB")
        
        logger.info("\n📈 Performance Metrics:")
        performance_metrics = stats.get('performance_metrics', {})
        for deployment_id, metrics in performance_metrics.items():
            logger.info(f"   {deployment_id}:")
            logger.info(f"     - Total requests: {metrics.get('total_requests', 0)}")
            logger.info(f"     - Successful: {metrics.get('successful_requests', 0)}")
            logger.info(f"     - Failed: {metrics.get('failed_requests', 0)}")
            logger.info(f"     - Average latency: {metrics.get('average_latency', 0)*1000:.2f}ms")
            logger.info(f"     - Cache hits: {metrics.get('cache_hits', 0)}")
            logger.info(f"     - Cache misses: {metrics.get('cache_misses', 0)}")
        
        logger.info("\n🏆 Demo Summary:")
        total_demos = len(self.demo_results)
        successful_demos = len([r for r in self.demo_results if r.get('success', True)])
        logger.info(f"   Total demos run: {total_demos}")
        logger.info(f"   Successful demos: {successful_demos}")
        logger.info(f"   Success rate: {successful_demos/total_demos*100:.1f}%")
        
        # Save results to file
        self._save_demo_results()
    
    def _save_demo_results(self):
        """Save demo results to JSON file."""
        try:
            results_file = f"demo_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'demo_results': self.demo_results,
                    'configuration': {
                        'optimized_serving_enabled': True,
                        'model_pool_size': 3,
                        'batch_timeout_ms': 100.0,
                        'max_batch_size': 8
                    }
                }, f, indent=2, default=str)
            
            logger.info(f"💾 Demo results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save demo results: {e}")
    
    async def cleanup(self):
        """Clean up demo resources."""
        try:
            if self.serving_manager:
                await self.serving_manager.undeploy_optimized_model("wildlife_detector_demo")
                await self.serving_manager.shutdown()
            
            logger.info("🧹 Demo cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Demo cleanup failed: {e}")


def run_flask_server_demo():
    """Run Flask server with optimized serving integration."""
    if not OPTIMIZATION_AVAILABLE:
        logger.error("Optimization modules not available for server demo")
        return
    
    try:
        from flask import Flask
        from optimized_api_integration import register_optimized_endpoints
        
        app = Flask(__name__)
        
        # Register optimized endpoints
        register_optimized_endpoints(app)
        
        @app.route("/")
        def index():
            return """
            <h1>NightScan Optimized ML Serving Demo Server</h1>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/api/optimized/health">Health Check</a></li>
                <li><a href="/api/optimized/stats">Statistics</a></li>
                <li>POST /api/optimized/predict - Optimized prediction endpoint</li>
            </ul>
            <p>Use curl or similar tool to test prediction endpoint:</p>
            <pre>curl -X POST -F "file=@your_audio.wav" http://localhost:8002/api/optimized/predict</pre>
            """
        
        logger.info("🌐 Starting Flask server demo on http://localhost:8002")
        logger.info("📝 Available endpoints:")
        logger.info("   GET  / - Demo homepage")
        logger.info("   GET  /api/optimized/health - Health check")
        logger.info("   GET  /api/optimized/stats - Performance statistics")
        logger.info("   POST /api/optimized/predict - Optimized prediction")
        
        # Setup graceful shutdown
        def signal_handler(signum, frame):
            logger.info("🛑 Shutting down server...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        app.run(host="0.0.0.0", port=8002, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"❌ Server demo failed: {e}")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="NightScan Optimized ML Serving Demo")
    parser.add_argument(
        "--mode", 
        choices=["demo", "benchmark", "server"],
        default="demo",
        help="Demo mode to run"
    )
    parser.add_argument(
        "--requests", 
        type=int, 
        default=50,
        help="Number of requests for benchmark mode"
    )
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=10,
        help="Concurrency level for benchmark mode"
    )
    
    args = parser.parse_args()
    
    if not OPTIMIZATION_AVAILABLE:
        logger.error("❌ Optimization modules not available")
        logger.error("   Please ensure all dependencies are installed:")
        logger.error("   - asyncpg (for database connection pooling)")
        logger.error("   - aioredis (for Redis connection pooling)")
        logger.error("   - uvloop (for improved async performance)")
        return 1
    
    if args.mode == "server":
        run_flask_server_demo()
        return 0
    
    demo = OptimizedServingDemo()
    
    try:
        # Initialize demo
        success = await demo.initialize()
        if not success:
            logger.error("❌ Failed to initialize demo")
            return 1
        
        # Run demo based on mode
        if args.mode == "demo":
            await demo.run_full_demo()
        elif args.mode == "benchmark":
            await demo.run_performance_benchmark(
                num_requests=args.requests,
                concurrency=args.concurrency
            )
        
        logger.info("✅ Demo completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Demo interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    try:
        # Use uvloop if available for better performance
        import uvloop
        uvloop.run(main())
    except ImportError:
        # Fall back to standard asyncio
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
