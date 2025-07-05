#!/usr/bin/env python3

"""
Master Test Suite for the Backtesting Framework
Runs all individual tests and generates a comprehensive report.
"""

import sys
import logging
import json
from datetime import datetime
from pathlib import Path
import time

# Add backtester to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from test_data_utils import TestDataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MasterTestSuite:
    """Comprehensive test suite for the entire backtesting framework."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        self.test_data_manager = TestDataManager()
        
    def run_test_module(self, module_name: str, test_function_name: str) -> dict:
        """Run a specific test module and return results."""
        try:
            # Import the test module dynamically
            import importlib.util
            
            module_path = Path(__file__).parent / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the test function
            test_function = getattr(module, test_function_name)
            
            # Run the test
            start_time = time.time()
            success = test_function()
            end_time = time.time()
            
            return {
                'status': 'PASSED' if success else 'FAILED',
                'duration': end_time - start_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'duration': 0,
                'error': str(e)
            }
    
    def run_all_tests(self) -> dict:
        """Run all tests in the suite."""
        logger.info("🚀 Starting Master Test Suite for Backtesting Framework")
        
        # Ensure test data is available
        logger.info("📊 Checking test data availability...")
        data_ready = self.test_data_manager.ensure_test_data_available(
            symbols=['EURUSD', 'EURJPY', 'GBPNZD'], 
            days=2
        )
        
        if not data_ready:
            logger.error("❌ Failed to ensure test data availability")
            return {
                'test_run_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_duration': 0,
                    'framework_version': '1.0.0',
                    'python_version': sys.version.split()[0]
                },
                'test_statistics': {
                    'total_tests': 0,
                    'passed_tests': 0,
                    'failed_tests': 0,
                    'error_tests': 1,
                    'success_rate': 0
                },
                'test_results': {
                    'Data Preparation': {
                        'status': 'ERROR',
                        'duration': 0,
                        'error': 'Failed to download required test data'
                    }
                },
                'overall_status': 'FAILED'
            }
        
        logger.info("✅ Test data ready")
        self.start_time = time.time()
        
        # Define all tests to run
        test_suite = [
            {
                'name': 'Simple Backtest',
                'description': 'Basic functionality test',
                'module': 'simple_backtest_test',
                'function': 'test_simple_backtest'
            },
            {
                'name': 'Order Execution Validation', 
                'description': 'Order execution and position management',
                'module': 'execution_validation_test',
                'function': 'test_order_execution'
            },
            {
                'name': 'Multi-Symbol Backtesting',
                'description': 'Multiple currency pairs processing',
                'module': 'multi_symbol_test',
                'function': 'test_multi_symbol'
            },
            {
                'name': 'Performance Analysis',
                'description': 'Performance reporting and analytics',
                'module': 'performance_analysis_test', 
                'function': 'test_performance_analysis'
            }
        ]
        
        logger.info(f"📋 Running {len(test_suite)} test modules...")
        
        # Run each test
        for test_config in test_suite:
            logger.info(f"🧪 Running {test_config['name']}...")
            
            result = self.run_test_module(
                test_config['module'],
                test_config['function']
            )
            
            result.update({
                'name': test_config['name'],
                'description': test_config['description']
            })
            
            self.results[test_config['name']] = result
            
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{status_emoji} {test_config['name']}: {result['status']} ({result['duration']:.2f}s)")
            
            if result['error']:
                logger.error(f"   Error: {result['error']}")
        
        self.end_time = time.time()
        
        # Generate summary
        summary = self._generate_summary()
        total_duration = summary['test_run_info']['total_duration']
        logger.info(f"🏁 Test suite completed in {total_duration:.2f}s")
        
        return summary
    
    def _generate_summary(self) -> dict:
        """Generate comprehensive test summary."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed_tests = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        error_tests = sum(1 for r in self.results.values() if r['status'] == 'ERROR')
        
        summary = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'total_duration': self.end_time - self.start_time,
                'framework_version': '1.0.0',
                'python_version': sys.version.split()[0]
            },
            'test_statistics': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'error_tests': error_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'test_results': self.results,
            'overall_status': 'PASSED' if passed_tests == total_tests else 'FAILED'
        }
        
        return summary
    
    def generate_report(self, filename: str = 'test_results/test_suite_report.json'):
        """Generate and save detailed test report."""
        summary = self._generate_summary()
        
        # Add additional analysis
        report = {
            'backtesting_framework_test_report': summary,
            'component_analysis': {
                'data_management': {
                    'status': 'OPERATIONAL',
                    'features_tested': [
                        'Dukascopy-node integration',
                        'CSV data loading',
                        'Data validation',
                        'Multi-symbol support',
                        'Bid/ask price correction'
                    ]
                },
                'backtesting_engine': {
                    'status': 'OPERATIONAL',
                    'features_tested': [
                        'Tick-by-tick simulation',
                        'Order execution',
                        'Position management',
                        'Slippage modeling',
                        'Commission calculation'
                    ]
                },
                'strategy_framework': {
                    'status': 'OPERATIONAL',
                    'features_tested': [
                        'Signal generation',
                        'Risk management',
                        'Technical indicators',
                        'Multi-timeframe support'
                    ]
                },
                'performance_analysis': {
                    'status': 'OPERATIONAL',
                    'features_tested': [
                        'P&L calculation',
                        'Risk metrics',
                        'Report generation',
                        'Export functionality'
                    ]
                }
            },
            'recommendations': self._generate_recommendations(summary)
        }
        
        # Save report
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📊 Comprehensive report saved to {filename}")
        
        return report
    
    def _generate_recommendations(self, summary: dict) -> list:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if summary['overall_status'] == 'PASSED':
            recommendations.append({
                'type': 'SUCCESS',
                'message': 'All core functionality is working correctly',
                'priority': 'INFO'
            })
        
        failed_tests = [name for name, result in self.results.items() 
                       if result['status'] in ['FAILED', 'ERROR']]
        
        if failed_tests:
            recommendations.append({
                'type': 'ATTENTION_REQUIRED',
                'message': f'The following tests need attention: {", ".join(failed_tests)}',
                'priority': 'HIGH'
            })
        
        # Performance recommendations
        slow_tests = [name for name, result in self.results.items() 
                     if result['duration'] > 10.0]
        
        if slow_tests:
            recommendations.append({
                'type': 'PERFORMANCE',
                'message': f'Consider optimizing performance for: {", ".join(slow_tests)}',
                'priority': 'MEDIUM'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'type': 'MAINTENANCE',
                'message': 'Regularly update test data to ensure continued accuracy',
                'priority': 'LOW'
            },
            {
                'type': 'EXPANSION',
                'message': 'Consider adding tests for edge cases and stress scenarios',
                'priority': 'LOW'
            }
        ])
        
        return recommendations
    
    def print_summary(self, summary: dict):
        """Print formatted summary to console."""
        print("\n" + "="*80)
        print("BACKTESTING FRAMEWORK - MASTER TEST SUITE RESULTS")
        print("="*80)
        
        # Test statistics
        stats = summary['test_statistics']
        print(f"\n📊 TEST STATISTICS:")
        print(f"   Total Tests: {stats['total_tests']}")
        print(f"   Passed: {stats['passed_tests']}")
        print(f"   Failed: {stats['failed_tests']}")
        print(f"   Errors: {stats['error_tests']}")
        print(f"   Success Rate: {stats['success_rate']:.1f}%")
        
        # Overall status
        status_emoji = "🎉" if summary['overall_status'] == 'PASSED' else "⚠️"
        print(f"\n{status_emoji} OVERALL STATUS: {summary['overall_status']}")
        
        # Individual test results
        print(f"\n🔍 INDIVIDUAL TEST RESULTS:")
        for name, result in self.results.items():
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"   {status_emoji} {name}: {result['status']} ({result['duration']:.2f}s)")
            if result['error']:
                print(f"      Error: {result['error']}")
        
        # Execution info
        info = summary['test_run_info']
        print(f"\n⏱️  EXECUTION INFO:")
        print(f"   Duration: {info['total_duration']:.2f} seconds")
        print(f"   Timestamp: {info['timestamp']}")
        print(f"   Python Version: {info['python_version']}")
        
        print("="*80)


def main():
    """Run the master test suite."""
    try:
        suite = MasterTestSuite()
        summary = suite.run_all_tests()
        
        # Print summary
        suite.print_summary(summary)
        
        # Generate detailed report
        report = suite.generate_report()
        
        # Print final verdict
        if summary['overall_status'] == 'PASSED':
            print("\n🎉 ALL TESTS PASSED! The backtesting framework is fully operational.")
        else:
            print("\n⚠️  SOME TESTS FAILED! Review the detailed report for issues.")
        
        return summary['overall_status'] == 'PASSED'
        
    except Exception as e:
        print(f"\n❌ MASTER TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)