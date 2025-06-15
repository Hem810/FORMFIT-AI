#!/usr/bin/env python3
"""
FORMFIT-AI System Setup Verification Script
==========================================

This script verifies that all required dependencies and system components
are properly installed and configured for FORMFIT-AI to run successfully.

Usage: python test_setup.py
"""

import sys
import os
import subprocess
import importlib
import json
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

class SystemTester:
    """System verification and testing class"""
    
    def __init__(self):
        self.results = {
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'tests': []
        }
        self.required_python_version = (3, 7)
        self.required_packages = {
            'opencv-python': '4.5.0',
            'onnxruntime': '1.10.0',
            'torch': '1.9.0',
            'torchvision':'',
            'numpy': '1.21.0',
            'Pillow': '8.3.0',
            'tkinter': None  # Built-in module
        }
        
    def print_header(self):
        """Print script header"""
        print(f"{Colors.BLUE}{Colors.BOLD}")
        print("=" * 60)
        print("         FORMFIT-AI System Setup Verification")
        print("=" * 60)
        print(f"{Colors.END}")
        
    def print_test(self, test_name, status, message=""):
        """Print test result with color coding"""
        if status == "PASS":
            color = Colors.GREEN
            self.results['passed'] += 1
        elif status == "FAIL":
            color = Colors.RED
            self.results['failed'] += 1
        else:  # WARNING
            color = Colors.YELLOW
            self.results['warnings'] += 1
            
        print(f"{color}[{status}]{Colors.END} {test_name}")
        if message:
            print(f"       {message}")
        
        self.results['tests'].append({
            'name': test_name,
            'status': status,
            'message': message
        })
        
    def check_python_version(self):
        """Verify Python version compatibility"""
        current_version = sys.version_info[:2]
        required = self.required_python_version
        
        if current_version >= required:
            self.print_test(
                f"Python Version ({'.'.join(map(str, current_version))})",
                "PASS",
                f"Meets minimum requirement {'.'.join(map(str, required))}"
            )
            return True
        else:
            self.print_test(
                f"Python Version ({'.'.join(map(str, current_version))})",
                "FAIL",
                f"Requires Python {'.'.join(map(str, required))} or higher"
            )
            return False
            
    def check_package_installation(self):
        """Check if required packages are installed"""
        all_installed = True
        
        for package, min_version in self.required_packages.items():
            try:
                if package == 'opencv-python':
                    # OpenCV has different import name
                    import cv2
                    version = cv2.__version__
                    package_name = 'OpenCV'
                elif package == 'Pillow':
                    from PIL import Image
                    import PIL
                    version = PIL.__version__
                    package_name = 'Pillow (PIL)'
                elif package == 'tkinter':
                    import tkinter
                    version = "Built-in"
                    package_name = 'Tkinter'
                else:
                    module = importlib.import_module(package.replace('-', '_'))
                    version = getattr(module, '__version__', 'Unknown')
                    package_name = package
                    
                self.print_test(
                    f"{package_name} ({version})",
                    "PASS",
                    "Installed and importable"
                )
                
            except ImportError as e:
                self.print_test(
                    f"{package}",
                    "FAIL",
                    f"Not installed or not importable: {str(e)}"
                )
                all_installed = False
                
        return all_installed
        
    def check_camera_access(self):
        """Test camera accessibility"""
        try:
            import cv2
            
            # Try to open default camera
            cap = cv2.VideoCapture(0)
            
            if cap.isOpened():
                # Try to read a frame
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    self.print_test(
                        "Camera Access",
                        "PASS",
                        "Default camera accessible and working"
                    )
                    return True
                else:
                    self.print_test(
                        "Camera Access",
                        "FAIL",
                        "Camera opens but cannot read frames"
                    )
            else:
                self.print_test(
                    "Camera Access",
                    "FAIL",
                    "Cannot access default camera (index 0)"
                )
                
        except Exception as e:
            self.print_test(
                "Camera Access",
                "FAIL",
                f"Error testing camera: {str(e)}"
            )
            
        return False
        
    def check_file_structure(self):
        """Verify required files and directories exist"""
        required_files = [
            'models/hrnet_pose.onnx',
            'scripts/main.py',
        ]
        
        required_dirs = [
            'models',
            'scripts'
        ]
        
        all_present = True
        
        # Check files
        for file_path in required_files:
            if os.path.exists(file_path):
                self.print_test(
                    f"File: {file_path}",
                    "PASS",
                    "Required file present"
                )
            else:
                self.print_test(
                    f"File: {file_path}",
                    "WARNING" if file_path == 'README.md' else "FAIL",
                    "File not found"
                )
                if file_path != 'README.md':
                    all_present = False
                    
        # Check directories
        for dir_path in required_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                self.print_test(
                    f"Directory: {dir_path}/",
                    "PASS",
                    "Required directory present"
                )
            else:
                self.print_test(
                    f"Directory: {dir_path}/",
                    "WARNING",
                    "Directory not found - may need to be created"
                )
                
        return all_present
        
    def check_model_files(self):
        """Check for AI model files"""
        model_dir = Path('models')
        required_models = [
            '/hrnet_pose.onnx'
        ]
        
        if not model_dir.exists():
            self.print_test(
                "Model Directory",
                "WARNING",
                "Models directory doesn't exist - create it and download models"
            )
            return False
            
        models_found = False
        for model_file in required_models:
            model_path = model_dir / model_file
            if model_path.exists():
                size_mb = model_path.stat().st_size / (1024 * 1024)
                self.print_test(
                    f"Model: {model_file}",
                    "PASS",
                    f"Model file present ({size_mb:.1f} MB)"
                )
                models_found = True
            else:
                self.print_test(
                    f"Model: {model_file}",
                    "WARNING",
                    "Model file not found - download required"
                )
                
        return models_found
        
    def check_system_resources(self):
        """Check system resources and performance"""
        try:
            import psutil
            
            # Check available RAM
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb >= 2.0:
                self.print_test(
                    f"Available RAM ({available_gb:.1f} GB)",
                    "PASS",
                    "Sufficient memory for AI processing"
                )
            else:
                self.print_test(
                    f"Available RAM ({available_gb:.1f} GB)",
                    "WARNING",
                    "Low memory - close other applications for better performance"
                )
                
            # Check CPU cores
            cpu_count = psutil.cpu_count()
            if cpu_count >= 4:
                self.print_test(
                    f"CPU Cores ({cpu_count})",
                    "PASS",
                    "Adequate processing power"
                )
            else:
                self.print_test(
                    f"CPU Cores ({cpu_count})",
                    "WARNING",
                    "Limited processing power - expect slower performance"
                )
                
        except ImportError:
            self.print_test(
                "System Resources",
                "WARNING",
                "psutil not available - cannot check system resources"
            )
            
            
    def run_performance_test(self):
        """Run a quick performance benchmark"""
        try:
            import time
            import numpy as np
            
            # Simple computation benchmark
            start_time = time.time()
            
            # Simulate pose processing workload
            for _ in range(100):
                data = np.random.rand(33, 3)  # 33 pose landmarks with x,y,z
                angles = np.arccos(np.dot(data[:10], data[10:20].T))
                
            end_time = time.time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            
            if processing_time < 100:
                self.print_test(
                    f"Performance Test ({processing_time:.1f}ms)",
                    "PASS",
                    "System performance adequate for real-time processing"
                )
            else:
                self.print_test(
                    f"Performance Test ({processing_time:.1f}ms)",
                    "WARNING",
                    "Slower performance - may affect real-time capabilities"
                )
                
        except Exception as e:
            self.print_test(
                "Performance Test",
                "WARNING",
                f"Could not run performance test: {str(e)}"
            )
            
    def generate_report(self):
        """Generate and save test report"""
        report = {
            'timestamp': str(__import__('datetime').datetime.now()),
            'system_info': {
                'python_version': '.'.join(map(str, sys.version_info[:3])),
                'platform': sys.platform,
                'architecture': __import__('platform').architecture()[0]
            },
            'test_results': self.results
        }
        
        try:
            with open('setup_test_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n{Colors.BLUE}Test report saved to: setup_test_report.json{Colors.END}")
        except Exception as e:
            print(f"\n{Colors.YELLOW}Warning: Could not save test report: {e}{Colors.END}")
            
    def print_summary(self):
        """Print test summary"""
        print(f"\n{Colors.BOLD}Test Summary:{Colors.END}")
        print(f"{Colors.GREEN}Passed: {self.results['passed']}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.results['failed']}{Colors.END}")
        print(f"{Colors.YELLOW}Warnings: {self.results['warnings']}{Colors.END}")
        
        if self.results['failed'] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ System is ready for FORMFIT-AI!{Colors.END}")
            print("\nYou can now run: python main.py")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ System setup incomplete{Colors.END}")
            print("\nPlease address the failed tests before running FORMFIT-AI")
            
        if self.results['warnings'] > 0:
            print(f"\n{Colors.YELLOW}Note: Warnings indicate non-critical issues that may affect performance{Colors.END}")
            
    def run_all_tests(self):
        """Execute all system tests"""
        self.print_header()
        
        print(f"{Colors.BOLD}Running system verification tests...\n{Colors.END}")
        
        # Core system tests
        print(f"{Colors.BOLD}System Requirements:{Colors.END}")
        self.check_python_version()
        self.check_system_resources()
        
        # Package installation tests
        print(f"\n{Colors.BOLD}Package Dependencies:{Colors.END}")
        self.check_package_installation()
        
        # Hardware tests
        print(f"\n{Colors.BOLD}Hardware & Peripherals:{Colors.END}")
        self.check_camera_access()
        
        # File structure tests
        print(f"\n{Colors.BOLD}Project Structure:{Colors.END}")
        self.check_file_structure()
        self.check_model_files()
        
        # Functionality tests
        print(f"\n{Colors.BOLD}Core Functionality:{Colors.END}")
        self.run_performance_test()
        
        # Generate report and summary
        self.generate_report()
        self.print_summary()
        
        return self.results['failed'] == 0

def main():
    """Main function"""
    tester = SystemTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
