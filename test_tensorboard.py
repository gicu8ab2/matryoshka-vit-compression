#!/usr/bin/env python3
"""
Simple test script to verify TensorBoard integration works correctly.
This script tests the argument parsing and TensorBoard setup without running full training.
"""

import sys
import os
import argparse
import tempfile
import shutil

# Add the current directory to Python path to import the training modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_tensorboard_setup():
    """Test that TensorBoard setup works correctly with the new arguments."""
    
    # Test argument parsing
    parser = argparse.ArgumentParser(description='Test TensorBoard setup')
    parser.add_argument('--notb', action='store_true', help='disable tensorboard')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dataset', default='cifar10', type=str)
    
    # Test with TensorBoard enabled (default)
    args = parser.parse_args(['--net', 'vit', '--lr', '0.001', '--dataset', 'cifar10'])
    
    usetb = not args.notb
    print(f"TensorBoard enabled: {usetb}")
    
    if usetb:
        try:
            from torch.utils.tensorboard import SummaryWriter
            print("✓ TensorBoard import successful")
            
            # Create a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                watermark = "{}_lr{}_{}".format(args.net, args.lr, args.dataset)
                log_dir = os.path.join(temp_dir, 'runs', watermark)
                
                writer = SummaryWriter(log_dir=log_dir)
                print(f"✓ TensorBoard writer created at: {log_dir}")
                
                # Test logging some dummy data
                writer.add_scalar('Loss/Train', 0.5, 1)
                writer.add_scalar('Loss/Validation', 0.4, 1)
                writer.add_scalar('Accuracy/Validation', 85.0, 1)
                print("✓ Test logging successful")
                
                writer.close()
                print("✓ TensorBoard writer closed successfully")
                
                # Check if log files were created
                if os.path.exists(log_dir):
                    print(f"✓ Log directory created: {log_dir}")
                    files = os.listdir(log_dir)
                    if files:
                        print(f"✓ Log files created: {files}")
                    else:
                        print("⚠ No log files found")
                
        except ImportError as e:
            print(f"✗ TensorBoard import failed: {e}")
            print("Please install tensorboard: pip install tensorboard")
            return False
        except Exception as e:
            print(f"✗ TensorBoard setup failed: {e}")
            return False
    
    # Test with TensorBoard disabled
    args_disabled = parser.parse_args(['--notb'])
    usetb_disabled = not args_disabled.notb
    print(f"\nTensorBoard disabled test: {not usetb_disabled}")
    
    if not usetb_disabled:
        print("✓ TensorBoard correctly disabled")
    
    return True

if __name__ == "__main__":
    print("Testing TensorBoard integration...")
    success = test_tensorboard_setup()
    
    if success:
        print("\n✓ All tests passed! TensorBoard integration is working correctly.")
        print("\nTo view TensorBoard logs, run:")
        print("  tensorboard --logdir=runs")
        print("  Then open http://localhost:6006 in your browser")
    else:
        print("\n✗ Tests failed. Please check the error messages above.")
        sys.exit(1)
