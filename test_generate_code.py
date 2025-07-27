#!/usr/bin/env python3
"""
Test suite for generate_code.py script.
"""

import subprocess
import sys
import os
import tempfile

def test_basic_generation():
    """Test basic code generation."""
    print("🔍 Testing Basic Code Generation")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            'python', 'generate_code.py', 'calculate sum of two numbers'
        ], capture_output=True, text=True, cwd='/home/runner/work/jubilant-palm-tree/jubilant-palm-tree',
        env=dict(os.environ, **{
            'PATH': '/home/runner/.local/share/gem/ruby/3.2.0/bin:' + os.environ.get('PATH', ''),
            'GEM_PATH': '/home/runner/.local/share/gem/ruby/3.2.0:' + os.environ.get('GEM_PATH', '')
        }), timeout=60)
        
        if result.returncode == 0:
            print("✅ Script executed successfully")
            print("Generated output:")
            print("-" * 20)
            print(result.stdout)
            
            # Check for expected elements in output
            expected_elements = [
                "Generated Ruby Code",
                "def "
            ]
            
            # Check for one of the loading messages (depends on generator type)
            loading_messages = [
                "Loading Code Generation Models",
                "Loading Autoregressive Code Generation Models"
            ]
            
            has_loading_message = any(msg in result.stdout for msg in loading_messages)
            if has_loading_message:
                print(f"✅ Found expected loading message")
            else:
                print(f"❌ Missing expected loading message")
                return False
            
            for element in expected_elements:
                if element in result.stdout:
                    print(f"✅ Found expected element: '{element}'")
                else:
                    print(f"❌ Missing expected element: '{element}'")
                    return False
            
            return True
        else:
            print(f"❌ Script failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_help_command():
    """Test help command."""
    print("\n🔍 Testing Help Command")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            'python', 'generate_code.py', '--help'
        ], capture_output=True, text=True, cwd='/home/runner/work/jubilant-palm-tree/jubilant-palm-tree')
        
        if result.returncode == 0 and "Generate Ruby code from natural language" in result.stdout:
            print("✅ Help command works correctly")
            return True
        else:
            print(f"❌ Help command failed")
            return False
            
    except Exception as e:
        print(f"❌ Help test failed: {e}")
        return False

def test_standard_generator_flag():
    """Test --use-standard flag for backward compatibility."""
    print("\n🔍 Testing Standard Generator Flag")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            'python', 'generate_code.py', 'simple test', '--use-standard'
        ], capture_output=True, text=True, cwd='/home/runner/work/jubilant-palm-tree/jubilant-palm-tree',
        env=dict(os.environ, **{
            'PATH': '/home/runner/.local/share/gem/ruby/3.2.0/bin:' + os.environ.get('PATH', ''),
            'GEM_PATH': '/home/runner/.local/share/gem/ruby/3.2.0:' + os.environ.get('GEM_PATH', '')
        }), timeout=60)
        
        if result.returncode == 0:
            print("✅ Standard generator flag works")
            
            # Check that it's using the standard generator
            if "Using Standard Code Generator" in result.stdout:
                print("✅ Correctly using standard generator")
                return True
            else:
                print("❌ Not using standard generator as expected")
                return False
        else:
            print(f"❌ Standard generator test failed with return code {result.returncode}")
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Standard generator test failed: {e}")
        return False

def test_method_name_override():
    """Test custom method name."""
    print("\n🔍 Testing Method Name Override")
    print("=" * 40)
    
    try:
        result = subprocess.run([
            'python', 'generate_code.py', 'get user data', '--method-name', 'fetch_user'
        ], capture_output=True, text=True, cwd='/home/runner/work/jubilant-palm-tree/jubilant-palm-tree',
        env=dict(os.environ, **{
            'PATH': '/home/runner/.local/share/gem/ruby/3.2.0/bin:' + os.environ.get('PATH', ''),
            'GEM_PATH': '/home/runner/.local/share/gem/ruby/3.2.0:' + os.environ.get('GEM_PATH', '')
        }), timeout=60)
        
        if result.returncode == 0 and "def fetch_user" in result.stdout:
            print("✅ Method name override works correctly")
            return True
        else:
            print(f"❌ Method name override failed")
            print("Output:", result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ Method name test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing generate_code.py Script")
    print("=" * 50)
    
    tests = [
        test_help_command,
        test_basic_generation,
        test_standard_generator_flag,
        test_method_name_override
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n📊 Test Results")
    print("=" * 20)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {passed}/{len(tests)} ({100*passed/len(tests):.1f}%)")
    
    if failed == 0:
        print("\n🎉 All tests passed!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)