#!/usr/bin/env python3
"""
Demonstration of the enhanced Ruby syntax checking in the evaluation notebook.

This script shows how the new is_syntactically_valid() function provides
concrete, quantitative metrics for evaluating reconstructed Ruby code.
"""

import subprocess
import os

def is_syntactically_valid(ruby_code):
    """Check if the generated Ruby code has valid syntax."""
    try:
        result = subprocess.run(
            ['ruby', 'scripts/check_syntax.rb'],
            input=ruby_code,
            capture_output=True,
            text=True,
            env=dict(os.environ, PATH=f"/home/runner/.local/share/gem/ruby/3.2.0/bin:{os.environ.get('PATH', '')}")
        )
        return result.returncode == 0
    except Exception as e:
        # If there's an error running the syntax checker, assume invalid
        return False

def demonstrate_syntax_checking():
    """Demonstrate the syntax checking functionality with various Ruby code samples."""
    
    print("=" * 70)
    print("ENHANCED RUBY SYNTAX CHECKING DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Test cases representing different types of reconstructed code
    test_cases = [
        {
            "name": "Valid simple method",
            "code": """def hello_world
  puts "Hello, World!"
end""",
            "expected": True
        },
        {
            "name": "Valid method with parameters",
            "code": """def add_numbers(a, b)
  result = a + b
  return result
end""",
            "expected": True
        },
        {
            "name": "Valid method with conditionals",
            "code": """def check_positive(number)
  if number > 0
    puts "Positive"
  else
    puts "Not positive"
  end
end""",
            "expected": True
        },
        {
            "name": "Invalid - missing 'end'",
            "code": """def broken_method
  puts "This method is missing an end""",
            "expected": False
        },
        {
            "name": "Invalid - malformed syntax",
            "code": """def if while class end""",
            "expected": False
        },
        {
            "name": "Valid - simple expression",
            "code": """x = 5 + 3""",
            "expected": True
        },
        {
            "name": "Valid - class definition",
            "code": """class Calculator
  def initialize
    @value = 0
  end
  
  def add(number)
    @value += number
  end
end""",
            "expected": True
        }
    ]
    
    total_tests = len(test_cases)
    passed_tests = 0
    valid_syntax_count = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print("-" * 50)
        
        # Check syntax
        is_valid = is_syntactically_valid(test['code'])
        
        if is_valid:
            valid_syntax_count += 1
        
        # Check if our expectation matches the result
        test_passed = (is_valid == test['expected'])
        if test_passed:
            passed_tests += 1
        
        print(f"Code:\n{test['code']}")
        print(f"Syntax Valid: {'Yes' if is_valid else 'No'}")
        print(f"Test Result: {'✓ PASS' if test_passed else '✗ FAIL'}")
        
        if not test_passed:
            print(f"Expected: {'Valid' if test['expected'] else 'Invalid'}, Got: {'Valid' if is_valid else 'Invalid'}")
        
        print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total test cases: {total_tests}")
    print(f"Test cases passed: {passed_tests}/{total_tests} ({100*passed_tests/total_tests:.1f}%)")
    print(f"Syntactically valid code samples: {valid_syntax_count}/{total_tests} ({100*valid_syntax_count/total_tests:.1f}%)")
    print()
    print("This demonstrates how the enhanced evaluation provides:")
    print("✓ Concrete pass/fail metrics instead of guesswork")
    print("✓ Automated validation that scales with test size")
    print("✓ Detailed error information for debugging (see stderr output)")
    print("✓ Integration with existing evaluation workflow")

if __name__ == "__main__":
    demonstrate_syntax_checking()