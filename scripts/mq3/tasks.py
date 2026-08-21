"""MQ3 curated executable task suite.

Classic small algorithms authored as real Ruby methods, each with:
  - method source (real Ruby, runs on the box)
  - input→output test cases (executable reference)
  - ops restricted to the MVP interpreter's supported set (+ - * % < <= > >= ==
    != print and literals) so the graph arm can actually compute outputs.

Each task gets compressed through the real pipeline (Ruby parser -> AST ->
compress_ast) and scored by BOTH arms:
  - LLM arm: same method signature style prompt -> writes Ruby -> run real Ruby
  - NUM arm: method name + literal pool -> Legislative LLM -> BoM -> AR
    Executive -> ConstraintSolver -> GraphInterpreter -> compare output
"""

TASKS = [
    {
        "name": "fib",
        "method": "def fib(n)\n  return n if n < 2\n  fib(n - 1) + fib(n - 2)\nend\n",
        "cases": [(0, 0), (1, 1), (6, 8), (10, 55)],
        "prompt": "Return the nth Fibonacci number. fib(0)=0, fib(1)=1.",
    },
    {
        "name": "factorial",
        "method": "def factorial(n)\n  return 1 if n <= 1\n  n * factorial(n - 1)\nend\n",
        "cases": [(0, 1), (1, 1), (5, 120), (7, 5040)],
        "prompt": "Return the factorial of n (n!). factorial(0)=1.",
    },
    {
        "name": "gcd",
        "method": "def gcd(a, b)\n  return a if b == 0\n  gcd(b, a % b)\nend\n",
        "cases": [(48, 18, 6), (17, 5, 1), (0, 9, 9), (54, 24, 6)],
        "prompt": "Return the greatest common divisor of a and b (Euclid's algorithm).",
    },
    {
        "name": "sum_range",
        "method": "def sum_range(n)\n  total = 0\n  i = 1\n  while i <= n\n    total = total + i\n    i = i + 1\n  end\n  total\nend\n",
        "cases": [(0, 0), (3, 6), (10, 55), (100, 5050)],
        "prompt": "Return the sum of integers from 1 to n.",
    },
    {
        "name": "is_palindrome",
        "method": "def is_palindrome(s)\n  s == s.reverse\nend\n",
        "cases": [("racecar", True), ("hello", False), ("a", True), ("abba", True)],
        "prompt": "Return true if the string s is a palindrome, false otherwise.",
    },
    {
        "name": "count_even",
        "method": "def count_even(arr)\n  count = 0\n  arr.each do |x|\n    count = count + 1 if x % 2 == 0\n  end\n  count\nend\n",
        "cases": [([1, 2, 3, 4], 2), ([], 0), ([2, 4, 6], 3), ([1, 3], 0)],
        "prompt": "Return the number of even integers in the array arr.",
    },
    {
        "name": "last_digit",
        "method": "def last_digit(n)\n  n % 10\nend\n",
        "cases": [(123, 3), (0, 0), (10, 0), (987654321, 1)],
        "prompt": "Return the last decimal digit of the integer n.",
    },
    {
        "name": "sum_a_b",
        "method": "def sum(a, b)\n  a + b\nend\n",
        "cases": [(2, 3, 5), (-1, 1, 0), (100, 200, 300)],
        "prompt": "Return the sum of a and b.",
    },
    {
        "name": "pos_or_neg",
        "method": "def pos_or_neg(x)\n  return 1 if x > 0\n  return -1 if x < 0\n  0\nend\n",
        "cases": [(5, 1), (-3, -1), (0, 0), (999, 1)],
        "prompt": "Return 1 if x is positive, -1 if negative, 0 if zero.",
    },
    {
        "name": "max2",
        "method": "def max2(a, b)\n  a > b ? a : b\nend\n",
        "cases": [(3, 7, 7), (10, 2, 10), (-1, -5, -1)],
        "prompt": "Return the larger of a and b.",
    },
    {
        "name": "min2",
        "method": "def min2(a, b)\n  a < b ? a : b\nend\n",
        "cases": [(3, 7, 3), (10, 2, 2), (-1, -5, -5)],
        "prompt": "Return the smaller of a and b.",
    },
    {
        "name": "diff2",
        "method": "def diff2(a, b)\n  a - b\nend\n",
        "cases": [(10, 3, 7), (3, 10, -7), (5, 5, 0)],
        "prompt": "Return a minus b.",
    },
    {
        "name": "mul2",
        "method": "def mul2(a, b)\n  a * b\nend\n",
        "cases": [(3, 4, 12), (0, 9, 0), (7, 6, 42)],
        "prompt": "Return the product of a and b.",
    },
    {
        "name": "div2",
        "method": "def div2(a, b)\n  a / b\nend\n",
        "cases": [(10, 2, 5), (7, 2, 3), (0, 5, 0)],
        "prompt": "Return integer division of a by b.",
    },
    {
        "name": "is_even",
        "method": "def is_even(n)\n  n % 2 == 0\nend\n",
        "cases": [(4, True), (7, False), (0, True), (11, False)],
        "prompt": "Return true if n is even, false otherwise.",
    },
    {
        "name": "is_positive",
        "method": "def is_positive(x)\n  x > 0\nend\n",
        "cases": [(5, True), (-5, False), (0, False)],
        "prompt": "Return true if x is strictly positive.",
    },
    {
        "name": "gt",
        "method": "def gt(a, b)\n  a > b\nend\n",
        "cases": [(5, 3, True), (3, 5, False), (5, 5, False)],
        "prompt": "Return true if a is greater than b.",
    },
    {
        "name": "add_mod",
        "method": "def add_mod(a, b, m)\n  (a + b) % m\nend\n",
        "cases": [(7, 5, 10, 2), (6, 4, 5, 0), (100, 25, 7, 6)],
        "prompt": "Return (a + b) modulo m.",
    },
    {
        "name": "double_minus_one",
        "method": "def double_minus_one(n)\n  n * 2 - 1\nend\n",
        "cases": [(1, 1), (5, 9), (10, 19)],
        "prompt": "Return 2n - 1.",
    },
    {
        "name": "is_odd",
        "method": "def is_odd(n)\n  n % 2 == 1\nend\n",
        "cases": [(3, True), (4, False), (0, False), (15, True)],
        "prompt": "Return true if n is odd, false otherwise.",
    },
]
