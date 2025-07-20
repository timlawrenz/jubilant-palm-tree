#!/usr/bin/env ruby

require 'parser/current'
require 'json'

# Test script to validate improved RSpec heuristics
# Creates test cases and checks if the improved heuristics work better

class HeuristicTester
  def initialize
    @test_cases = []
    setup_test_cases
  end

  def run_tests
    puts "Testing improved RSpec heuristics..."
    puts "="*50

    correct_old = 0
    correct_new = 0
    total = @test_cases.length

    @test_cases.each_with_index do |test_case, index|
      old_result = old_heuristic(test_case[:method], test_case[:description])
      new_result = new_heuristic(test_case[:method], test_case[:description])
      expected = test_case[:should_match]

      correct_old += 1 if old_result == expected
      correct_new += 1 if new_result == expected

      status_old = old_result == expected ? "✓" : "✗"
      status_new = new_result == expected ? "✓" : "✗"

      puts "#{index + 1}. Method: #{test_case[:method]}"
      puts "   Description: \"#{test_case[:description]}\""
      puts "   Expected: #{expected}, Old: #{old_result} #{status_old}, New: #{new_result} #{status_new}"
      puts "   Improvement: #{new_result == expected && old_result != expected ? 'YES' : 'NO'}"
      puts
    end

    puts "="*50
    puts "RESULTS:"
    puts "Old heuristic accuracy: #{correct_old}/#{total} (#{(correct_old.to_f/total*100).round(1)}%)"
    puts "New heuristic accuracy: #{correct_new}/#{total} (#{(correct_new.to_f/total*100).round(1)}%)"
    puts "Improvement: +#{correct_new - correct_old} correct matches"
  end

  private

  def setup_test_cases
    # Test cases from real RSpec files plus edge cases
    @test_cases = [
      # Direct method name matches
      { method: "authorize", description: "infers the policy and authorizes based on it", should_match: true },
      { method: "authorize", description: "returns the record on successful authorization", should_match: true },
      { method: "inspect", description: "prints a human readable description when inspected", should_match: true },
      
      # Underscore to space conversion
      { method: "policy_scope", description: "raises an exception when policy scope is not used", should_match: true },
      { method: "user_name", description: "validates user name format", should_match: true },
      
      # Verb forms
      { method: "create", description: "creates a new record successfully", should_match: true },
      { method: "validate", description: "validates the input correctly", should_match: true },
      { method: "calculate", description: "calculates the total price", should_match: true },
      { method: "apply", description: "applies the discount correctly", should_match: true },
      
      # Progressive forms
      { method: "run", description: "is running the process", should_match: true },
      { method: "get", description: "getting the user data", should_match: true },
      
      # Behavioral patterns
      { method: "total_price", description: "returns the correct total", should_match: true },
      { method: "user_count", description: "gives the number of users", should_match: true },
      { method: "authorize", description: "when user is admin", should_match: true },
      
      # Negative cases that shouldn't match
      { method: "create", description: "raises an error for invalid input", should_match: false },
      { method: "delete", description: "validates the user permissions", should_match: false },
      { method: "foo", description: "tests bar functionality", should_match: false },
      
      # Edge cases
      { method: "is_valid", description: "validates the object state", should_match: true },
      { method: "can_edit", description: "checks if user can edit", should_match: true },
    ]
  end

  def old_heuristic(method_name, description)
    # Original simple heuristic - just check if method name is in description
    description.downcase.include?(method_name.downcase)
  end

  def new_heuristic(method_name, description)
    method_name_in_description?(method_name, description) || 
    behavioral_pattern_matches?(method_name, description)
  end

  def method_name_in_description?(method_name, description)
    desc_lower = description.downcase
    method_lower = method_name.to_s.downcase
    
    # Direct match
    return true if desc_lower.include?(method_lower)
    
    # Handle underscore to space conversion
    method_spaced = method_lower.gsub('_', ' ')
    return true if desc_lower.include?(method_spaced)
    
    # Handle verb forms and common variations
    if method_lower.end_with?('e') && method_lower.length > 3
      # "create" -> "creates", "calculate" -> "calculates"
      verb_form = method_lower + 's'
      return true if desc_lower.include?(verb_form)
      
      # "create" -> "creating", "calculate" -> "calculating"
      ing_form = method_lower.chomp('e') + 'ing'
      return true if desc_lower.include?(ing_form)
    elsif method_lower.end_with?('y') && method_lower.length > 2
      # "apply" -> "applies"
      verb_form = method_lower.chomp('y') + 'ies'
      return true if desc_lower.include?(verb_form)
    elsif !method_lower.end_with?('s')
      # Most other cases: "get" -> "gets", "run" -> "runs"
      verb_form = method_lower + 's'
      return true if desc_lower.include?(verb_form)
      
      # Progressive form: "get" -> "getting", "run" -> "running"
      if method_lower.end_with?('t', 'n', 'p', 'm')
        ing_form = method_lower + method_lower[-1] + 'ing'
        return true if desc_lower.include?(ing_form)
      else
        ing_form = method_lower + 'ing'
        return true if desc_lower.include?(ing_form)
      end
    end
    
    # Handle common method name patterns
    # "user_name" -> "username" or "user name"
    if method_lower.include?('_')
      compact_form = method_lower.gsub('_', '')
      return true if desc_lower.include?(compact_form)
    end
    
    false
  end

  def behavioral_pattern_matches?(method_name, description)
    desc_lower = description.downcase
    method_lower = method_name.to_s.downcase
    
    # Pattern: "returns X" might be testing a method that provides X
    if desc_lower.match?(/\b(returns?|gives?|provides?)\b/)
      # Look for method name parts in the description
      method_parts = method_lower.split('_')
      return true if method_parts.any? { |part| part.length > 2 && desc_lower.include?(part) }
    end
    
    # Pattern: "when X" might be testing method behavior under condition X
    if desc_lower.match?(/\b(when|if|given)\b/)
      # Check if method name suggests the condition or action being tested
      method_parts = method_lower.split('_')
      return true if method_parts.any? { |part| part.length > 2 && desc_lower.include?(part) }
      
      # Handle authorization patterns specifically
      if method_lower.include?('authoriz') && desc_lower.match?(/\b(admin|user|permiss)\b/)
        return true
      end
    end
    
    # Pattern: "validates X" for validation methods
    if desc_lower.include?('validat') && method_lower.include?('valid')
      return true
    end
    
    # Pattern: authorization/permission tests
    if desc_lower.match?(/\b(authoriz|permit|allow|forbid|deny)\b/) && 
       method_lower.match?(/\b(authoriz|permit|allow|forbid|deny)\b/)
      return true
    end
    
    false
  end
end

# Run the test
if __FILE__ == $0
  tester = HeuristicTester.new
  tester.run_tests
end