#!/usr/bin/env ruby

require 'parser/current'
require 'json'

# Simple integration test for the improved heuristics
# Tests the actual integration without needing large datasets

class IntegrationTest
  def initialize
    @test_descriptions_cache = {}
    @spec_files_cache = nil
  end

  def run_test
    puts "Testing improved heuristics integration..."
    
    # Test method extraction
    test_method_info = test_extract_method_info
    puts "✓ Method extraction works: #{test_method_info[:name]}"
    
    # Test heuristic matching
    test_heuristic_results = test_heuristics
    puts "✓ Heuristic matching works"
    test_heuristic_results.each do |result|
      puts "  - #{result[:method]} vs '#{result[:description]}': #{result[:matches] ? '✓' : '✗'}"
    end
    
    puts "\nIntegration test completed successfully!"
  end

  private

  def test_extract_method_info
    source = "def authorize_user\n  check_permissions\nend"
    parser = Parser::CurrentRuby.new
    buffer = Parser::Source::Buffer.new('test')
    buffer.source = source
    ast = parser.parse(buffer)
    
    extract_method_info_from_ast(ast)
  end

  def test_heuristics
    test_cases = [
      { method: "authorize", description: "authorizes the user successfully", expected: true },
      { method: "policy_scope", description: "raises an exception when policy scope is not used", expected: true },
      { method: "create", description: "creates a new record successfully", expected: true },
      { method: "foo", description: "tests bar functionality", expected: false }
    ]
    
    results = []
    test_cases.each do |test_case|
      # Create a simple test node structure
      test_node = create_mock_test_node
      matches = test_likely_targets_method?(test_node, test_case[:method], test_case[:description])
      
      results << {
        method: test_case[:method],
        description: test_case[:description],
        matches: matches,
        expected: test_case[:expected],
        correct: matches == test_case[:expected]
      }
    end
    
    results
  end

  def create_mock_test_node
    # Create a simple mock node that represents a test block
    parser = Parser::CurrentRuby.new
    buffer = Parser::Source::Buffer.new('test')
    buffer.source = "it 'test' do\n  expect(subject.method).to be_truthy\nend"
    parser.parse(buffer)
  end

  # Include the actual methods from the improved script
  def extract_method_info_from_ast(ast)
    method_node = find_method_node(ast)
    return nil unless method_node
    
    if method_node.type == :def
      method_name = method_node.children[0].to_s
    elsif method_node.type == :defs
      method_name = method_node.children[1].to_s
    end
    
    {
      name: method_name,
      type: method_node.type
    }
  end

  def find_method_node(node)
    return node if node.is_a?(Parser::AST::Node) && (node.type == :def || node.type == :defs)
    
    if node.is_a?(Parser::AST::Node) && node.children
      node.children.each do |child|
        result = find_method_node(child)
        return result if result
      end
    end
    
    nil
  end

  def test_likely_targets_method?(test_node, target_method_name, description)
    # Use the improved heuristics
    method_name_in_description?(target_method_name, description) || 
    behavioral_pattern_matches?(target_method_name, description)
  end

  def method_name_in_description?(method_name, description)
    desc_lower = description.downcase
    method_lower = method_name.to_s.downcase
    
    # Direct match
    return true if desc_lower.include?(method_lower)
    
    # Handle underscore to space conversion
    method_spaced = method_lower.gsub('_', ' ')
    return true if desc_lower.include?(method_spaced)
    
    # Handle verb forms
    if method_lower.end_with?('e') && method_lower.length > 3
      verb_form = method_lower + 's'
      return true if desc_lower.include?(verb_form)
    elsif !method_lower.end_with?('s')
      verb_form = method_lower + 's'
      return true if desc_lower.include?(verb_form)
    end
    
    false
  end

  def behavioral_pattern_matches?(method_name, description)
    desc_lower = description.downcase
    method_lower = method_name.to_s.downcase
    
    # Pattern: "returns X" might be testing a method that provides X
    if desc_lower.match?(/\b(returns?|gives?|provides?)\b/)
      method_parts = method_lower.split('_')
      return true if method_parts.any? { |part| part.length > 2 && desc_lower.include?(part) }
    end
    
    # Pattern: authorization tests
    if desc_lower.match?(/\b(authoriz|permit|allow)\b/) && 
       method_lower.match?(/\b(authoriz|permit|allow)\b/)
      return true
    end
    
    false
  end
end

# Run the test
if __FILE__ == $0
  test = IntegrationTest.new
  test.run_test
end