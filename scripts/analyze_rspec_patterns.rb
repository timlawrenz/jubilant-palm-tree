#!/usr/bin/env ruby

require 'parser/current'
require 'json'
require 'set'

# RSpec Analysis Script to improve matching heuristics
# Analyzes real RSpec files to find patterns and test heuristic accuracy

class RSpecPatternAnalyzer
  def initialize
    @test_cases = []
    @rspec_methods_found = Set.new
    @all_method_calls = []
  end

  def analyze_repository(repo_path)
    puts "Analyzing repository: #{repo_path}"
    
    # Find all spec files
    spec_files = Dir.glob("#{repo_path}/**/*_spec.rb")
    puts "Found #{spec_files.length} spec files"

    spec_files.each do |spec_file|
      analyze_spec_file(spec_file)
    end

    generate_report
  end

  private

  def analyze_spec_file(spec_file)
    return unless File.exist?(spec_file)

    begin
      source = File.read(spec_file)
      parser = Parser::CurrentRuby.new
      buffer = Parser::Source::Buffer.new(spec_file)
      buffer.source = source
      ast = parser.parse(buffer)

      analyze_ast_for_patterns(ast, spec_file)
    rescue => e
      puts "Error parsing #{spec_file}: #{e.message}"
    end
  end

  def analyze_ast_for_patterns(node, file_path)
    return unless node.is_a?(Parser::AST::Node)

    # Look for 'it' blocks and other RSpec DSL elements
    case node.type
    when :block
      analyze_block_node(node, file_path)
    when :send
      analyze_send_node(node, file_path)
    end

    # Recursively analyze child nodes
    node.children.each do |child|
      analyze_ast_for_patterns(child, file_path)
    end
  end

  def analyze_block_node(node, file_path)
    send_node = node.children[0]
    return unless send_node.is_a?(Parser::AST::Node) && send_node.type == :send

    method_name = send_node.children[1]
    
    # Collect RSpec DSL methods
    if method_name.is_a?(Symbol)
      @rspec_methods_found << method_name
      
      # For 'it' and 'specify' blocks, extract test descriptions
      if [:it, :specify].include?(method_name) && send_node.children[2]
        description_node = send_node.children[2]
        if description_node.type == :str
          description = description_node.children[0]
          test_body = node.children[2] # The block body
          
          method_calls = extract_method_calls_from_node(test_body)
          @all_method_calls.concat(method_calls)
          
          @test_cases << {
            file: file_path,
            description: description,
            method_calls: method_calls
          }
        end
      end
    end
  end

  def analyze_send_node(node, file_path)
    method_name = node.children[1]
    
    # Collect method calls that might be RSpec DSL
    if method_name && method_name.is_a?(Symbol)
      @rspec_methods_found << method_name
    end
  end

  def extract_method_calls_from_node(node)
    calls = []
    return calls unless node.is_a?(Parser::AST::Node)

    if node.type == :send && node.children[1]
      calls << node.children[1]
    end

    node.children.each do |child|
      calls.concat(extract_method_calls_from_node(child))
    end

    calls.uniq
  end

  def generate_report
    puts "\n" + "="*60
    puts "RSPEC PATTERN ANALYSIS REPORT"
    puts "="*60

    puts "\n1. RSpec DSL Methods Found (#{@rspec_methods_found.size} unique):"
    @rspec_methods_found.sort.each_slice(8) do |slice|
      puts "   #{slice.join(', ')}"
    end

    puts "\n2. Test Cases Analysis:"
    puts "   Total test cases found: #{@test_cases.size}"
    
    analyze_description_patterns
    analyze_method_calls
    generate_recommendations
  end

  def analyze_description_patterns
    puts "\n3. Test Description Pattern Analysis:"
    
    # Pattern analysis
    patterns = analyze_test_descriptions
    
    patterns.each do |pattern_name, cases|
      next if cases.empty?
      
      puts "   #{pattern_name}: #{cases.size} cases"
      cases.first(2).each do |test_case|
        puts "     • \"#{test_case[:description]}\""
      end
      puts "     ..." if cases.size > 2
      puts
    end
  end

  def analyze_test_descriptions
    patterns = {
      "Method name in description" => [],
      "Returns/gives pattern" => [],
      "When/if condition pattern" => [],
      "Should/can behavior pattern" => [],
      "Negative tests (not/does not)" => [],
      "Error handling tests" => [],
      "Other descriptive tests" => []
    }

    @test_cases.each do |test_case|
      desc = test_case[:description].downcase
      
      # Check if any method calls are mentioned in description
      mentioned_methods = test_case[:method_calls].select do |method|
        next false if method.to_s.length < 3 # Skip very short method names
        desc.include?(method.to_s.downcase.gsub('_', ' ')) || 
        desc.include?(method.to_s.downcase)
      end
      
      if mentioned_methods.any?
        patterns["Method name in description"] << test_case
      elsif desc.match?(/\b(returns?|gives?|provides?)\b/)
        patterns["Returns/gives pattern"] << test_case
      elsif desc.match?(/\b(when|if|given|with)\b/)
        patterns["When/if condition pattern"] << test_case
      elsif desc.match?(/\b(should|can|is able to|behaves?|acts?)\b/)
        patterns["Should/can behavior pattern"] << test_case
      elsif desc.match?(/\b(not|does not|doesn't|cannot|can't|fails?|raises?|errors?)\b/)
        patterns["Negative tests (not/does not)"] << test_case
      elsif desc.match?(/\b(error|exception|rescue|fail)\b/)
        patterns["Error handling tests"] << test_case
      else
        patterns["Other descriptive tests"] << test_case
      end
    end

    patterns
  end

  def analyze_method_calls
    puts "4. Method Call Frequency Analysis:"
    
    method_frequency = @all_method_calls.each_with_object(Hash.new(0)) { |method, counts| counts[method] += 1 }
    
    # Find the most common method calls
    common_methods = method_frequency.sort_by { |_, count| -count }.first(25)
    
    puts "   Top 25 method calls in test bodies:"
    common_methods.each_with_index do |(method, count), index|
      puts "   #{sprintf('%2d', index + 1)}. #{method}: #{count} times"
    end
  end

  def generate_recommendations
    puts "\n5. Recommendations for Improved Heuristics:"
    
    # Current hardcoded RSpec methods from the original script
    current_rspec_methods = %w[expect to not_to eq be be_a be_an be_nil be_empty be_truthy be_falsy 
                              have have_key have_attributes include match raise_error change 
                              receive allow allow_any_instance_of and_return and_raise
                              before after let let! subject described_class instance_double double
                              stub_const assign it describe context specify shared_examples
                              shared_context within].map(&:to_sym)

    current_set = Set.new(current_rspec_methods)
    found_set = @rspec_methods_found

    missing_methods = found_set - current_set
    
    puts "\n   A. Missing RSpec DSL Methods to Add to Filter:"
    if missing_methods.any?
      missing_methods.sort.each_slice(6) do |slice|
        puts "      #{slice.join(', ')}"
      end
    else
      puts "      No significant new methods found (current list is comprehensive)"
    end

    puts "\n   B. Heuristic Improvements Based on Patterns:"
    puts "      1. Enhance method name matching:"
    puts "         - Add fuzzy matching (e.g., 'authorize' matches 'authorization')"
    puts "         - Handle underscore/space conversion (e.g., 'user_name' matches 'user name')"
    puts "         - Consider verb forms (e.g., 'validates' matches 'validate')"
    
    puts "\n      2. Add semantic pattern matching:"
    puts "         - Look for 'returns X' patterns and match to methods that return X"
    puts "         - Detect 'when X' patterns that describe method behavior conditions"
    puts "         - Handle negative tests ('does not Y') for methods related to Y"
    
    puts "\n      3. Improve context awareness:"
    puts "         - Consider describe/context block names for additional method hints"
    puts "         - Analyze subject and described_class patterns more thoroughly"
    puts "         - Look for let/let! variable names that might indicate methods"

    puts "\n   C. Additional Filtering Needed:"
    # Identify methods that appear frequently but shouldn't be filtered
    frequent_non_rspec = @all_method_calls
      .select { |m| @all_method_calls.count(m) >= 3 }
      .reject { |m| current_set.include?(m) }
      .uniq
      .sort
    
    if frequent_non_rspec.any?
      puts "      Consider adding these frequent methods to RSpec filter:"
      frequent_non_rspec.each_slice(6) do |slice|
        puts "         #{slice.join(', ')}"
      end
    end
  end
end

# Main execution
if __FILE__ == $0
  analyzer = RSpecPatternAnalyzer.new
  
  # Analyze the cloned repositories
  repos_to_analyze = [
    './repos/pundit',
    './repos/rspec-core',
    './repos/rubocop'
  ]
  
  repos_to_analyze.each do |repo_path|
    if Dir.exist?(repo_path)
      analyzer.analyze_repository(repo_path)
      puts "\n" + "="*60 + "\n"
    else
      puts "Repository not found: #{repo_path}"
    end
  end
end