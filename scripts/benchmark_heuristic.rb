#!/usr/bin/env ruby

require 'json'

# Heuristic Benchmark Script
# Calculates complexity prediction error using simple keyword counting heuristic
# Reads test.jsonl file and compares keyword-based predictions to true complexity scores

class HeuristicBenchmark
  def initialize(test_file)
    @test_file = test_file
    @predictions = []
    @true_scores = []
    @processed_count = 0
    @error_count = 0
    
    # Define complexity-indicating keywords
    @complexity_keywords = [
      # Control flow
      'if', 'elsif', 'else', 'unless', 'case', 'when',
      'while', 'until', 'for', 'loop', 'break', 'next', 'return',
      
      # Iterators and blocks
      'each', 'map', 'select', 'collect', 'times', 'upto', 'downto',
      'find', 'detect', 'reject', 'inject', 'reduce',
      
      # Exception handling
      'rescue', 'ensure', 'raise', 'begin',
      
      # Block and proc operations
      'yield', 'lambda', 'proc', 'block_given?',
      
      # Method definitions (nested complexity)
      'def ', 'class ', 'module ',
      
      # Other complexity indicators
      'and', 'or', '&&', '||', '?', ':',
      'respond_to?', 'send', 'eval', 'assert'
    ]
  end

  def run_benchmark
    puts "Starting heuristic benchmark on: #{@test_file}"
    
    unless File.exist?(@test_file)
      puts "Error: Test file '#{@test_file}' does not exist."
      exit 1
    end

    File.foreach(@test_file) do |line|
      begin
        entry = JSON.parse(line.strip)
        process_entry(entry)
      rescue JSON::ParserError => e
        puts "Warning: Skipping malformed JSON line: #{e.message}"
        @error_count += 1
        next
      rescue => e
        puts "Warning: Error processing entry: #{e.message}"
        @error_count += 1
        next
      end
    end

    calculate_and_report_mae
  end

  private

  def process_entry(entry)
    raw_source = entry['raw_source']
    true_complexity = entry['complexity_score']
    
    if raw_source.nil? || true_complexity.nil?
      puts "Warning: Missing raw_source or complexity_score in entry"
      @error_count += 1
      return
    end

    # Calculate heuristic prediction
    heuristic_prediction = calculate_heuristic_complexity(raw_source)
    
    @predictions << heuristic_prediction
    @true_scores << true_complexity.to_f
    @processed_count += 1
    
    if @processed_count <= 5
      puts "Sample #{@processed_count}: True=#{true_complexity}, Predicted=#{heuristic_prediction}, Error=#{(heuristic_prediction - true_complexity).abs.round(2)}"
    end
  end

  def calculate_heuristic_complexity(source_code)
    return 1.0 if source_code.nil? || source_code.empty?
    
    keyword_count = 0
    
    # Count each complexity keyword occurrence
    @complexity_keywords.each do |keyword|
      # Use word boundaries for exact matches, except for operators
      if keyword.match?(/[a-zA-Z_]/)
        # Use word boundaries for alphabetic keywords
        keyword_count += source_code.scan(/\b#{Regexp.escape(keyword)}\b/).length
      else
        # Simple string matching for operators and symbols
        keyword_count += source_code.scan(Regexp.escape(keyword)).length
      end
    end
    
    # Apply heuristic scaling
    # Base complexity of 2.0, plus 1.5 per keyword
    # These values may need tuning based on results
    base_complexity = 2.0
    keyword_factor = 1.5
    
    predicted_complexity = base_complexity + (keyword_count * keyword_factor)
    predicted_complexity.round(2)
  end

  def calculate_and_report_mae
    if @predictions.empty?
      puts "Error: No valid predictions generated"
      exit 1
    end

    # Calculate Mean Absolute Error
    absolute_errors = @predictions.zip(@true_scores).map do |pred, true_val|
      (pred - true_val).abs
    end
    
    mae = absolute_errors.sum / absolute_errors.length
    
    # Calculate additional statistics
    min_true = @true_scores.min
    max_true = @true_scores.max
    mean_true = @true_scores.sum / @true_scores.length
    
    min_pred = @predictions.min
    max_pred = @predictions.max
    mean_pred = @predictions.sum / @predictions.length

    puts "\n" + "="*50
    puts "HEURISTIC BENCHMARK RESULTS"
    puts "="*50
    puts "Processed entries: #{@processed_count}"
    puts "Error entries: #{@error_count}"
    puts ""
    puts "True Complexity Statistics:"
    puts "  Min: #{min_true.round(2)}"
    puts "  Max: #{max_true.round(2)}"
    puts "  Mean: #{mean_true.round(2)}"
    puts ""
    puts "Predicted Complexity Statistics:"
    puts "  Min: #{min_pred.round(2)}"
    puts "  Max: #{max_pred.round(2)}"
    puts "  Mean: #{mean_pred.round(2)}"
    puts ""
    puts "FINAL MEAN ABSOLUTE ERROR (MAE): #{mae.round(4)}"
    puts ""
    puts "This is the benchmark score that the GNN needs to beat."
    puts "="*50
  end
end

# Main execution
if __FILE__ == $0
  test_file = ARGV[0] || 'dataset/test.jsonl'
  
  unless File.exist?(test_file)
    puts "Usage: ruby #{$0} [test_file]"
    puts "Default test file: dataset/test.jsonl"
    exit 1
  end
  
  benchmark = HeuristicBenchmark.new(test_file)
  benchmark.run_benchmark
end