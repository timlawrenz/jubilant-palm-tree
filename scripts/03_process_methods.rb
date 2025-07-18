#!/usr/bin/env ruby

require 'flog'
require 'parser/current'
require 'json'
require 'fileutils'

# Method Processing Script
# Reads methods.json and generates complexity scores and AST representations
# Outputs processed_methods.jsonl with original data plus complexity_score and ast_json fields

class MethodProcessor
  def initialize(input_file, output_file)
    @input_file = input_file
    @output_file = output_file
    @processed_count = 0
    @error_count = 0
  end

  def process_methods
    puts "Starting method processing from: #{@input_file}"
    
    # Ensure output directory exists
    output_dir = File.dirname(@output_file)
    FileUtils.mkdir_p(output_dir) unless Dir.exist?(output_dir)
    
    # Read input data
    unless File.exist?(@input_file)
      puts "Error: Input file '#{@input_file}' does not exist."
      exit 1
    end
    
    methods_data = JSON.parse(File.read(@input_file))
    puts "Found #{methods_data.length} methods to process"
    
    # Process methods and write JSONL output
    File.open(@output_file, 'w') do |output|
      methods_data.each_with_index do |method_data, index|
        puts "Processing method #{index + 1}/#{methods_data.length}" if (index + 1) % 100 == 0 || methods_data.length <= 10
        
        processed_method = process_single_method(method_data)
        if processed_method
          output.puts(JSON.generate(processed_method))
          @processed_count += 1
        else
          @error_count += 1
        end
      end
    end
    
    puts "Processing complete."
    puts "Successfully processed: #{@processed_count} methods"
    puts "Errors encountered: #{@error_count} methods"
    puts "Output written to: #{@output_file}"
  end

  private

  def process_single_method(method_data)
    begin
      # Extract raw source code
      raw_source = method_data['raw_source']
      
      # Calculate complexity score using flog
      complexity_score = calculate_complexity(raw_source)
      
      # Generate AST using parser
      ast_json = generate_ast_json(raw_source)
      
      # Create processed method with original data plus new fields
      processed_method = method_data.dup
      processed_method['complexity_score'] = complexity_score
      processed_method['ast_json'] = ast_json
      
      return processed_method
      
    rescue => e
      puts "Error processing method from #{method_data['file_path']}:#{method_data['start_line']}: #{e.message}"
      # Return nil to indicate failure
      return nil
    end
  end

  def calculate_complexity(raw_source)
    flogger = Flog.new
    flogger.flog_ruby(raw_source)
    
    # Get complexity score for the method
    if flogger.calls.any?
      # Sum all complexity scores for this method
      method_name = flogger.calls.keys.first
      method_calls = flogger.calls[method_name]
      complexity = method_calls.values.sum
    else
      # If no complexity detected, default to 0
      complexity = 0.0
    end
    
    complexity.round(2)
  end

  def generate_ast_json(raw_source)
    parser = Parser::CurrentRuby.new
    buffer = Parser::Source::Buffer.new('(method)')
    buffer.source = raw_source
    ast = parser.parse(buffer)
    
    # Convert AST to hash structure for JSON serialization
    ast_hash = ast_to_hash(ast)
    
    # Return as JSON string
    JSON.generate(ast_hash)
  end

  def ast_to_hash(node)
    if node.is_a?(Parser::AST::Node)
      {
        type: node.type,
        children: node.children.map { |child| ast_to_hash(child) }
      }
    else
      node
    end
  end
end

# Main execution
if __FILE__ == $0
  # Configuration
  input_file = './output/methods.json'
  output_file = './output/processed_methods.jsonl'
  
  # Create and run the processor
  processor = MethodProcessor.new(input_file, output_file)
  processor.process_methods
end