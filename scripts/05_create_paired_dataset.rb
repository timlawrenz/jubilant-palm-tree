#!/usr/bin/env ruby

require 'parser/current'
require 'json'
require 'fileutils'
require 'digest'

# Paired Dataset Creation Script
# Reads processed_methods.jsonl and enriches it with natural language descriptions
# sourced from method names (transformed to sentences) and RDoc/YARD comments.
# Outputs ALL methods with method name descriptions, plus docstrings where available.

class PairedDatasetCreator
  def initialize(input_file, output_file)
    @input_file = input_file
    @output_file = output_file
    @processed_count = 0
    @methods_successfully_paired = 0
    @methods_failed_to_process = 0
  end

  def create_paired_dataset
    puts "Starting paired dataset creation from: #{@input_file}"
    
    # Ensure output directory exists
    output_dir = File.dirname(@output_file)
    FileUtils.mkdir_p(output_dir) unless Dir.exist?(output_dir)
    
    # Check if input file exists
    unless File.exist?(@input_file)
      puts "Error: Input file '#{@input_file}' does not exist."
      exit 1
    end
    
    # Process methods line by line (JSONL format)
    File.open(@output_file, 'w') do |output|
      File.foreach(@input_file) do |line|
        next if line.strip.empty?
        
        method_data = JSON.parse(line.strip)
        @processed_count += 1
        
        puts "Processing method #{@processed_count}" if @processed_count % 100 == 0
        
        paired_method = process_single_method(method_data)
        if paired_method
          output.puts(JSON.generate(paired_method))
          @methods_successfully_paired += 1
        else
          @methods_failed_to_process += 1
        end
      end
    end
    
    puts "Paired dataset creation complete."
    puts "Total methods processed: #{@processed_count}"
    puts "Methods successfully paired: #{@methods_successfully_paired}"
    puts "Methods failed to process: #{@methods_failed_to_process}"
    puts "Output written to: #{@output_file}"
  end

  private

  def process_single_method(method_data)
    begin
      raw_source = method_data['raw_source']
      
      # Parse the raw_source to extract method information and comments
      parser = Parser::CurrentRuby.new
      buffer = Parser::Source::Buffer.new('(method)')
      buffer.source = raw_source
      ast, comments = parser.parse_with_comments(buffer)
      
      # Extract method information from the AST
      method_info = extract_method_info_from_ast(ast)
      return nil unless method_info
      
      # Extract docstring from comments or inline comments (optional)
      docstring = extract_docstring_from_source(raw_source, comments)
      
      # Create the paired data entry (always create, docstring is optional)
      create_paired_entry(method_data, method_info, docstring)
      
    rescue => e
      puts "Error processing method from #{method_data['file_path']}:#{method_data['start_line']}: #{e.message}"
      return nil
    end
  end

  def extract_method_info_from_ast(ast)
    # Find the method definition node in the AST
    method_node = find_method_node(ast)
    return nil unless method_node
    
    if method_node.type == :def
      # Instance method: def method_name
      method_name = method_node.children[0].to_s
    elsif method_node.type == :defs
      # Class method: def self.method_name or def obj.method_name
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

  def extract_docstring_from_source(raw_source, comments)
    # Extract docstring from comments within the raw source
    return nil if comments.empty?
    
    # Get all comment texts and clean them
    comment_texts = comments.map { |comment| clean_comment_text(comment.text) }
                           .reject { |text| text.strip.empty? }
                           .reject { |text| text.strip == ':nodoc:' }  # Filter out pure :nodoc: comments
    
    return nil if comment_texts.empty?
    
    # Join all comments into a single docstring
    docstring = comment_texts.join(' ').strip
    
    # Return nil if the docstring is empty
    return nil if docstring.empty?
    
    docstring
  end

  def clean_comment_text(comment_text)
    # Remove leading # and whitespace, but preserve the content
    cleaned = comment_text.gsub(/^\s*#\s?/, '')
    
    # Remove common RDoc/YARD tags for now (keep it simple)
    cleaned = cleaned.gsub(/^\s*@\w+.*$/, '').strip
    
    # Don't completely remove :nodoc: and similar markers, just clean them
    # We'll filter them out later if they're the only content
    cleaned = cleaned.gsub(/^\s*:nodoc:\s*/, ':nodoc:').strip
    
    cleaned
  end

  def transform_method_name_to_description(method_name)
    # Convert snake_case method name to a sentence description
    return '' if method_name.nil? || method_name.strip.empty?
    
    # Split on underscores and convert to lowercase
    words = method_name.to_s.split('_').map(&:downcase)
    
    # Skip if no words or empty after processing
    return '' if words.empty? || words.all?(&:empty?)
    
    # For most methods, treat the first word as a verb and make it present tense
    # This handles common patterns like "calculate", "get", "set", "create", etc.
    first_word = words.first
    
    # Special cases for words that shouldn't be conjugated (past participles, adjectives, etc.)
    if %w[included inherited excluded loaded cached shared locked opened closed].include?(first_word)
      # For past participles used as method names, keep them as-is 
      result_words = words
    elsif first_word.end_with?('e') && !first_word.end_with?('ee') && !%w[are were].include?(first_word)
      # "create" → "creates", "calculate" → "calculates", but "see" → "sees"
      first_word = first_word + 's'
      result_words = [first_word] + words[1..-1]
    elsif first_word.end_with?('y') && first_word.length > 1 && !%w[by my].include?(first_word)
      # "apply" → "applies", but not "by" → "bys"
      first_word = first_word[0..-2] + 'ies'
      result_words = [first_word] + words[1..-1]
    elsif first_word.end_with?('s', 'x', 'z', 'ch', 'sh')
      # "process" → "processes", "fix" → "fixes"
      first_word = first_word + 'es'
      result_words = [first_word] + words[1..-1]
    elsif !first_word.end_with?('s')
      # Most other cases: "get" → "gets", "run" → "runs"
      first_word = first_word + 's'
      result_words = [first_word] + words[1..-1]
    else
      # Already ends with 's', keep as-is
      result_words = words
    end
    
    result_words.join(' ')
  end

  def create_paired_entry(original_data, method_info, docstring)
    # Generate a unique ID
    id = generate_unique_id(original_data, method_info)
    
    # Extract method source (same as raw_source)
    method_source = original_data['raw_source']
    
    # Create descriptions array - always include method_name description
    descriptions = []
    
    # Add method name description for all methods
    method_name_description = transform_method_name_to_description(method_info[:name])
    if !method_name_description.empty?
      descriptions << {
        "source" => "method_name",
        "text" => method_name_description
      }
    end
    
    # Add docstring description if available
    if docstring && !docstring.strip.empty?
      descriptions << {
        "source" => "docstring", 
        "text" => docstring
      }
    end
    
    {
      "id" => id,
      "repo_name" => original_data['repo_name'],
      "file_path" => original_data['file_path'],
      "method_name" => method_info[:name],
      "method_source" => method_source,
      "ast_json" => original_data['ast_json'],
      "descriptions" => descriptions
    }
  end

  def generate_unique_id(original_data, method_info)
    # Create a unique identifier based on repo, file, method name, and line
    identifier_string = "#{original_data['repo_name']}-#{original_data['file_path']}-#{method_info[:name]}-#{original_data['start_line']}"
    
    # Use MD5 hash to create a shorter, consistent ID
    Digest::MD5.hexdigest(identifier_string)
  end
end

# Main execution
if __FILE__ == $0
  # Configuration
  input_file = './output/processed_methods.jsonl'
  output_file = './dataset/paired_data.jsonl'
  
  # Create and run the paired dataset creator
  creator = PairedDatasetCreator.new(input_file, output_file)
  creator.create_paired_dataset
end