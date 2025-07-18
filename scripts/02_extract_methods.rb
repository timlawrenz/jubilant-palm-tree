#!/usr/bin/env ruby

require 'parser/current'
require 'json'
require 'find'
require 'fileutils'

# Method Extraction Script
# Recursively scans Ruby files and extracts method definitions using the parser gem

class MethodExtractor
  def initialize(repos_dir, output_file)
    @repos_dir = repos_dir
    @output_file = output_file
    @methods = []
  end

  def extract_methods
    puts "Starting method extraction from: #{@repos_dir}"
    
    # Ensure output directory exists
    output_dir = File.dirname(@output_file)
    FileUtils.mkdir_p(output_dir) unless Dir.exist?(output_dir)
    
    # Find all Ruby files recursively
    ruby_files = find_ruby_files(@repos_dir)
    puts "Found #{ruby_files.length} Ruby files to process"
    
    # Process each Ruby file
    ruby_files.each_with_index do |file_path, index|
      puts "Processing file #{index + 1}/#{ruby_files.length}: #{file_path}" if (index + 1) % 100 == 0 || ruby_files.length <= 10
      process_file(file_path)
    end
    
    # Write results to JSON file
    write_output
    
    puts "Extraction complete. Found #{@methods.length} methods."
    puts "Output written to: #{@output_file}"
  end

  private

  def find_ruby_files(directory)
    ruby_files = []
    
    Find.find(directory) do |path|
      if File.file?(path) && path.end_with?('.rb')
        ruby_files << path
      end
    end
    
    ruby_files.sort
  end

  def process_file(file_path)
    begin
      # Read file content
      content = File.read(file_path, encoding: 'UTF-8')
      
      # Create a fresh parser for each file
      parser = Parser::CurrentRuby.new
      
      # Create source buffer for the parser
      buffer = Parser::Source::Buffer.new(file_path)
      buffer.source = content
      
      # Parse the Ruby code
      ast = parser.parse(buffer)
      
      # Extract repository name from path
      repo_name = extract_repo_name(file_path)
      
      # Extract methods from AST
      initial_count = @methods.length
      extract_methods_from_ast(ast, content, file_path, repo_name)
      new_methods = @methods.length - initial_count
      puts "  Found #{new_methods} methods" if new_methods > 0 || ruby_files.length <= 10
      
    rescue => e
      puts "Error processing #{file_path}: #{e.message}"
      puts e.backtrace.join("\n")
      # Continue processing other files even if one fails
    end
  end

  def extract_repo_name(file_path)
    # Extract repository name from file path
    # Assumes structure: ./repos/repo_name/...
    path_parts = file_path.split('/')
    repos_index = path_parts.index('repos')
    
    if repos_index && repos_index + 1 < path_parts.length
      path_parts[repos_index + 1]
    else
      'unknown'
    end
  end

  def extract_methods_from_ast(node, content, file_path, repo_name)
    return unless node
    
    # Check if this node is a method definition
    if node.type == :def || node.type == :defs
      method_info = extract_method_info(node, content, file_path, repo_name)
      @methods << method_info if method_info
    end
    
    # Recursively process child nodes
    if node.respond_to?(:children) && node.children
      node.children.each do |child|
        extract_methods_from_ast(child, content, file_path, repo_name) if child.is_a?(Parser::AST::Node)
      end
    end
  end

  def extract_method_info(node, content, file_path, repo_name)
    # Get source location information
    location = node.location
    return nil unless location && location.expression
    
    start_line = location.expression.begin.line
    end_line = location.expression.end.line
    
    # Extract raw source code for the method
    content_lines = content.lines
    raw_source = content_lines[start_line - 1, end_line - start_line + 1].join
    
    {
      repo_name: repo_name,
      file_path: file_path,
      start_line: start_line,
      raw_source: raw_source.strip
    }
  end

  def write_output
    File.open(@output_file, 'w') do |file|
      file.write(JSON.pretty_generate(@methods))
    end
  end
end

# Main execution
if __FILE__ == $0
  # Configuration
  repos_dir = './repos'
  output_file = './output/methods.json'
  
  # Check if repos directory exists
  unless Dir.exist?(repos_dir)
    puts "Error: Repository directory '#{repos_dir}' does not exist."
    puts "Please run the repository cloning script first."
    exit 1
  end
  
  # Create and run the extractor
  extractor = MethodExtractor.new(repos_dir, output_file)
  extractor.extract_methods
end