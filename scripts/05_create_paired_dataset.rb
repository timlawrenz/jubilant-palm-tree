#!/usr/bin/env ruby

require 'parser/current'
require 'json'
require 'fileutils'
require 'digest'
require 'rspec/core'
require 'stringio'
require_relative 'custom_rspec_formatter'

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
    @test_descriptions_cache = {}
    @spec_files_cache = nil
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

  def find_all_spec_files
    return @spec_files_cache if @spec_files_cache
    
    @spec_files_cache = []
    
    # Look for spec files in common locations relative to the repos
    base_dirs = ['./repos', '.']
    
    base_dirs.each do |base_dir|
      next unless Dir.exist?(base_dir)
      
      # Find all _spec.rb files
      Dir.glob("#{base_dir}/**/*_spec.rb").each do |spec_file|
        @spec_files_cache << spec_file if File.exist?(spec_file)
      end
    end
    
    @spec_files_cache
  end

  def find_test_descriptions_for_method(repo_name, file_path, method_name)
    cache_key = "#{repo_name}-#{file_path}-#{method_name}"
    return @test_descriptions_cache[cache_key] if @test_descriptions_cache.key?(cache_key)
    
    test_descriptions = []
    
    # Map implementation file to potential spec file paths
    potential_spec_files = map_implementation_to_spec_files(repo_name, file_path)
    
    potential_spec_files.each do |spec_file|
      next unless File.exist?(spec_file)
      
      begin
        descriptions = extract_test_descriptions_using_rspec_formatter(spec_file, method_name)
        
        # If RSpec formatter returns empty results but manual parsing finds results,
        # prefer the manual parsing results
        if descriptions.empty?
          puts "RSpec formatter returned no results for #{spec_file}, trying manual parsing as backup" if ENV['DEBUG']
          manual_descriptions = extract_test_descriptions_from_spec_file(spec_file, method_name)
          if manual_descriptions.length > 0
            puts "Manual parsing found #{manual_descriptions.length} descriptions, using those instead" if ENV['DEBUG']
            descriptions = manual_descriptions
          end
        end
        
        test_descriptions.concat(descriptions)
      rescue => e
        # Fall back to manual parsing if RSpec formatter fails
        puts "Warning: RSpec formatter failed for #{spec_file}, falling back to manual parsing: #{e.message}" if ENV['DEBUG']
        begin
          descriptions = extract_test_descriptions_from_spec_file(spec_file, method_name)
          test_descriptions.concat(descriptions)
        rescue => fallback_error
          puts "Warning: Could not parse spec file #{spec_file}: #{fallback_error.message}" if ENV['DEBUG']
        end
      end
    end
    
    @test_descriptions_cache[cache_key] = test_descriptions
    test_descriptions
  end

  def map_implementation_to_spec_files(repo_name, file_path)
    potential_specs = []
    
    # Handle different file path formats
    if file_path.include?('/repos/')
      # Remove ./repos/ prefix and repo name to get relative path
      relative_path = file_path.gsub(/^\.\/repos\/#{Regexp.escape(repo_name)}\//, '')
      base_repo_path = "./repos/#{repo_name}"
    else
      # For test cases where the path includes the full path structure
      if file_path.include?(repo_name)
        parts = file_path.split(repo_name)
        relative_path = parts.last.gsub(/^\//, '')
        # Find the base path up to the repo name
        base_repo_path = parts.first + repo_name
      else
        # Fallback: treat entire path as relative
        relative_path = file_path.gsub(/^\.\//, '')
        base_repo_path = '.'
      end
    end
    
    # Common mappings from implementation to spec files
    mappings = [
      # app/models/user.rb -> spec/models/user_spec.rb
      { from: /^app\//, to: 'spec/' },
      # lib/foo.rb -> spec/lib/foo_spec.rb or spec/foo_spec.rb
      { from: /^lib\//, to: 'spec/lib/' },
      { from: /^lib\//, to: 'spec/' },
      # Direct mapping: foo.rb -> foo_spec.rb
      { from: /^/, to: '' }
    ]
    
    mappings.each do |mapping|
      if relative_path =~ mapping[:from]
        spec_relative_path = relative_path.gsub(mapping[:from], mapping[:to])
        spec_relative_path = spec_relative_path.gsub(/\.rb$/, '_spec.rb')
        
        # Try with the base repo path and current directory
        [base_repo_path, '.'].each do |base_path|
          spec_full_path = File.join(base_path, spec_relative_path)
          potential_specs << spec_full_path
        end
      end
    end
    
    potential_specs.uniq
  end

  def determine_repo_context(spec_file)
    # Extract repository root and relative spec file path from the full spec file path
    # Examples:
    # ./repos/rspec-core/spec/rspec/core/runner_spec.rb -> ["./repos/rspec-core", "spec/rspec/core/runner_spec.rb"]
    # ./spec/models/user_spec.rb -> [".", "spec/models/user_spec.rb"]
    
    # Normalize the path
    normalized_path = File.expand_path(spec_file)
    current_path = File.expand_path('.')
    
    # Check if this is in a repos subdirectory
    if spec_file.include?('/repos/') && spec_file.start_with?('./')
      # Extract the repository name and construct the repository root
      parts = spec_file.split('/repos/')
      if parts.length == 2
        repos_base = parts[0] + '/repos'
        remaining_path = parts[1]
        
        # Find the repository name (first directory after repos/)
        path_segments = remaining_path.split('/')
        if path_segments.length >= 2
          repo_name = path_segments[0]
          repo_root = File.join(repos_base, repo_name)
          
          # The relative path is everything after the repo name
          relative_path = path_segments[1..-1].join('/')
          
          return [repo_root, relative_path]
        end
      end
    end
    
    # Fallback: try to find spec/ directory and use its parent as repo root
    spec_index = spec_file.rindex('/spec/')
    if spec_index
      repo_root = spec_file[0...spec_index]
      relative_path = spec_file[(spec_index + 1)..-1]  # Remove leading slash
      
      # Ensure the repo root exists and has a spec directory
      if Dir.exist?(repo_root) && Dir.exist?(File.join(repo_root, 'spec'))
        return [repo_root, relative_path]
      end
    end
    
    # Final fallback: use current directory
    return ['.', spec_file.gsub(/^\.\//, '')]
  end

  def extract_test_descriptions_using_rspec_formatter(spec_file, target_method_name)
    return [] unless File.exist?(spec_file)
    
    begin
      # Clear any previous RSpec configuration to avoid interference
      RSpec.clear_examples
      RSpec.reset
      
      # Create our custom formatter
      formatter_output = StringIO.new
      formatter = CustomRSpecFormatter.new(formatter_output)
      
      # Configure RSpec to use our formatter in dry-run mode
      RSpec.configure do |config|
        config.add_formatter(formatter)
        config.dry_run = true
        config.color = false
        config.profile_examples = false
        # Disable automatic spec_helper loading by clearing the spec_helper option
        config.spec_helper_file = nil if config.respond_to?(:spec_helper_file=)
      end
      
      # Determine the repository root directory and relative spec file path
      # to ensure RSpec runs from the correct context
      repo_root, relative_spec_file = determine_repo_context(spec_file)
      
      # Run RSpec on the spec file in dry-run mode from the repository root
      current_dir = Dir.pwd
      begin
        Dir.chdir(repo_root) if repo_root && Dir.exist?(repo_root)
        
        # Add additional safety by temporarily suppressing some RSpec configuration
        # that might require external dependencies
        exit_code = RSpec::Core::Runner.run([
          relative_spec_file, 
          '--dry-run',
          '--no-color',
          '--no-profile'
        ])
      rescue LoadError => load_error
        # If we get a LoadError, it means the spec file has dependencies we can't resolve
        # This is expected for many external repositories
        puts "Warning: Could not load dependencies for #{spec_file}: #{load_error.message}" if ENV['DEBUG']
        return []
      ensure
        Dir.chdir(current_dir)
      end
      
      # Extract relevant test descriptions from the collected examples
      relevant_descriptions = []
      
      formatter.collected_examples.each do |example_info|
        if test_likely_targets_method_from_rspec(example_info, target_method_name)
          # Use the full contextual description
          relevant_descriptions << example_info[:full_description]
        end
      end
      
      relevant_descriptions
      
    rescue => e
      puts "Error running RSpec formatter on #{spec_file}: #{e.message}" if ENV['DEBUG']
      []
    end
  end

  def test_likely_targets_method_from_rspec(example_info, target_method_name)
    description = example_info[:full_description]
    
    # Heuristic 1: Method name appears in the test description (improved fuzzy matching)
    return true if method_name_in_description?(target_method_name, description)
    
    # Heuristic 2: Check if described_class information provides context
    if example_info[:described_class] && example_info[:described_class].length > 0
      # If we have described_class info, this is likely a unit test
      # Check if the method name appears in any part of the description
      return true if method_name_in_description?(target_method_name, description)
    end
    
    # Heuristic 3: Pattern-based matching for behavioral descriptions
    return true if behavioral_pattern_matches?(target_method_name, description)
    
    # Heuristic 4: Context-aware matching
    # Check if any context descriptions relate to the method
    example_info[:context_descriptions].each do |context|
      return true if method_name_in_description?(target_method_name, context)
    end
    
    false
  end

  def extract_test_descriptions_from_spec_file(spec_file, target_method_name)
    
    source = File.read(spec_file)
    
    begin
      parser = Parser::CurrentRuby.new
      buffer = Parser::Source::Buffer.new(spec_file)
      buffer.source = source
      ast = parser.parse(buffer)
      
      test_descriptions = []
      extract_it_blocks_from_ast(ast, test_descriptions, target_method_name)
      test_descriptions
    rescue => e
      puts "Error parsing spec file #{spec_file}: #{e.message}" if ENV['DEBUG']
      []
    end
  end

  def extract_it_blocks_from_ast(node, test_descriptions, target_method_name)
    return unless node.is_a?(Parser::AST::Node)
    
    # Look for 'it' blocks: (block (send nil :it (str "description")) ...)
    if node.type == :block && 
       node.children[0].is_a?(Parser::AST::Node) &&
       node.children[0].type == :send &&
       node.children[0].children[1] == :it &&
       node.children[0].children[2] &&
       node.children[0].children[2].type == :str
      
      description = node.children[0].children[2].children[0]
      
      # Check if this test might be testing our target method
      if test_likely_targets_method?(node, target_method_name, description)
        test_descriptions << description
      end
    end
    
    # Recursively search child nodes
    node.children.each do |child|
      extract_it_blocks_from_ast(child, test_descriptions, target_method_name)
    end
  end

  def test_likely_targets_method?(test_node, target_method_name, description)
    # Heuristic 1: Method name appears in the test description (improved fuzzy matching)
    return true if method_name_in_description?(target_method_name, description)
    
    # Heuristic 2: Look for method calls in the test body
    method_calls = find_method_calls_in_node(test_node)
    
    # Filter out common RSpec methods (expanded list based on analysis)
    rspec_methods = %w[expect to not_to eq be be_a be_an be_nil be_empty be_truthy be_falsy 
                      have have_key have_attributes include match raise_error change 
                      receive allow allow_any_instance_of and_return and_raise
                      before after let let! subject described_class instance_double double
                      stub_const assign it describe context specify shared_examples
                      shared_context within output contain_exactly avoid_changing
                      have_received respond_to kind_of instance_of strip chomp
                      gsub split join size length first last push pop shift unshift
                      map select reject find detect collect each times call new
                      class send respond_to? is_a? nil? empty? any? all? none?
                      start_with end_with cover an_instance_of a_kind_of be_instance_of
                      be_kind_of be_a_kind_of satisfy and_call_original
                      exactly once twice thrice at_least at_most ordered]
    
    relevant_calls = method_calls.reject { |call| rspec_methods.include?(call.to_s) }
    
    # Check if target method is called in the test
    return true if relevant_calls.include?(target_method_name.to_sym)
    
    # Heuristic 3: Look for calls on common test subjects
    subject_patterns = %w[subject described_class]
    relevant_calls.each do |call|
      # Look for patterns like subject.target_method or described_class.target_method
      return true if call.to_s == target_method_name
    end
    
    # Heuristic 4: Pattern-based matching for behavioral descriptions
    return true if behavioral_pattern_matches?(target_method_name, description)
    
    false
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

  def find_method_calls_in_node(node)
    calls = []
    return calls unless node.is_a?(Parser::AST::Node)
    
    # Look for send nodes (method calls)
    if node.type == :send && node.children[1]
      calls << node.children[1]
    end
    
    # Recursively search child nodes
    node.children.each do |child|
      calls.concat(find_method_calls_in_node(child))
    end
    
    calls
  end

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
      
      # Find test descriptions for this method
      test_descriptions = find_test_descriptions_for_method(
        method_data['repo_name'], 
        method_data['file_path'], 
        method_info[:name]
      )
      
      # Create the paired data entry (always create, docstring and test descriptions are optional)
      create_paired_entry(method_data, method_info, docstring, test_descriptions)
      
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

  def create_paired_entry(original_data, method_info, docstring, test_descriptions = [])
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
    
    # Add test descriptions if available
    test_descriptions.each do |test_desc|
      if test_desc && !test_desc.strip.empty?
        descriptions << {
          "source" => "test_description",
          "text" => test_desc
        }
      end
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