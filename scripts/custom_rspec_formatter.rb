#!/usr/bin/env ruby

require 'rspec'
require 'json'

# Custom RSpec Formatter for collecting test descriptions and metadata
# without actually running the tests. This formatter hooks into RSpec's
# event system to capture contextual test information.
class CustomRSpecFormatter
  # Register for events we want to listen to  
  RSpec::Core::Formatters.register self, 
    :example_group_started, 
    :example_group_finished,
    :example_started,
    :dump_summary

  attr_reader :output

  def initialize(output)
    @output = output
    @context_stack = []
    @collected_examples = []
  end

  # Called when a describe/context block is started
  def example_group_started(notification)
    group = notification.group
    
    # Build the description for this group
    description = group.description.to_s.strip
    
    # Store metadata about the group
    group_info = {
      description: description,
      file_path: group.metadata[:file_path],
      line_number: group.metadata[:line_number],
      scoped_id: group.metadata[:scoped_id]
    }
    
    @context_stack.push(group_info)
  end

  # Called when a describe/context block is finished
  def example_group_finished(notification)
    @context_stack.pop
  end

  # Called when an it/specify/example block is started
  def example_started(notification)
    example = notification.example
    
    # Build the full contextual description
    context_parts = @context_stack.map { |ctx| ctx[:description] }.reject(&:empty?)
    example_description = example.description.to_s.strip
    
    # Create a full contextual sentence
    full_description = if context_parts.any?
      # Combine context and example: "A User, with a valid profile, can log in successfully"
      "#{context_parts.join(', ')}, #{example_description}"
    else
      example_description
    end
    
    # Collect the example information
    example_info = {
      full_description: full_description,
      example_description: example_description,
      context_descriptions: context_parts,
      file_path: example.metadata[:file_path],
      line_number: example.metadata[:line_number],
      scoped_id: example.metadata[:scoped_id],
      # Store the raw AST location for linking back to methods
      location: example.metadata[:location],
      # Store any subject or described_class information if available
      described_class: example.metadata[:described_class],
      subject: extract_subject_info(example)
    }
    
    @collected_examples << example_info
  end

  # Called at the end of the test run
  def dump_summary(notification)
    # Output the collected examples as JSON for easy parsing
    require 'json'
    
    result = {
      total_examples: @collected_examples.length,
      examples: @collected_examples
    }
    
    output.puts JSON.pretty_generate(result)
  end

  # Accessor for external usage
  def collected_examples
    @collected_examples
  end

  private

  # Extract subject information from the example if available
  def extract_subject_info(example)
    # Try to get subject class information
    subject_info = {}
    
    # Check if described_class is available
    if example.metadata[:described_class]
      subject_info[:described_class] = example.metadata[:described_class].to_s
    end
    
    # Try to extract any explicitly named subjects
    # This is more complex and would require analyzing the example group's
    # let/subject declarations, which is beyond scope for now
    
    subject_info
  end
end