#!/usr/bin/env ruby

require_relative 'scripts/custom_rspec_formatter'
require 'rspec/core'
require 'stringio'

# Test our custom RSpec formatter
def test_formatter
  spec_file = './test_specs/user_spec.rb'
  
  # Clear any previous configuration
  RSpec.clear_examples
  
  begin
    # Run RSpec with our custom formatter in dry-run mode
    args = [
      '--dry-run',
      '--format', 'CustomRSpecFormatter',
      spec_file
    ]
    
    # Capture the output
    original_stdout = $stdout
    formatter_output = StringIO.new
    $stdout = formatter_output
    
    # Create a formatter instance manually
    formatter = CustomRSpecFormatter.new(formatter_output)
    
    # Set up RSpec to use our formatter
    RSpec.configure do |config|
      config.formatter = formatter
      config.dry_run = true
    end
    
    # Run RSpec
    exit_code = RSpec::Core::Runner.run([spec_file, '--dry-run'])
    
    # Restore stdout
    $stdout = original_stdout
    
    puts "Exit code: #{exit_code}"
    puts "Collected examples:"
    formatter.collected_examples.each do |example|
      puts "- #{example[:full_description]}"
      puts "  File: #{example[:file_path]}:#{example[:line_number]}"
      puts
    end
    
  rescue => e
    $stdout = original_stdout if original_stdout
    puts "Error: #{e.message}"
    puts e.backtrace.join("\n")
  end
end

if __FILE__ == $0
  test_formatter
end