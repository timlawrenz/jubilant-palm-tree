#!/usr/bin/env ruby

require_relative '05_create_paired_dataset'
require 'fileutils'
require 'tempfile'

# Test the improved RSpec formatter approach
class RSpecFormatterTest
  def run_test
    puts "Testing thin RSpec-shell approach..."
    
    # Create a temporary spec file for testing
    test_spec_content = create_test_spec_content
    
    Tempfile.create(['test_spec', '.rb']) do |temp_file|
      temp_file.write(test_spec_content)
      temp_file.flush
      
      puts "Created test spec file: #{temp_file.path}"
      puts "Test spec content:"
      puts test_spec_content
      puts "=" * 50
      
      # Test our RSpec formatter approach
      creator = PairedDatasetCreator.new('dummy', 'dummy')
      
      puts "Testing new thin RSpec-shell approach..."
      descriptions = creator.send(:extract_test_descriptions_using_rspec_formatter, 
                                  temp_file.path, 'find_formatter')
      
      puts "Results from thin RSpec-shell:"
      puts "Found #{descriptions.length} descriptions:"
      descriptions.each_with_index do |desc, i|
        puts "  #{i + 1}. #{desc}"
      end
      
      # Test fallback to manual parsing
      puts "\nTesting manual parsing fallback..."
      manual_descriptions = creator.send(:extract_test_descriptions_from_spec_file, 
                                        temp_file.path, 'find_formatter')
      
      puts "Results from manual parsing:"
      puts "Found #{manual_descriptions.length} descriptions:"
      manual_descriptions.each_with_index do |desc, i|
        puts "  #{i + 1}. #{desc}"
      end
    end
    
    puts "\nTest completed!"
  end
  
  private
  
  def create_test_spec_content
    <<~SPEC
      RSpec.describe 'FormatterRegistry' do
        describe '#find_formatter' do
          it 'returns the correct formatter for a given key' do
            expect(true).to be true
          end
          
          it 'handles missing formatters gracefully' do
            expect(true).to be true
          end
        end
        
        describe '#register' do
          it 'registers a new formatter' do
            expect(true).to be true
          end
        end
        
        describe '#add' do
          it 'adds an item to the collection' do
            expect(true).to be true
          end
        end
      end
    SPEC
  end
end

# Run the test
if __FILE__ == $0
  test = RSpecFormatterTest.new
  test.run_test
end