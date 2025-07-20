#!/usr/bin/env ruby

require_relative '05_create_paired_dataset'
require 'fileutils'
require 'tempfile'

# Test different method name matching scenarios
class MethodMatchingTest
  def run_test
    puts "Testing method matching for various scenarios..."
    
    test_cases = [
      { method: 'find_formatter', expected_min: 1 },
      { method: 'register', expected_min: 1 },
      { method: 'add', expected_min: 1 },
      { method: 'nonexistent_method', expected_min: 0 }
    ]
    
    test_spec_content = create_test_spec_content
    
    Tempfile.create(['test_spec', '.rb']) do |temp_file|
      temp_file.write(test_spec_content)
      temp_file.flush
      
      creator = PairedDatasetCreator.new('dummy', 'dummy')
      
      test_cases.each do |test_case|
        puts "\nTesting method: #{test_case[:method]}"
        
        # Test thin RSpec-shell
        rspec_descriptions = creator.send(:extract_test_descriptions_using_rspec_formatter, 
                                         temp_file.path, test_case[:method])
        
        # Test manual parsing
        manual_descriptions = creator.send(:extract_test_descriptions_from_spec_file, 
                                          temp_file.path, test_case[:method])
        
        puts "  RSpec-shell: #{rspec_descriptions.length} descriptions"
        rspec_descriptions.each_with_index do |desc, i|
          puts "    #{i + 1}. #{desc}"
        end
        
        puts "  Manual: #{manual_descriptions.length} descriptions"
        manual_descriptions.each_with_index do |desc, i|
          puts "    #{i + 1}. #{desc}"
        end
        
        success = rspec_descriptions.length >= test_case[:expected_min]
        puts "  Result: #{success ? '✓' : '✗'} (expected >= #{test_case[:expected_min]})"
      end
    end
    
    puts "\nMethod matching test completed!"
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
          
          context 'when formatter is cached' do
            it 'returns cached formatter quickly' do
              expect(true).to be true
            end
          end
        end
        
        describe '#register' do
          it 'registers a new formatter successfully' do
            expect(true).to be true
          end
          
          context 'when registering duplicate' do
            it 'overwrites existing formatter' do
              expect(true).to be true
            end
          end
        end
        
        describe '#add' do
          it 'adds an item to the collection' do
            expect(true).to be true
          end
          
          it 'adds multiple items correctly' do
            expect(true).to be true
          end
        end
        
        describe '#some_other_method' do
          it 'does something completely different' do
            expect(true).to be true
          end
        end
      end
    SPEC
  end
end

# Run the test
if __FILE__ == $0
  test = MethodMatchingTest.new
  test.run_test
end