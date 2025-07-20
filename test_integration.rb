#!/usr/bin/env ruby

require_relative 'scripts/05_create_paired_dataset'

# Test the updated script with our sample spec file
def test_integration
  creator = PairedDatasetCreator.new('./output/processed_methods.jsonl', './dataset/paired_data_test.jsonl')
  
  # Test the RSpec formatter with our sample spec file
  test_descriptions = creator.send(:extract_test_descriptions_using_rspec_formatter, 
                                   './test_specs/user_spec.rb', 
                                   'calculate_total_price')
  
  puts "Test descriptions found for 'calculate_total_price' method:"
  test_descriptions.each do |desc|
    puts "- #{desc}"
  end
  
  # Test with another method name
  puts "\nTest descriptions found for 'log_in' method:"
  test_descriptions2 = creator.send(:extract_test_descriptions_using_rspec_formatter, 
                                    './test_specs/user_spec.rb', 
                                    'log_in')
  
  test_descriptions2.each do |desc|
    puts "- #{desc}"
  end
end

if __FILE__ == $0
  test_integration
end