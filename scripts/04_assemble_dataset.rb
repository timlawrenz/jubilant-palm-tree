#!/usr/bin/env ruby

require 'json'
require 'fileutils'
require 'securerandom'

# Dataset Assembly & Cleaning Script
# Reads processed_methods.jsonl and performs final cleaning and validation
# Creates train/validation/test splits and outputs final dataset files

class DatasetAssembler
  def initialize(input_file, output_dir)
    @input_file = input_file
    @output_dir = output_dir
    @complexity_min = 2.0
    @complexity_max = 100.0
    @train_ratio = 0.8
    @validation_ratio = 0.1
    @test_ratio = 0.1
    
    @total_methods = 0
    @filtered_methods = 0
    @final_methods = 0
  end

  def assemble_dataset
    puts "Starting dataset assembly from: #{@input_file}"
    puts "Output directory: #{@output_dir}"
    puts "Complexity filter: #{@complexity_min} <= complexity <= #{@complexity_max}"
    puts "Split ratios: train=#{@train_ratio}, validation=#{@validation_ratio}, test=#{@test_ratio}"
    puts

    # Ensure output directory exists
    FileUtils.mkdir_p(@output_dir) unless Dir.exist?(@output_dir)

    # Read and filter methods
    filtered_methods = read_and_filter_methods
    
    # Assign unique IDs and shuffle
    methods_with_ids = assign_ids_and_shuffle(filtered_methods)
    
    # Split into train/validation/test
    train_data, validation_data, test_data = split_dataset(methods_with_ids)
    
    # Write output files
    write_dataset_files(train_data, validation_data, test_data)
    
    print_summary
  end

  private

  def read_and_filter_methods
    puts "Reading and filtering methods..."
    
    unless File.exist?(@input_file)
      puts "Error: Input file '#{@input_file}' does not exist."
      exit 1
    end

    filtered_methods = []
    
    File.foreach(@input_file) do |line|
      @total_methods += 1
      
      begin
        method_data = JSON.parse(line.strip)
        complexity = method_data['complexity_score'].to_f
        
        # Filter by complexity
        if complexity >= @complexity_min && complexity <= @complexity_max
          filtered_methods << method_data
          @filtered_methods += 1
        end
        
      rescue JSON::ParserError => e
        puts "Warning: Could not parse line #{@total_methods}: #{e.message}"
        next
      end
      
      # Progress indicator
      if @total_methods % 500 == 0
        puts "Processed #{@total_methods} methods, #{@filtered_methods} passed filter"
      end
    end
    
    puts "Filtering complete: #{@filtered_methods}/#{@total_methods} methods passed complexity filter"
    filtered_methods
  end

  def assign_ids_and_shuffle(methods)
    puts "Assigning unique IDs and shuffling dataset..."
    
    # Assign unique IDs
    methods_with_ids = methods.map.with_index do |method, index|
      method['id'] = SecureRandom.uuid
      method
    end
    
    # Shuffle to ensure random distribution
    methods_with_ids.shuffle!
    
    @final_methods = methods_with_ids.length
    puts "Assigned IDs to #{@final_methods} methods"
    
    methods_with_ids
  end

  def split_dataset(methods)
    puts "Splitting dataset..."
    
    total_count = methods.length
    train_count = (total_count * @train_ratio).round
    validation_count = (total_count * @validation_ratio).round
    test_count = total_count - train_count - validation_count
    
    train_data = methods[0...train_count]
    validation_data = methods[train_count...(train_count + validation_count)]
    test_data = methods[(train_count + validation_count)..-1]
    
    puts "Split sizes: train=#{train_data.length}, validation=#{validation_data.length}, test=#{test_data.length}"
    
    [train_data, validation_data, test_data]
  end

  def write_dataset_files(train_data, validation_data, test_data)
    puts "Writing dataset files..."
    
    # Write train.jsonl
    train_file = File.join(@output_dir, 'train.jsonl')
    write_jsonl_file(train_file, train_data)
    
    # Write validation.jsonl
    validation_file = File.join(@output_dir, 'validation.jsonl')
    write_jsonl_file(validation_file, validation_data)
    
    # Write test.jsonl
    test_file = File.join(@output_dir, 'test.jsonl')
    write_jsonl_file(test_file, test_data)
    
    puts "Dataset files written successfully:"
    puts "  #{train_file} (#{train_data.length} entries)"
    puts "  #{validation_file} (#{validation_data.length} entries)"
    puts "  #{test_file} (#{test_data.length} entries)"
  end

  def write_jsonl_file(filename, data)
    File.open(filename, 'w') do |file|
      data.each do |entry|
        file.puts(JSON.generate(entry))
      end
    end
  end

  def print_summary
    puts
    puts "=== Dataset Assembly Summary ==="
    puts "Total methods read: #{@total_methods}"
    puts "Methods after filtering: #{@filtered_methods}"
    puts "Final dataset size: #{@final_methods}"
    puts "Complexity filter: #{@complexity_min} <= complexity <= #{@complexity_max}"
    puts "Filter rate: #{(@filtered_methods.to_f / @total_methods * 100).round(2)}%"
    puts "Output directory: #{@output_dir}"
    puts "Assembly complete!"
  end
end

# Main execution
if __FILE__ == $0
  # Configuration
  input_file = './output/processed_methods.jsonl'
  output_dir = './dataset'
  
  # Create and run the assembler
  assembler = DatasetAssembler.new(input_file, output_dir)
  assembler.assemble_dataset
end