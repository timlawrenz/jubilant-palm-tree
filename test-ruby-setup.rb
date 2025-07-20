#!/usr/bin/env ruby
# frozen_string_literal: true

# Test script to verify Ruby dependencies are properly installed
# Usage: ruby test-ruby-setup.rb

begin
  puts "🔍 Testing Ruby dependency setup..."
  
  # Test core Ruby
  puts "✅ Ruby #{RUBY_VERSION} available"
  
  # Test required gems
  require 'json'
  puts "✅ JSON gem loaded"
  
  require 'parser/current'
  puts "✅ Parser gem loaded"
  
  # Test basic AST parsing
  code = "def hello; puts 'world'; end"
  ast = Parser::CurrentRuby.parse(code)
  puts "✅ AST parsing works"
  
  puts ""
  puts "🎉 All Ruby dependencies are working correctly!"
  puts "💡 Tip: Use 'source .env-ruby' to set up environment in new shell sessions"
  
rescue LoadError => e
  puts "❌ Missing dependency: #{e.message}"
  puts ""
  puts "🔧 To fix this, run:"
  puts "   ./setup-ruby.sh"
  puts "   source .env-ruby"
  exit 1
rescue StandardError => e
  puts "❌ Error: #{e.message}"
  exit 1
end