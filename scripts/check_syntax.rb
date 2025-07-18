#!/usr/bin/env ruby
# frozen_string_literal: true

require 'parser/current'

# Reads code from stdin and exits with 0 if syntax is valid, 1 otherwise.
begin
  code = $stdin.read
  Parser::CurrentRuby.parse(code)
  exit 0
rescue Parser::SyntaxError => e
  $stderr.puts e.message
  exit 1
end