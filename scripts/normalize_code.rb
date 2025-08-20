#!/usr/bin/env ruby
# frozen_string_literal: true

require 'parser/current'
require 'json'

# Normalizes Ruby code by replacing identifiers with generic placeholders.
# This allows for semantic comparison of code structure without being
# sensitive to specific variable or method names.
class CodeNormalizer
  def initialize
    @variable_map = {}
    @method_map = {}
    @ivar_map = {}
    @const_map = {}
    @variable_count = 0
    @method_count = 0
    @ivar_count = 0
    @const_count = 0
  end

  def normalize(code)
    @variable_map = {}
    @method_map = {}
    @ivar_map = {}
    @const_map = {}
    @variable_count = 0
    @method_count = 0
    @ivar_count = 0
    @const_count = 0
    
    begin
      ast = Parser::CurrentRuby.parse(code)
      rewrite(ast).to_s
    rescue Parser::SyntaxError
      # If syntax is invalid, return a special token
      "SYNTAX_ERROR"
    end
  end

  private

  def rewrite(node)
    return node unless node.is_a?(Parser::AST::Node)

    handler = :"on_#{node.type}"
    if respond_to?(handler, true)
      send(handler, node)
    else
      node.updated(nil, node.children.map { |child| rewrite(child) })
    end
  end

  def on_def(node)
    name, args, body = node.children
    
    # Normalize method name
    normalized_name = (@method_map[name] ||= "METHOD_#{@method_count += 1}")

    # Process args and body
    normalized_args = rewrite(args)
    normalized_body = rewrite(body)
    
    node.updated(:def, [normalized_name.to_sym, normalized_args, normalized_body])
  end

  def on_send(node)
    receiver, method_name, *args = node.children
    
    normalized_receiver = rewrite(receiver)
    
    # Normalize method name
    normalized_method = (@method_map[method_name] ||= "METHOD_#{@method_count += 1}")
    
    normalized_args = args.map { |arg| rewrite(arg) }
    
    node.updated(:send, [normalized_receiver, normalized_method.to_sym, *normalized_args])
  end

  def on_lvar(node)
    name = node.children.first
    normalized_name = (@variable_map[name] ||= "VAR_#{@variable_count += 1}")
    node.updated(:lvar, [normalized_name.to_sym])
  end
  
  def on_ivar(node)
    name = node.children.first
    normalized_name = (@ivar_map[name] ||= "IVAR_#{@ivar_count += 1}")
    node.updated(:ivar, [normalized_name.to_sym])
  end

  def on_const(node)
    _scope, name = node.children
    normalized_name = (@const_map[name] ||= "CONST_#{@const_count += 1}")
    node.updated(:const, [nil, normalized_name.to_sym])
  end

  def on_lvasgn(node)
    name, value = node.children
    normalized_name = (@variable_map[name] ||= "VAR_#{@variable_count += 1}")
    normalized_value = rewrite(value)
    node.updated(:lvasgn, [normalized_name.to_sym, normalized_value])
  end
  
  def on_ivasgn(node)
    name, value = node.children
    normalized_name = (@ivar_map[name] ||= "IVAR_#{@ivar_count += 1}")
    normalized_value = rewrite(value)
    node.updated(:ivasgn, [normalized_name.to_sym, normalized_value])
  end

  def on_arg(node)
    name = node.children.first
    normalized_name = (@variable_map[name] ||= "VAR_#{@variable_count += 1}")
    node.updated(:arg, [normalized_name.to_sym])
  end
end

if __FILE__ == $PROGRAM_NAME
  input = $stdin.read
  normalizer = CodeNormalizer.new
  puts normalizer.normalize(input)
end
