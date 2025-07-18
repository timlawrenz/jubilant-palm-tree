#!/usr/bin/env ruby
# frozen_string_literal: true

require 'json'
require 'parser/current'

# Pretty-printing script to convert AST JSON back to formatted Ruby code
#
# This script takes the JSON representation of an AST (as used in the project's dataset)
# and converts it back into formatted Ruby code that is syntactically equivalent
# to the original source.

class ASTPrettyPrinter
  # Main method to convert AST JSON to pretty-printed Ruby code
  #
  # @param ast_json [String, Hash] JSON string or parsed hash representing the AST
  # @return [String] Pretty-printed Ruby code
  def self.ast_to_ruby(ast_json)
    ast_hash = ast_json.is_a?(String) ? JSON.parse(ast_json) : ast_json
    new.convert_node(ast_hash)
  end

  # Convert a single AST node to Ruby code
  #
  # @param node [Hash, String, nil] AST node to convert
  # @param indent_level [Integer] Current indentation level
  # @return [String] Ruby code representation
  def convert_node(node, indent_level = 0)
    return '' if node.nil?
    return node.to_s if node.is_a?(String) || node.is_a?(Numeric)

    return '' unless node.is_a?(Hash) && node['type']

    case node['type']
    when 'def'
      convert_def(node, indent_level)
    when 'args'
      convert_args(node)
    when 'begin'
      convert_begin(node, indent_level)
    when 'send'
      convert_send(node)
    when 'block'
      convert_block(node, indent_level)
    when 'self'
      'self'
    when 'nil'
      'nil'
    when 'true'
      'true'
    when 'false'
      'false'
    when 'str'
      convert_string(node)
    when 'int'
      convert_int(node)
    when 'float'
      convert_float(node)
    when 'sym'
      convert_symbol(node)
    when 'lvar'
      convert_lvar(node)
    when 'ivar'
      convert_ivar(node)
    when 'cvar'
      convert_cvar(node)
    when 'gvar'
      convert_gvar(node)
    when 'const'
      convert_const(node)
    when 'if'
      convert_if(node, indent_level)
    when 'unless'
      convert_unless(node, indent_level)
    when 'while'
      convert_while(node, indent_level)
    when 'until'
      convert_until(node, indent_level)
    when 'for'
      convert_for(node, indent_level)
    when 'case'
      convert_case(node, indent_level)
    when 'when'
      convert_when(node, indent_level)
    when 'return'
      convert_return(node)
    when 'break'
      convert_break(node)
    when 'next'
      convert_next(node)
    when 'yield'
      convert_yield(node)
    when 'and'
      convert_and(node)
    when 'or'
      convert_or(node)
    when 'not'
      convert_not(node)
    when 'array'
      convert_array(node)
    when 'hash'
      convert_hash(node)
    when 'pair'
      convert_pair(node)
    when 'splat'
      convert_splat(node)
    when 'kwsplat'
      convert_kwsplat(node)
    when 'class'
      convert_class(node, indent_level)
    when 'module'
      convert_module(node, indent_level)
    when 'sclass'
      convert_sclass(node, indent_level)
    when 'lvasgn'
      convert_lvasgn(node)
    when 'ivasgn'
      convert_ivasgn(node)
    when 'cvasgn'
      convert_cvasgn(node)
    when 'gvasgn'
      convert_gvasgn(node)
    when 'casgn'
      convert_casgn(node)
    when 'dstr'
      convert_dstr(node)
    when 'dnum'
      convert_dnum(node)
    when 'regexp'
      convert_regexp(node)
    when 'resbody'
      convert_resbody(node, indent_level)
    when 'ensure'
      convert_ensure(node, indent_level)
    when 'retry'
      'retry'
    when 'kwarg'
      convert_kwarg(node)
    when 'kwoptarg'
      convert_kwoptarg(node)
    when 'kwrestarg'
      convert_kwrestarg(node)
    when 'blockarg'
      convert_blockarg(node)
    when 'optarg'
      convert_optarg(node)
    when 'restarg'
      convert_restarg(node)
    when 'arg'
      convert_arg(node)
    else
      # Fallback for unknown node types
      convert_unknown(node)
    end
  end

  private

  def indent(level)
    '  ' * level
  end

  def convert_def(node, indent_level)
    children = node['children'] || []
    method_name = children[0]
    args_node = children[1]
    body_node = children[2]

    result = "#{indent(indent_level)}def #{method_name}"
    
    if args_node && args_node['children'] && !args_node['children'].empty?
      result += "(#{convert_node(args_node)})"
    end
    
    result += "\n"
    
    if body_node
      body_code = convert_node(body_node, indent_level + 1)
      result += body_code unless body_code.strip.empty?
    end
    
    result += "\n#{indent(indent_level)}end"
    result
  end

  def convert_args(node)
    children = node['children'] || []
    children.map { |child| convert_node(child) }.join(', ')
  end

  def convert_begin(node, indent_level)
    children = node['children'] || []
    children.map { |child| 
      code = convert_node(child, indent_level)
      # Add indentation if the code doesn't already have it
      if code.start_with?(indent(indent_level))
        code
      else
        "#{indent(indent_level)}#{code}"
      end
    }.join("\n")
  end

  def convert_send(node)
    children = node['children'] || []
    receiver = children[0]
    method_name = children[1]
    args = children[2..-1]

    result = ''
    
    if receiver
      result += convert_node(receiver)
      
      # Special case for array access
      if method_name.to_s == '[]'
        result += "[#{args.map { |arg| convert_node(arg) }.join(', ')}]"
        return result
      end
      
      result += '.'
    end
    
    if method_name.to_s.end_with?('=') && args && !args.empty?
      # For assignment methods, handle specially
      base_name = method_name.to_s.chomp('=')
      result += "#{base_name} = #{args.map { |arg| convert_node(arg) }.join(', ')}"
    else
      # Regular method calls
      result += method_name.to_s
      
      if args && !args.empty?
        # For common methods like raise, don't use parentheses
        if should_omit_parentheses?(method_name.to_s)
          result += " #{args.map { |arg| convert_node(arg) }.join(', ')}"
        else
          result += "(#{args.map { |arg| convert_node(arg) }.join(', ')})"
        end
      end
    end
    
    result
  end

  def should_omit_parentheses?(method_name)
    # Common methods that look better without parentheses
    common_methods = %w[raise puts print p require require_relative include extend]
    # Operators that should not have parentheses
    operators = %w[+ - * / % ** == != < > <= >= <=> === =~ !~ << >> & | ^ ~ && ||]
    
    common_methods.include?(method_name) || operators.include?(method_name)
  end

  def convert_block(node, indent_level)
    children = node['children'] || []
    call_node = children[0]
    block_args = children[1]
    block_body = children[2]

    result = convert_node(call_node)
    
    # Determine if we should use do...end (for multi-line) or { } (for single-line)
    use_do_end = should_use_do_end_block?(block_body)
    
    if use_do_end
      # Multi-line do...end block
      if block_args && block_args['children'] && !block_args['children'].empty?
        result += " do |#{convert_node(block_args)}|\n"
      else
        result += " do\n"
      end

      if block_body
        body_code = convert_node(block_body, indent_level + 1)
        result += body_code
        result += "\n"
      end

      result += "#{indent(indent_level)}end"
    else
      # Single-line { } block
      if block_args && block_args['children'] && !block_args['children'].empty?
        result += " { |#{convert_node(block_args)}|"
      else
        result += " {"
      end

      if block_body
        body_code = convert_node(block_body)
        result += " #{body_code} "
      end

      result += "}"
    end
    
    result
  end

  def should_use_do_end_block?(block_body)
    return false unless block_body
    
    # Use do...end for multi-statement blocks
    if block_body['type'] == 'begin'
      children = block_body['children'] || []
      return children.length > 1
    end
    
    # Use do...end for complex single statements (control flow, etc.)
    case block_body['type']
    when 'if', 'unless', 'case', 'while', 'until', 'for', 'def', 'class', 'module'
      true
    else
      false
    end
  end

  def convert_string(node)
    children = node['children'] || []
    "\"#{children[0]}\""
  end

  def convert_int(node)
    children = node['children'] || []
    children[0].to_s
  end

  def convert_float(node)
    children = node['children'] || []
    children[0].to_s
  end

  def convert_symbol(node)
    children = node['children'] || []
    ":#{children[0]}"
  end

  def convert_lvar(node)
    children = node['children'] || []
    children[0].to_s
  end

  def convert_ivar(node)
    children = node['children'] || []
    "@#{children[0]}"
  end

  def convert_cvar(node)
    children = node['children'] || []
    "@@#{children[0]}"
  end

  def convert_gvar(node)
    children = node['children'] || []
    "$#{children[0]}"
  end

  def convert_const(node)
    children = node['children'] || []
    scope = children[0]
    const_name = children[1]
    
    result = ""
    if scope
      result += "#{convert_node(scope)}::"
    end
    result += const_name.to_s
    result
  end

  def convert_lvasgn(node)
    children = node['children'] || []
    var_name = children[0]
    value = children[1]
    
    if value
      "#{var_name} = #{convert_node(value)}"
    else
      var_name.to_s
    end
  end

  def convert_ivasgn(node)
    children = node['children'] || []
    var_name = children[0]
    value = children[1]
    
    if value
      "@#{var_name} = #{convert_node(value)}"
    else
      "@#{var_name}"
    end
  end

  def convert_cvasgn(node)
    children = node['children'] || []
    var_name = children[0]
    value = children[1]
    
    if value
      "@@#{var_name} = #{convert_node(value)}"
    else
      "@@#{var_name}"
    end
  end

  def convert_gvasgn(node)
    children = node['children'] || []
    var_name = children[0]
    value = children[1]
    
    if value
      "$#{var_name} = #{convert_node(value)}"
    else
      "$#{var_name}"
    end
  end

  def convert_casgn(node)
    children = node['children'] || []
    scope = children[0]
    const_name = children[1]
    value = children[2]
    
    result = ""
    if scope
      result += "#{convert_node(scope)}::"
    end
    result += const_name.to_s
    
    if value
      result += " = #{convert_node(value)}"
    end
    
    result
  end

  def convert_dstr(node)
    children = node['children'] || []
    result = "\""
    
    children.each do |child|
      if child.is_a?(String)
        result += child
      else
        case child['type']
        when 'begin'
          # String interpolation
          result += "\#{#{convert_node(child)}}"
        else
          result += convert_node(child)
        end
      end
    end
    
    result += "\""
    result
  end

  def convert_dnum(node)
    children = node['children'] || []
    children[0].to_s
  end

  def convert_regexp(node)
    children = node['children'] || []
    pattern = children[0]
    flags = children[1]
    
    result = "/#{convert_node(pattern)}/"
    if flags && flags != 0
      result += convert_regexp_flags(flags)
    end
    result
  end

  def convert_regexp_flags(flags)
    flag_str = ""
    flag_str += "i" if flags & 1 != 0   # IGNORECASE
    flag_str += "x" if flags & 2 != 0   # EXTENDED
    flag_str += "m" if flags & 4 != 0   # MULTILINE
    flag_str
  end

  def convert_if(node, indent_level)
    children = node['children'] || []
    condition = children[0]
    then_body = children[1]
    else_body = children[2]

    result = "#{indent(indent_level)}if #{convert_node(condition)}\n"
    
    if then_body
      result += convert_node(then_body, indent_level + 1)
      result += "\n"
    end
    
    if else_body
      result += "#{indent(indent_level)}else\n"
      result += convert_node(else_body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_unless(node, indent_level)
    children = node['children'] || []
    condition = children[0]
    then_body = children[1]
    else_body = children[2]

    result = "#{indent(indent_level)}unless #{convert_node(condition)}\n"
    
    if then_body
      result += convert_node(then_body, indent_level + 1)
      result += "\n"
    end
    
    if else_body
      result += "#{indent(indent_level)}else\n"
      result += convert_node(else_body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_return(node)
    children = node['children'] || []
    if children.empty?
      'return'
    else
      "return #{children.map { |child| convert_node(child) }.join(', ')}"
    end
  end

  def convert_break(node)
    children = node['children'] || []
    if children.empty?
      'break'
    else
      "break #{children.map { |child| convert_node(child) }.join(', ')}"
    end
  end

  def convert_next(node)
    children = node['children'] || []
    if children.empty?
      'next'
    else
      "next #{children.map { |child| convert_node(child) }.join(', ')}"
    end
  end

  def convert_yield(node)
    children = node['children'] || []
    if children.empty?
      'yield'
    else
      "yield(#{children.map { |child| convert_node(child) }.join(', ')})"
    end
  end

  def convert_and(node)
    children = node['children'] || []
    left = convert_node_with_precedence(children[0], 'and')
    right = convert_node_with_precedence(children[1], 'and')
    "#{left} && #{right}"
  end

  def convert_or(node)
    children = node['children'] || []
    left = convert_node_with_precedence(children[0], 'or')
    right = convert_node_with_precedence(children[1], 'or')
    "#{left} || #{right}"
  end

  def convert_node_with_precedence(node, parent_op)
    return convert_node(node) unless node.is_a?(Hash) && node['type']
    
    # Define operator precedence (higher number = higher precedence)
    precedence = {
      'or' => 1,   # ||
      'and' => 2   # &&
    }
    
    node_op = node['type']
    
    # Add parentheses if the child operator has lower precedence than parent
    if precedence[node_op] && precedence[parent_op] && 
       precedence[node_op] < precedence[parent_op]
      "(#{convert_node(node)})"
    else
      convert_node(node)
    end
  end

  def convert_not(node)
    children = node['children'] || []
    "!#{convert_node(children[0])}"
  end

  def convert_array(node)
    children = node['children'] || []
    "[#{children.map { |child| convert_node(child) }.join(', ')}]"
  end

  def convert_hash(node)
    children = node['children'] || []
    "{ #{children.map { |child| convert_node(child) }.join(', ')} }"
  end

  def convert_pair(node)
    children = node['children'] || []
    key = children[0]
    value = children[1]
    
    # Use modern syntax for symbol keys (key: value instead of :key => value)
    if key && key['type'] == 'sym'
      key_name = key['children'][0]
      "#{key_name}: #{convert_node(value)}"
    else
      "#{convert_node(key)} => #{convert_node(value)}"
    end
  end

  def convert_splat(node)
    children = node['children'] || []
    "*#{convert_node(children[0])}"
  end

  def convert_kwsplat(node)
    children = node['children'] || []
    "**#{convert_node(children[0])}"
  end

  def convert_arg(node)
    children = node['children'] || []
    children[0].to_s
  end

  def convert_optarg(node)
    children = node['children'] || []
    name = children[0]
    default = children[1]
    "#{name} = #{convert_node(default)}"
  end

  def convert_restarg(node)
    children = node['children'] || []
    if children.empty?
      '*'
    else
      "*#{children[0]}"
    end
  end

  def convert_kwarg(node)
    children = node['children'] || []
    "#{children[0]}:"
  end

  def convert_kwoptarg(node)
    children = node['children'] || []
    name = children[0]
    default = children[1]
    "#{name}: #{convert_node(default)}"
  end

  def convert_kwrestarg(node)
    children = node['children'] || []
    if children.empty?
      '**'
    else
      "**#{children[0]}"
    end
  end

  def convert_blockarg(node)
    children = node['children'] || []
    "&#{children[0]}"
  end

  def convert_class(node, indent_level)
    children = node['children'] || []
    name = children[0]
    superclass = children[1]
    body = children[2]

    result = "#{indent(indent_level)}class #{convert_node(name)}"
    
    if superclass
      result += " < #{convert_node(superclass)}"
    end
    
    result += "\n"
    
    if body
      result += convert_node(body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_module(node, indent_level)
    children = node['children'] || []
    name = children[0]
    body = children[1]

    result = "#{indent(indent_level)}module #{convert_node(name)}\n"
    
    if body
      result += convert_node(body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_while(node, indent_level)
    children = node['children'] || []
    condition = children[0]
    body = children[1]

    result = "#{indent(indent_level)}while #{convert_node(condition)}\n"
    
    if body
      result += convert_node(body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_until(node, indent_level)
    children = node['children'] || []
    condition = children[0]
    body = children[1]

    result = "#{indent(indent_level)}until #{convert_node(condition)}\n"
    
    if body
      result += convert_node(body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_case(node, indent_level)
    children = node['children'] || []
    expr = children[0]
    branches = children[1..-1]

    result = "#{indent(indent_level)}case #{convert_node(expr)}\n"
    
    # Separate when branches from else clause
    when_branches = []
    else_clause = nil
    
    branches.each do |branch|
      if branch && branch['type'] == 'when'
        when_branches << branch
      elsif branch && branch['type'] != 'when'
        # This is likely the else clause (any non-when node at the end)
        else_clause = branch
      end
    end
    
    # Process when branches
    when_branches.each do |when_node|
      result += convert_node(when_node, indent_level)
      result += "\n"
    end
    
    # Process else clause if present
    if else_clause
      result += "#{indent(indent_level)}else\n"
      result += convert_node(else_clause, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_when(node, indent_level)
    children = node['children'] || []
    conditions = children[0..-2]
    body = children[-1]

    result = "#{indent(indent_level)}when #{conditions.map { |cond| convert_node(cond) }.join(', ')}\n"
    
    if body
      result += convert_node(body, indent_level + 1)
    end
    
    result
  end

  def convert_for(node, indent_level)
    children = node['children'] || []
    var = children[0]
    collection = children[1]
    body = children[2]

    result = "#{indent(indent_level)}for #{convert_node(var)} in #{convert_node(collection)}\n"
    
    if body
      result += convert_node(body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_sclass(node, indent_level)
    children = node['children'] || []
    expr = children[0]
    body = children[1]

    result = "#{indent(indent_level)}class << #{convert_node(expr)}\n"
    
    if body
      result += convert_node(body, indent_level + 1)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}end"
    result
  end

  def convert_rescue(node, indent_level)
    children = node['children'] || []
    body = children[0]
    rescue_clauses = children[1..-2]
    else_clause = children[-1]

    result = ""
    
    if body
      result += convert_node(body, indent_level)
      result += "\n"
    end
    
    rescue_clauses.each do |rescue_clause|
      if rescue_clause && rescue_clause['type'] == 'resbody'
        result += convert_resbody(rescue_clause, indent_level)
        result += "\n"
      end
    end
    
    if else_clause
      result += "#{indent(indent_level)}else\n"
      result += convert_node(else_clause, indent_level + 1)
      result += "\n"
    end
    
    result
  end

  def convert_resbody(node, indent_level)
    children = node['children'] || []
    exception_types = children[0]
    exception_var = children[1]
    body = children[2]

    result = "#{indent(indent_level)}rescue"
    
    if exception_types && exception_types['children'] && !exception_types['children'].empty?
      result += " #{exception_types['children'].map { |type| convert_node(type) }.join(', ')}"
    end
    
    if exception_var
      result += " => #{convert_node(exception_var)}"
    end
    
    result += "\n"
    
    if body
      result += convert_node(body, indent_level + 1)
    end
    
    result
  end

  def convert_ensure(node, indent_level)
    children = node['children'] || []
    body = children[0]
    ensure_body = children[1]

    result = ""
    
    if body
      result += convert_node(body, indent_level)
      result += "\n"
    end
    
    result += "#{indent(indent_level)}ensure\n"
    
    if ensure_body
      result += convert_node(ensure_body, indent_level + 1)
      result += "\n"
    end
    
    result
  end

  def convert_unknown(node)
    # Fallback: try to handle unknown nodes gracefully
    children = node['children'] || []
    if children.empty?
      node['type']
    else
      "#{node['type']}(#{children.map { |child| convert_node(child) }.join(', ')})"
    end
  end
end

# Command-line interface
if __FILE__ == $0
  begin
    if ARGV.empty? && !$stdin.tty?
      # Read from stdin when no arguments and input is piped
      input = $stdin.read
    elsif ARGV.length == 1
      if ARGV[0] == '-'
        # Explicit stdin
        input = $stdin.read
      else
        # Read from file
        input = File.read(ARGV[0])
      end
    else
      puts "Usage: #{$0} <ast_json_file>"
      puts "   or: echo '<ast_json>' | #{$0}"
      puts "   or: #{$0} -"
      exit 1
    end

    result = ASTPrettyPrinter.ast_to_ruby(input)
    puts result
  rescue => e
    $stderr.puts "Error: #{e.message}"
    exit 1
  end
end