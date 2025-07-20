#!/bin/bash
# Ruby dependency setup script for Copilot coding agents
#
# This script handles the common Ruby gem installation workflow that 
# Copilot agents encounter, avoiding permission errors and missing dependencies.
#
# Usage: ./setup-ruby.sh

set -e

echo "🚀 Setting up Ruby dependencies for jubilant-palm-tree..."

# Detect Ruby version for gem path
RUBY_VERSION=$(ruby -e "puts RUBY_VERSION.match(/\d+\.\d+/)[0]")
GEM_USER_DIR="$HOME/.local/share/gem/ruby/$RUBY_VERSION.0"
GEM_USER_BIN="$GEM_USER_DIR/bin"

echo "📍 Ruby version: $(ruby --version)"
echo "📦 User gem directory: $GEM_USER_DIR"

# Install bundler to user directory if not already installed
if ! command -v bundle >/dev/null 2>&1; then
    echo "📥 Installing bundler to user directory..."
    gem install --user-install bundler
else
    echo "✅ Bundler already installed"
fi

# Configure bundler to use user installation by default
echo "⚙️  Configuring bundler for user installation..."
mkdir -p .bundle
"$GEM_USER_BIN/bundle" config set --local path "$GEM_USER_DIR"
"$GEM_USER_BIN/bundle" config set --local bin "$GEM_USER_BIN"

# Install gems using the user-configured bundler
echo "📦 Installing gems with bundler..."
export PATH="$GEM_USER_BIN:$PATH"
export GEM_PATH="$GEM_USER_DIR:$GEM_PATH"
bundle install

echo "✅ Ruby dependencies installed successfully!"
echo ""
echo "To use Ruby gems in this session, run:"
echo "  export PATH=\"$GEM_USER_BIN:\$PATH\""
echo "  export GEM_PATH=\"$GEM_USER_DIR:\$GEM_PATH\""
echo ""
echo "Or source the environment file:"
echo "  source .env-ruby"