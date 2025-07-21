#!/bin/bash

# Source Code Aggregation Script
# Clones high-quality Ruby repositories for dataset generation

set -e  # Exit on any error

# Define the repositories to clone
REPOS=(
    "rails/rails"
    "sinatra/sinatra"
    "forem/forem"
    "mastodon/mastodon"
    "discourse/discourse"
    "fastlane/fastlane"
    "spree/spree"
    "Shopify/liquid"
    "hanami/hanami"
    "rspec/rspec-core"
    "rubocop/rubocop"
    "sidekiq/sidekiq"
    "ankane/ahoy"
    "procore-oss/blueprinter"
    "givelively/gl_command"
    "wardencommunity/warden"
    "doorkeeper-gem/doorkeeper"
    "varvet/pundit"
    "krisleech/wisper"
    "Shopify/identity_cache"
    "piotrmurach/tty"
    "presidentbeef/brakeman"
    "seattlerb/flog"
    "troessner/reek"
    "rubocop/rubocop"
    "fxn/zeitwerk"
    "grosser/parallel"
    "Multiwoven/multiwoven"
    "puma/puma"
    "fluent/fluentd"
    "aasm/aasm"
    "haml/haml"
    "rack/rack"
    "github/hooks"
    "ManageIQ/manageiq"
    "gitlabhq/rouge"
    "gitlabhq/gitlabhq"
)

# Create repos directory if it doesn't exist
REPOS_DIR="./repos"
echo "Creating repos directory at: $REPOS_DIR"
mkdir -p "$REPOS_DIR"

# Function to clone a repository
clone_repo() {
    local repo=$1
    local repo_name=$(basename "$repo")
    local target_dir="$REPOS_DIR/$repo_name"
    
    echo "----------------------------------------"
    echo "Cloning $repo..."
    
    if [ -d "$target_dir" ]; then
        echo "Directory $target_dir already exists. Skipping clone."
        return 0
    fi
    
    if git clone "https://github.com/$repo.git" "$target_dir"; then
        echo "✓ Successfully cloned $repo"
    else
        echo "✗ Failed to clone $repo"
        return 1
    fi
}

# Main execution
echo "Starting Ruby repository aggregation..."
echo "Target directory: $(pwd)/$REPOS_DIR"
echo "Repositories to clone: ${#REPOS[@]}"
echo ""

# Clone each repository
FAILED_REPOS=()
for repo in "${REPOS[@]}"; do
    if ! clone_repo "$repo"; then
        FAILED_REPOS+=("$repo")
    fi
done

# Summary
echo ""
echo "========================================="
echo "AGGREGATION COMPLETE"
echo "========================================="

if [ ${#FAILED_REPOS[@]} -eq 0 ]; then
    echo "✓ All repositories cloned successfully!"
    echo "Total repositories: ${#REPOS[@]}"
else
    echo "✗ Some repositories failed to clone:"
    for failed_repo in "${FAILED_REPOS[@]}"; do
        echo "  - $failed_repo"
    done
    echo ""
    echo "Successfully cloned: $((${#REPOS[@]} - ${#FAILED_REPOS[@]}))/${#REPOS[@]} repositories"
    exit 1
fi

echo ""
echo "Repositories are available in: $(pwd)/$REPOS_DIR"
ls -la "$REPOS_DIR"
