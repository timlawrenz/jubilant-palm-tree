#!/bin/bash

# Source Code Aggregation Script
# Clones high-quality Ruby repositories for dataset generation

set -e  # Exit on any error

# Define the repositories to clone
REPOS=(
    "aasm/aasm"
    "ankane/ahoy"
    "chatwoot/chatwoot"
    "crmne/ruby_llm"
    "discourse/discourse"
    "doorkeeper-gem/doorkeeper"
    "fastlane/fastlane"
    "fluent/fluentd"
    "forem/forem"
    "Freika/dawarich"
    "fxn/zeitwerk"
    "github/hooks"
    "givelively/gl_command"
    "gitlabhq/gitlabhq"
    "gitlabhq/rouge"
    "grosser/parallel"
    "haml/haml"
    "hanami/hanami"
    "krisleech/wisper"
    "ManageIQ/manageiq"
    "mastodon/mastodon"
    "Multiwoven/multiwoven"
    "opf/openproject"
    "piotrmurach/tty"
    "presidentbeef/brakeman"
    "procore-oss/blueprinter"
    "puma/puma"
    "rack/rack"
    "rails/rails"
    "rapid7/metasploit-framework"
    "reek/reek"
    "rspec/rspec-core"
    "rubocop/rubocop"
    "seattlerb/flog"
    "Shopify/identity_cache"
    "Shopify/liquid"
    "sidekiq/sidekiq"
    "sinatra/sinatra"
    "sous-chefs/apt"
    "spree/spree"
    "varvet/pundit"
    "wardencommunity/warden"
)

# Create repos directory if it doesn't exist
REPOS_DIR="./repos"
echo "Creating repos directory at: $REPOS_DIR"
mkdir -p "$REPOS_DIR"

# Function to clone or update a repository
clone_or_update_repo() {
    local repo=$1
    local repo_name=$(basename "$repo")
    local target_dir="$REPOS_DIR/$repo_name"
    
    echo "----------------------------------------"
    
    if [ -d "$target_dir" ]; then
        echo "Directory $target_dir already exists. Updating..."
        if (cd "$target_dir" && git pull); then
            echo "✓ Successfully updated $repo"
        else
            echo "✗ Failed to update $repo"
            return 1
        fi
    else
        echo "Cloning $repo..."
        if git clone "https://github.com/$repo.git" "$target_dir"; then
            echo "✓ Successfully cloned $repo"
        else
            echo "✗ Failed to clone $repo"
            return 1
        fi
    fi
}

# Main execution
echo "Starting Ruby repository aggregation..."
echo "Target directory: $(pwd)/$REPOS_DIR"
echo "Repositories to clone: ${#REPOS[@]}"
echo ""

# Clone or update each repository
FAILED_REPOS=()
for repo in "${REPOS[@]}"; do
    if ! clone_or_update_repo "$repo"; then
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
