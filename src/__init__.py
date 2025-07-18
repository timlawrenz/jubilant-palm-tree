"""
jubilant-palm-tree: GNN-based Ruby Code Complexity Analysis

This package provides tools for training Graph Neural Networks to predict
the complexity of Ruby code based on Abstract Syntax Tree structure.
"""

__version__ = "0.1.0"
__author__ = "Tim Lawrenz"

# Import main components for easy access
from .models import RubyComplexityGNN, ASTDecoder, ASTAutoencoder
from .loss import ast_reconstruction_loss, ast_reconstruction_loss_simple
from .data_processing import RubyASTDataset, create_data_loaders

__all__ = [
    'RubyComplexityGNN',
    'ASTDecoder', 
    'ASTAutoencoder',
    'ast_reconstruction_loss',
    'ast_reconstruction_loss_simple',
    'RubyASTDataset',
    'create_data_loaders'
]