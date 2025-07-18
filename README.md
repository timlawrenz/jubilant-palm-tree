# jubilant-palm-tree

## Overview

This project is an experiment to determine if a Graph Neural Network (GNN) can learn to understand the structural complexity of Ruby code. The central hypothesis is that by training a GNN on the Abstract Syntax Tree (AST) of thousands of Ruby methods, it can learn to accurately predict a standard complexity metric (Cyclomatic Complexity) without being explicitly taught the rules of the language.

The ultimate goal is not just to predict complexity, but to validate a methodology. If successful, the learned structural embeddings can serve as a foundation for a more advanced generative model capable of writing syntactically correct and logically coherent Ruby code.

This document breaks down the required work into distinct phases and tickets.

## Phase 1: Data Generation & Preprocessing

**Goal:** To produce a clean, structured dataset from raw source code, ready for model training.