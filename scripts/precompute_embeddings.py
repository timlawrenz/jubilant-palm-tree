#!/usr/bin/env python3
"""
Pre-compute Text Embeddings Script

This script pre-computes text embeddings for all unique descriptions in the
paired dataset using the AlignmentModel. This eliminates the expensive 
text encoding operations from the training loop, significantly improving
training performance.

Usage:
    python scripts/precompute_embeddings.py
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import Dict, Set, List
from tqdm import tqdm

# Add src directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))

from models import AlignmentModel


def load_jsonl_file(filepath: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines
    except FileNotFoundError:
        print(f"Warning: Could not find file {filepath}")
    return data


def extract_unique_text_descriptions(data_files: List[str]) -> Set[str]:
    """
    Extract all unique text descriptions from the dataset files.
    
    Args:
        data_files: List of paths to JSONL files
        
    Returns:
        Set of unique text descriptions
    """
    unique_descriptions = set()
    
    for file_path in data_files:
        print(f"Processing {file_path}...")
        data = load_jsonl_file(file_path)
        
        for sample in data:
            # Extract descriptions from the sample
            descriptions = sample.get('descriptions', [])
            for desc in descriptions:
                if isinstance(desc, dict) and 'text' in desc:
                    text = desc['text'].strip()
                    if text:
                        unique_descriptions.add(text)
            
            # Also check for method_name as fallback
            method_name = sample.get('method_name', '')
            if method_name and method_name.strip():
                unique_descriptions.add(method_name.strip())
    
    return unique_descriptions


def precompute_embeddings(alignment_model: AlignmentModel, 
                         unique_texts: Set[str], 
                         batch_size: int = 32) -> Dict[str, torch.Tensor]:
    """
    Pre-compute embeddings for all unique text descriptions.
    
    Args:
        alignment_model: Loaded AlignmentModel for text encoding
        unique_texts: Set of unique text descriptions
        batch_size: Batch size for processing
        
    Returns:
        Dictionary mapping text -> embedding tensor
    """
    print(f"Pre-computing embeddings for {len(unique_texts)} unique text descriptions...")
    
    text_list = list(unique_texts)
    text_embeddings = {}
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(text_list), batch_size), desc="Computing embeddings"):
        batch_texts = text_list[i:i + batch_size]
        
        # Compute embeddings for this batch
        with torch.no_grad():
            batch_embeddings = alignment_model.encode_text(batch_texts)
        
        # Store individual embeddings
        for text, embedding in zip(batch_texts, batch_embeddings):
            text_embeddings[text] = embedding.cpu()  # Store on CPU to save GPU memory
    
    return text_embeddings


def main():
    """Main function to pre-compute text embeddings."""
    print("🚀 Starting Text Embedding Pre-computation")
    print("=" * 50)
    
    # Set up paths
    project_root = Path(__file__).parent.parent
    dataset_dir = project_root / 'dataset'
    output_dir = project_root / 'output'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Dataset files to process
    data_files = [
        str(dataset_dir / 'paired_data.jsonl'),
        str(dataset_dir / 'train_paired_data.jsonl'),
        str(dataset_dir / 'validation_paired_data.jsonl')
    ]
    
    # Check if any data files exist
    existing_files = [f for f in data_files if os.path.exists(f)]
    if not existing_files:
        print("⚠️  Warning: No dataset files found!")
        print("Expected files:")
        for f in data_files:
            print(f"  - {f}")
        print("\nCreating empty embeddings file for testing...")
        
        # Create empty embeddings for testing
        empty_embeddings = {}
        output_file = output_dir / 'text_embeddings.pt'
        torch.save(empty_embeddings, output_file)
        print(f"✅ Empty embeddings saved to {output_file}")
        return
    
    print(f"Found {len(existing_files)} dataset files:")
    for f in existing_files:
        print(f"  - {f}")
    
    # Extract unique text descriptions
    print("\n📊 Extracting unique text descriptions...")
    unique_texts = extract_unique_text_descriptions(existing_files)
    
    if not unique_texts:
        print("⚠️  Warning: No text descriptions found in dataset files!")
        print("Creating empty embeddings file...")
        
        empty_embeddings = {}
        output_file = output_dir / 'text_embeddings.pt'
        torch.save(empty_embeddings, output_file)
        print(f"✅ Empty embeddings saved to {output_file}")
        return
    
    print(f"Found {len(unique_texts)} unique text descriptions")
    
    # Sample a few descriptions for debugging
    sample_texts = list(unique_texts)[:3]
    print("Sample descriptions:")
    for i, text in enumerate(sample_texts):
        print(f"  {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Load AlignmentModel
    print("\n📦 Loading AlignmentModel...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        alignment_model = AlignmentModel(
            input_dim=74,  # Based on dataset
            hidden_dim=64,
            text_model_name='all-MiniLM-L6-v2',
            code_encoder_weights_path=str(project_root / 'best_model.pt')
        )
        alignment_model = alignment_model.to(device)
        alignment_model.eval()
        print("✅ AlignmentModel loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading AlignmentModel: {e}")
        print("Creating empty embeddings file for testing...")
        
        empty_embeddings = {}
        output_file = output_dir / 'text_embeddings.pt'
        torch.save(empty_embeddings, output_file)
        print(f"✅ Empty embeddings saved to {output_file}")
        return
    
    # Pre-compute embeddings
    print(f"\n⚙️  Pre-computing embeddings...")
    try:
        text_embeddings = precompute_embeddings(
            alignment_model=alignment_model,
            unique_texts=unique_texts,
            batch_size=32
        )
        
        # Save embeddings to file
        output_file = output_dir / 'text_embeddings.pt'
        torch.save(text_embeddings, output_file)
        
        print(f"\n✅ Pre-computation completed!")
        print(f"📁 Embeddings saved to: {output_file}")
        print(f"📊 Total embeddings: {len(text_embeddings)}")
        
        # Verify the saved file
        try:
            loaded_embeddings = torch.load(output_file, map_location='cpu')
            print(f"✅ Verification successful - loaded {len(loaded_embeddings)} embeddings")
            
            # Show embedding info
            if loaded_embeddings:
                sample_key = next(iter(loaded_embeddings.keys()))
                sample_embedding = loaded_embeddings[sample_key]
                print(f"📏 Embedding shape: {sample_embedding.shape}")
                print(f"🔧 Embedding dtype: {sample_embedding.dtype}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not verify saved file: {e}")
            
    except Exception as e:
        print(f"❌ Error during pre-computation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n🎉 Text embedding pre-computation completed successfully!")
    print("This will significantly speed up training by eliminating repeated text encoding.")


if __name__ == "__main__":
    main()