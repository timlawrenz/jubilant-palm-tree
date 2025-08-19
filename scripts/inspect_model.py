
import torch
import sys
import os
import argparse

# Add src to path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models import ASTAutoencoder

def inspect_model(checkpoint_path):
    """
    Loads a model checkpoint and prints its configuration, architecture,
    and training information.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at: {checkpoint_path}")
        sys.exit(1)

    device = torch.device('cpu')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint file: {e}")
        sys.exit(1)

    # --- Print Checkpoint Structure ---
    print(f"--- Inspecting Checkpoint: {checkpoint_path} ---")
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    print("-" * 40)

    # --- Extract and Print Model Configuration ---
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print("Saved Model Configuration:")
        for key, value in config.items():
            print(f"  - {key}: {value}")
        print("-" * 40)

        # --- Re-create and Print Model Architecture ---
        try:
            # Assuming input_dim is consistent with the dataset (74 features)
            encoder_input_dim = 74 
            model = ASTAutoencoder(
                encoder_input_dim=encoder_input_dim,
                node_output_dim=encoder_input_dim,
                hidden_dim=config.get('hidden_dim', 64),
                num_layers=config.get('num_layers', 3),
                conv_type=config.get('conv_type', 'SAGE'),
                dropout=config.get('dropout', 0.1),
                freeze_encoder=True,
                encoder_weights_path=None  # Don't reload from file here
            )
            
            # The decoder state dict is what's saved in this checkpoint
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            model.eval()

            print("Reconstructed Model Architecture:")
            print(model.get_model_info())
            print("-" * 40)
            
            total_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
            print(f"Decoder Trainable Parameters: {total_params:,}")

        except KeyError as e:
            print(f"Could not reconstruct model. Missing key in checkpoint: {e}")
        except Exception as e:
            print(f"An error occurred during model reconstruction: {e}")

    else:
        print("No 'model_config' found in checkpoint. Cannot determine architecture.")

    # --- Print Training Info ---
    print("-" * 40)
    print("Training Information from Checkpoint:")
    epoch = checkpoint.get('epoch', 'N/A')
    val_loss = checkpoint.get('val_loss', 'N/A')
    
    print(f"  - Trained for Epochs: {epoch}")
    
    if isinstance(val_loss, float):
        print(f"  - Best Validation Loss: {val_loss:.4f}")
    else:
        print(f"  - Best Validation Loss: {val_loss}")
        
    print("--- End of Inspection ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch model checkpoint for the AST Autoencoder.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'model_path',
        type=str,
        help="Path to the model checkpoint file (e.g., models/best_decoder.pt)"
    )
    args = parser.parse_args()
    inspect_model(args.model_path)
