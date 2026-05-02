import argparse
import json
import re
import subprocess
import sys
import time

import requests


def get_latest_generation(chromosome_id):
    """Finds the latest generation for a given chromosome."""
    url = f"http://localhost:3001/chromosomes/{chromosome_id}/generations.json"
    print(f"Fetching generations from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        generations = response.json()
        if not generations:
            print(f"No generations found for chromosome {chromosome_id}.", file=sys.stderr)
            return None
        latest_generation_id = max(g['id'] for g in generations)
        print(f"Found latest generation: {latest_generation_id}")
        return latest_generation_id
    except requests.exceptions.RequestException as e:
        print(f"Error fetching generations for chromosome {chromosome_id}: {e}", file=sys.stderr)
        return None
    except (ValueError, KeyError) as e:
        print(f"Error parsing generations response: {e}", file=sys.stderr)
        return None


def get_untrained_organism(chromosome_id, generation_id):
    """Finds the first organism in a generation with a null fitness."""
    url = f"http://localhost:3001/chromosomes/{chromosome_id}/generations/{generation_id}/organisms.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        organisms = response.json()
        for organism in organisms:
            if organism.get("fitness") is None:
                return organism
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching organisms: {e}", file=sys.stderr)
        raise


def run_training(organism):
    """Constructs and runs the training command for a given organism."""
    organism_id = organism["id"]
    command = [
        "python3",
        "train_autoencoder.py",
        "--epochs",
        "200",
        "--output_path",
        f"models/o_{organism_id}.pt",
        "--batch_size",
        "4096",
        "--decoder_conv_type",
        organism["conv_type"],
        "--dropout",
        str(organism["dropout"]),
        "--hidden_dim",
        str(organism["hidden_dim"]),
        "--learning_rate",
        str(organism["learning_rate"]),
        "--num_layers",
        str(organism["num_layers"]),
        "--parent_weight",
        str(organism["parent_weight"]),
        "--type_weight",
        str(organism["type_weight"]),
    ]
    print(f"Running training for organism {organism_id}...")
    print(" ".join(command))
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=True
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error during training for organism {organism_id}:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return None


def parse_loss(output):
    """Parses the training output to find the best validation loss."""
    match = re.search(r"Best validation loss: (\d+\.\d+)", output)
    if match:
        return float(match.group(1))
    return None


def update_fitness(chromosome_id, generation_id, organism_id, fitness):
    """Updates the fitness of an organism."""
    url = f"http://localhost:3001/chromosomes/{chromosome_id}/generations/{generation_id}/organisms/{organism_id}.json"
    try:
        response = requests.patch(url, json={"fitness": str(fitness)})
        response.raise_for_status()
        print(f"Successfully updated fitness for organism {organism_id} to {fitness}")
    except requests.exceptions.RequestException as e:
        print(f"Error updating fitness for organism {organism_id}: {e}", file=sys.stderr)


def procreate(chromosome_id, generation_id):
    """Triggers the creation of a new generation."""
    url = f"http://localhost:3001/chromosomes/{chromosome_id}/generations/{generation_id}/procreate.json"
    print(f"All organisms trained for generation {generation_id}. Triggering procreation...")
    try:
        response = requests.post(url, timeout=120)
        response.raise_for_status()
        new_generation = response.json()
        new_generation_id = new_generation.get("id")
        if new_generation_id:
            print(f"Successfully created new generation with ID: {new_generation_id}")
            return new_generation_id
        else:
            print("Error: Procreation response did not include a new generation ID.", file=sys.stderr)
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error during procreation for generation {generation_id}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run training for a generation of organisms."
    )
    parser.add_argument(
        "chromosome_id", type=int, help="The ID of the chromosome to process."
    )
    parser.add_argument(
        "--stop_after_current_generation",
        action="store_true",
        help="Stop training after the current generation is complete.",
    )
    args = parser.parse_args()
    chromosome_id = args.chromosome_id

    current_generation_id = get_latest_generation(chromosome_id)
    if current_generation_id is None:
        sys.exit(1)

    while True:
        while True:
            try:
                organism = get_untrained_organism(chromosome_id, current_generation_id)
            except requests.exceptions.RequestException:
                print("Stopping training due to error fetching organisms.", file=sys.stderr)
                return

            if not organism:
                print(f"All organisms have been trained for generation {current_generation_id}.")
                break

            output = run_training(organism)
            if output:
                loss = parse_loss(output)
                if loss is not None:
                    fitness = 17 - loss
                    update_fitness(chromosome_id, current_generation_id, organism["id"], fitness)
                else:
                    print(f"Could not parse loss from output for organism {organism['id']}.")
        
            # Optional: Add a small delay to avoid overwhelming the server
            time.sleep(1)

        if args.stop_after_current_generation:
            print("Stopping training after completing the current generation as requested.")
            break

        new_generation_id = procreate(chromosome_id, current_generation_id)
        if new_generation_id:
            current_generation_id = new_generation_id
        else:
            print("Stopping training as procreation failed.")
            break


if __name__ == "__main__":
    main()
