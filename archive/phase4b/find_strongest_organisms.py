import requests
import sys

def get_all_generations(chromosome_id):
    """Fetches all generations for a given chromosome."""
    url = f"http://localhost:3001/chromosomes/{chromosome_id}/generations.json"
    print(f"Fetching generations from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching generations for chromosome {chromosome_id}: {e}", file=sys.stderr)
        return None

def get_organisms_for_generation(chromosome_id, generation_id):
    """Fetches all organisms for a given chromosome and generation."""
    url = f"http://localhost:3001/chromosomes/{chromosome_id}/generations/{generation_id}/organisms.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching organisms for generation {generation_id}: {e}", file=sys.stderr)
        return None

def main():
    chromosome_id = 4
    generations = get_all_generations(chromosome_id)

    if not generations:
        print(f"No generations found for chromosome {chromosome_id}.")
        return

    strongest_organisms = []

    for gen in generations:
        gen_id = gen.get('id')
        if gen_id is not None and gen_id < 80:
            organisms = get_organisms_for_generation(chromosome_id, gen_id)
            if organisms:
                fittest_organism = max(organisms, key=lambda org: float(org.get('fitness', -1)))
                fittest_organism['generation_id'] = gen_id  # Add generation_id
                strongest_organisms.append(fittest_organism)

    if not strongest_organisms:
        print("No organisms found in generations before 80.")
        return

    print("\n--- Strongest Organisms (Chromosome 4, Generations < 80) ---")
    for org in sorted(strongest_organisms, key=lambda o: o['generation_id']):
        print(
            f"Generation: {org['generation_id']}, "
            f"Organism ID: {org['id']}, "
            f"Fitness: {org['fitness']}, "
            f"Parameters: conv_type={org['conv_type']}, "
            f"dropout={org['dropout']}, "
            f"hidden_dim={org['hidden_dim']}, "
            f"learning_rate={org['learning_rate']}, "
            f"num_layers={org['num_layers']}, "
            f"parent_weight={org['parent_weight']}, "
            f"type_weight={org['type_weight']}"
        )

if __name__ == "__main__":
    main()