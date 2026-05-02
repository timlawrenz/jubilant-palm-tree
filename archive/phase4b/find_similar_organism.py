import requests
import sys
import numpy as np

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

def get_all_organisms(chromosome_id):
    """Fetches all organisms for a given chromosome across all generations."""
    generations = get_all_generations(chromosome_id)
    if not generations:
        return None

    all_organisms = []
    for gen in generations:
        gen_id = gen.get('id')
        if gen_id is not None:
            organisms = get_organisms_for_generation(chromosome_id, gen_id)
            if organisms:
                for org in organisms:
                    org['generation_id'] = gen_id
                all_organisms.extend(organisms)
    return all_organisms

def normalize_params(organisms):
    """Normalizes the hyperparameters of the organisms."""
    params = {
        'dropout': [float(o['dropout']) for o in organisms],
        'hidden_dim': [int(o['hidden_dim']) for o in organisms],
        'learning_rate': [float(o['learning_rate']) for o in organisms],
        'num_layers': [int(o['num_layers']) for o in organisms],
        'parent_weight': [float(o['parent_weight']) for o in organisms],
        'type_weight': [float(o['type_weight']) for o in organisms]
    }

    normalized_params = {}
    for key, values in params.items():
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            normalized_params[key] = [0.5] * len(values)
        else:
            normalized_params[key] = [(v - min_val) / (max_val - min_val) for v in values]

    for i, organism in enumerate(organisms):
        organism['normalized_params'] = {key: normalized_params[key][i] for key in params.keys()}
        # Also handle conv_type separately
        conv_types = ['GCN', 'SAGE', 'GAT', 'GIN', 'GraphConv']
        organism['normalized_params']['conv_type'] = conv_types.index(organism['conv_type']) / (len(conv_types) - 1)


def vector_distance(org1, org2):
    """Calculates the Euclidean distance between the normalized parameter vectors of two organisms."""
    vec1 = np.array(list(org1['normalized_params'].values()))
    vec2 = np.array(list(org2['normalized_params'].values()))
    return np.linalg.norm(vec1 - vec2)

def main():
    chromosome_id = 4
    target_organism_id = 2818

    all_organisms = get_all_organisms(chromosome_id)
    if not all_organisms:
        print("Could not retrieve any organisms.")
        return

    normalize_params(all_organisms)

    target_organism = next((o for o in all_organisms if o['id'] == target_organism_id), None)
    if not target_organism:
        print(f"Organism {target_organism_id} not found.")
        return

    most_similar_organism = None
    smallest_distance = float('inf')

    for organism in all_organisms:
        if organism['id'] == target_organism_id:
            continue
        distance = vector_distance(target_organism, organism)
        if distance < smallest_distance:
            smallest_distance = distance
            most_similar_organism = organism

    print("\n--- Target Organism ---")
    print(
        f"Generation: {target_organism['generation_id']}, "
        f"Organism ID: {target_organism['id']}, "
        f"Fitness: {target_organism['fitness']}"
    )
    print(f"Parameters: { {k: v for k, v in target_organism.items() if k in ['conv_type', 'dropout', 'hidden_dim', 'learning_rate', 'num_layers', 'parent_weight', 'type_weight']} }")


    if most_similar_organism:
        print("\n--- Most Similar Organism ---")
        print(
            f"Generation: {most_similar_organism['generation_id']}, "
            f"Organism ID: {most_similar_organism['id']}, "
            f"Fitness: {most_similar_organism['fitness']}"
        )
        print(f"Parameters: { {k: v for k, v in most_similar_organism.items() if k in ['conv_type', 'dropout', 'hidden_dim', 'learning_rate', 'num_layers', 'parent_weight', 'type_weight']} }")
        print(f"Similarity Distance: {smallest_distance}")

if __name__ == "__main__":
    main()
