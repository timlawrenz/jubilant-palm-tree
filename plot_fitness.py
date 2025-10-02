import requests
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_organisms(chromosome_id, generation_id):
    """Fetches organisms for a given chromosome and generation."""
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
    generation_ids = range(76, 101)
    
    fitness_data = {
        'generation': [],
        'min_fitness': [],
        'max_fitness': [],
        'avg_fitness': []
    }

    for gen_id in generation_ids:
        organisms = get_organisms(chromosome_id, gen_id)
        if organisms:
            fitness_values = [float(org['fitness']) for org in organisms if org.get('fitness') is not None]
            if fitness_values:
                fitness_data['generation'].append(gen_id)
                fitness_data['min_fitness'].append(np.min(fitness_values))
                fitness_data['max_fitness'].append(np.max(fitness_values))
                fitness_data['avg_fitness'].append(np.mean(fitness_values))

    if not fitness_data['generation']:
        print("No fitness data found for any generation.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(fitness_data['generation'], fitness_data['min_fitness'], label='Min Fitness')
    plt.plot(fitness_data['generation'], fitness_data['max_fitness'], label='Max Fitness')
    plt.plot(fitness_data['generation'], fitness_data['avg_fitness'], label='Avg Fitness')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Progression for Chromosome 4')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('fitness_progression_chromosome_4.png')
    print("Plot saved to fitness_progression_chromosome_4.png")

if __name__ == "__main__":
    main()
