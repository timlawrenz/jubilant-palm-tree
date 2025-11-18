import argparse
import json
import os
import subprocess

def find_syntactically_similar_code(prompt, dataset_path):
    """Finds a code snippet that contains the keywords in the prompt."""
    
    keywords = prompt.lower().split()
    best_match = None
    best_score = 0
    
    with open(dataset_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            descriptions = [d['text'].lower() for d in sample.get('descriptions', [])]
            
            score = 0
            for keyword in keywords:
                for desc in descriptions:
                    if keyword in desc:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_match = sample
                
    return best_match

def main():
    parser = argparse.ArgumentParser(description="Find a syntactically similar code snippet in the dataset to a given prompt.")
    parser.add_argument("prompt", type=str, help="The text prompt to search for.")
    parser.add_argument("--dataset_path", type=str, default="dataset/paired_data.jsonl", help="Path to the paired dataset.")
    args = parser.parse_args()

    best_sample = find_syntactically_similar_code(args.prompt, args.dataset_path)
    
    if best_sample:
        print("\nFound a potential match:")
        print("\nOriginal Descriptions:")
        for desc in best_sample.get('descriptions', []):
            print(f"- {desc['text']}")
        print("\nGenerated Code:")
        
        # Convert AST to Ruby code
        ast_json = best_sample['ast_json']
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'pretty_print_ast.rb')
        ruby_executable = os.environ.get('RUBY_EXECUTABLE', 'ruby')
        
        try:
            result = subprocess.run(
                [ruby_executable, script_path],
                input=ast_json,
                text=True,
                capture_output=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running pretty_print_ast.rb: {e.stderr}")
    else:
        print("Could not find a similar code snippet.")

if __name__ == "__main__":
    main()
