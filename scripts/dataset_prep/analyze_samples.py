import json

with open("dataset/compressed/compressed_motifs.jsonl", "r") as f:
    sample_1 = json.loads(f.readline())
    sample_2 = json.loads(f.readline())

def analyze(sample):
    print(f"\n{'='*50}")
    print(f"Method: {sample['method_name']}")
    print(f"File: {sample['file_path']}")
    
    graph = sample['compressed_graph']
    nodes = graph['nodes']
    edges = graph['edges']
    literal_pool = graph['literal_pool']
    
    print(f"\n--- Literal Pool ({len(literal_pool)} elements) ---")
    for k, v in literal_pool.items():
        print(f"  {k}: {repr(v)}")
        
    print(f"\n--- Motifs Summary ({len(nodes)} total nodes) ---")
    motif_counts = {}
    for n in nodes:
        motif_counts[n['motif']] = motif_counts.get(n['motif'], 0) + 1
    for m, c in motif_counts.items():
        print(f"  {m}: {c}")
        
    print(f"\n--- Edges Summary ({len(edges)} total edges) ---")
    edge_counts = {}
    for e in edges:
        edge_type = "EXECUTION" if e['edge_type'] == 0 else "DATA"
        edge_counts[edge_type] = edge_counts.get(edge_type, 0) + 1
    for t, c in edge_counts.items():
        print(f"  {t}: {c}")

analyze(sample_1)
analyze(sample_2)
