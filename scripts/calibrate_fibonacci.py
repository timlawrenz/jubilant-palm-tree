import torch
from src.execution_engine.demo import test_fibonacci_graph, ExecutionGraph, Node, Edge, MotifType, EdgeType
from src.models.validation import GraphValidator

# 1. First, reconstruct the graph locally to have access to the object (since test_fibonacci_graph() doesn't return it)
pool = {
    0: "a", 1: "b", 2: "count", 3: "temp",
    4: 0, 5: 1, 6: 10,
    7: "+", 8: "<", 9: "print"
}

nodes = [
    # --- Initialization ---
    Node(node_id=0, motif=MotifType.BOUNDARY),
    Node(node_id=1, motif=MotifType.MESSAGE, literal_pointer=4), # Constant 0
    Node(node_id=2, motif=MotifType.STATE, literal_pointer=0),   # a = 0
    Node(node_id=3, motif=MotifType.MESSAGE, literal_pointer=5), # Constant 1
    Node(node_id=4, motif=MotifType.STATE, literal_pointer=1),   # b = 1
    Node(node_id=5, motif=MotifType.STATE, literal_pointer=2),   # count = 0 (using constant 0 from node 1)
    
    # --- Loop Condition ---
    Node(node_id=6, motif=MotifType.STATE, literal_pointer=2),   # Read count
    Node(node_id=7, motif=MotifType.MESSAGE, literal_pointer=6), # Constant 10
    Node(node_id=8, motif=MotifType.MESSAGE, literal_pointer=8), # count < 10
    Node(node_id=9, motif=MotifType.LOOP),                       # The Loop Switch
    
    # --- Loop Body ---
    Node(node_id=10, motif=MotifType.STATE, literal_pointer=0),  # Read a
    Node(node_id=11, motif=MotifType.STATE, literal_pointer=1),  # Read b
    Node(node_id=12, motif=MotifType.MESSAGE, literal_pointer=7),# a + b
    Node(node_id=13, motif=MotifType.STATE, literal_pointer=3),  # temp = a + b
    Node(node_id=14, motif=MotifType.STATE, literal_pointer=1),  # Read b
    Node(node_id=15, motif=MotifType.STATE, literal_pointer=0),  # a = b
    Node(node_id=16, motif=MotifType.STATE, literal_pointer=3),  # Read temp
    Node(node_id=17, motif=MotifType.STATE, literal_pointer=1),  # b = temp
    Node(node_id=18, motif=MotifType.STATE, literal_pointer=2),  # Read count
    Node(node_id=19, motif=MotifType.MESSAGE, literal_pointer=5),# Constant 1
    Node(node_id=20, motif=MotifType.MESSAGE, literal_pointer=7),# count + 1
    Node(node_id=21, motif=MotifType.STATE, literal_pointer=2),  # count = count + 1
    
    # --- Exit ---
    Node(node_id=22, motif=MotifType.STATE, literal_pointer=0),  # Read a
    Node(node_id=23, motif=MotifType.MESSAGE, literal_pointer=9),# print(a)
    Node(node_id=24, motif=MotifType.BOUNDARY),                  # End
]

edges = [
    # Initialization Execution Flow
    Edge(source_node=0, target_node=2, edge_type=EdgeType.EXECUTION, input_index=0),
    Edge(source_node=2, target_node=4, edge_type=EdgeType.EXECUTION, input_index=0),
    Edge(source_node=4, target_node=5, edge_type=EdgeType.EXECUTION, input_index=0),
    Edge(source_node=5, target_node=9, edge_type=EdgeType.EXECUTION, input_index=0),
    
    # Initialization Data Flow
    Edge(source_node=1, target_node=2, edge_type=EdgeType.DATA, input_index=0), # 0 -> a
    Edge(source_node=3, target_node=4, edge_type=EdgeType.DATA, input_index=0), # 1 -> b
    Edge(source_node=1, target_node=5, edge_type=EdgeType.DATA, input_index=0), # 0 -> count
    
    # Loop Condition Data Flow
    Edge(source_node=6, target_node=8, edge_type=EdgeType.DATA, input_index=0), # count
    Edge(source_node=7, target_node=8, edge_type=EdgeType.DATA, input_index=1), # 10
    Edge(source_node=8, target_node=9, edge_type=EdgeType.DATA, input_index=0), # (count < 10) -> LOOP
    
    # Loop Execution Routing
    Edge(source_node=9, target_node=13, edge_type=EdgeType.EXECUTION, input_index=0), # True path -> Body
    Edge(source_node=9, target_node=23, edge_type=EdgeType.EXECUTION, input_index=1), # False path -> Print
    
    # Loop Body Data Flow
    Edge(source_node=10, target_node=12, edge_type=EdgeType.DATA, input_index=0), # a
    Edge(source_node=11, target_node=12, edge_type=EdgeType.DATA, input_index=1), # b
    Edge(source_node=12, target_node=13, edge_type=EdgeType.DATA, input_index=0), # (a+b) -> temp
    Edge(source_node=14, target_node=15, edge_type=EdgeType.DATA, input_index=0), # b -> a
    Edge(source_node=16, target_node=17, edge_type=EdgeType.DATA, input_index=0), # temp -> b
    Edge(source_node=18, target_node=20, edge_type=EdgeType.DATA, input_index=0), # count
    Edge(source_node=19, target_node=20, edge_type=EdgeType.DATA, input_index=1), # 1
    Edge(source_node=20, target_node=21, edge_type=EdgeType.DATA, input_index=0), # (count+1) -> count
    
    # Loop Body Execution Flow
    Edge(source_node=13, target_node=15, edge_type=EdgeType.EXECUTION, input_index=0),
    Edge(source_node=15, target_node=17, edge_type=EdgeType.EXECUTION, input_index=0),
    Edge(source_node=17, target_node=21, edge_type=EdgeType.EXECUTION, input_index=0),
    Edge(source_node=21, target_node=9,  edge_type=EdgeType.EXECUTION, input_index=0), # Loop back to Condition
    
    # Exit Flow
    Edge(source_node=22, target_node=23, edge_type=EdgeType.DATA, input_index=0),      # a -> print
    Edge(source_node=23, target_node=24, edge_type=EdgeType.EXECUTION, input_index=0), # print -> End
]

# 2. Convert to Tensors
N = 25
B = 1
motifs = torch.zeros((B, N), dtype=torch.int32)
adj = torch.zeros((B, 3, N, N), dtype=torch.float32)

MOTIF_STR_TO_INT = {
    MotifType.BOUNDARY: 1,
    MotifType.SEQUENCE: 2,
    MotifType.CONDITION: 3,
    MotifType.LOOP: 4,
    MotifType.STATE: 5,
    MotifType.MESSAGE: 6
}

for node in nodes:
    motifs[0, node.node_id] = MOTIF_STR_TO_INT[node.motif]

for edge in edges:
    u = edge.source_node
    v = edge.target_node
    adj[0, 0, u, v] = 1.0 # presence
    adj[0, 1, u, v] = 0.0 if edge.edge_type == EdgeType.EXECUTION else 1.0 # edge_type
    adj[0, 2, u, v] = float(edge.input_index) # index

# 3. Evaluate
results = GraphValidator.evaluate_batch(adj, motifs)

print("\n--- Fibonacci Calibration Results ---")
for k, v in results.items():
    print(f"{k}: {v:.2f}%")
