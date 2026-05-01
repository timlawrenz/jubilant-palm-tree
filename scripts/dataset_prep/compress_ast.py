import json
import os
import sys

from src.execution_engine.schema import ExecutionGraph, Node, Edge, MotifType, EdgeType

MOTIF_MAP = {
    # Boundaries
    "def": MotifType.BOUNDARY, "defs": MotifType.BOUNDARY, "class": MotifType.BOUNDARY, "module": MotifType.BOUNDARY,
    
    # Sequences
    "begin": MotifType.SEQUENCE, "kwbegin": MotifType.SEQUENCE, "block": MotifType.SEQUENCE, 
    "resbody": MotifType.SEQUENCE, "rescue": MotifType.SEQUENCE, "ensure": MotifType.SEQUENCE,
    
    # Conditions
    "if": MotifType.CONDITION, "case": MotifType.CONDITION, "when": MotifType.CONDITION, 
    "and": MotifType.CONDITION, "or": MotifType.CONDITION,
    
    # Loops
    "while": MotifType.LOOP, "until": MotifType.LOOP, "for": MotifType.LOOP, 
    "while_post": MotifType.LOOP, "until_post": MotifType.LOOP,
    
    # States (Variables & Assignments)
    "lvar": MotifType.STATE, "ivar": MotifType.STATE, "cvar": MotifType.STATE, "gvar": MotifType.STATE, 
    "lvasgn": MotifType.STATE, "ivasgn": MotifType.STATE, "cvasgn": MotifType.STATE, "gvasgn": MotifType.STATE,
    "op_asgn": MotifType.STATE, "or_asgn": MotifType.STATE, "and_asgn": MotifType.STATE,
    "arg": MotifType.STATE, "optarg": MotifType.STATE, "restarg": MotifType.STATE, "kwarg": MotifType.STATE,
    "const": MotifType.STATE, "casgn": MotifType.STATE, "mlhs": MotifType.STATE, "masgn": MotifType.STATE,
    "return": MotifType.STATE, "yield": MotifType.STATE,
    
    # Messages & Literals (Everything else defaults here)
    "send": MotifType.MESSAGE, "csend": MotifType.MESSAGE, "super": MotifType.MESSAGE, "zsuper": MotifType.MESSAGE,
    "str": MotifType.MESSAGE, "dstr": MotifType.MESSAGE, "int": MotifType.MESSAGE, "float": MotifType.MESSAGE, "sym": MotifType.MESSAGE,
    "nil": MotifType.MESSAGE, "true": MotifType.MESSAGE, "false": MotifType.MESSAGE, "array": MotifType.MESSAGE, 
    "hash": MotifType.MESSAGE, "pair": MotifType.MESSAGE, "args": MotifType.MESSAGE
}

def get_motif(ruby_type: str) -> MotifType:
    return MOTIF_MAP.get(ruby_type, MotifType.MESSAGE)

class ASTCompressor:
    def __init__(self):
        self.literal_pool = {}
        self.reverse_pool = {} # value -> literal_pointer
        self.nodes = []
        self.edges = []
        self.node_counter = 0

    def get_literal_pointer(self, value):
        if value in self.reverse_pool:
            return self.reverse_pool[value]
        ptr = len(self.literal_pool)
        self.literal_pool[ptr] = value
        self.reverse_pool[value] = ptr
        return ptr

    def process_ast(self, ast_dict) -> int:
        if not ast_dict or not isinstance(ast_dict, dict) or "type" not in ast_dict:
            return -1

        current_id = self.node_counter
        self.node_counter += 1
        
        ruby_type = ast_dict["type"]
        motif = get_motif(ruby_type)
        
        # Check if the node inherently represents a literal value or identifier
        literal_ptr = None
        children = ast_dict.get("children", [])
        
        # Heuristic: If a child is a string/int/float, it's a literal for this node
        # e.g., {"type": "send", "children": [null, "puts", ...]} -> "puts" is a literal
        # e.g., {"type": "str", "children": ["hello"]} -> "hello" is a literal
        valid_children = []
        for child in children:
            if isinstance(child, (str, int, float, bool)) or child is None:
                if child is not None:
                    # Treat this primitive as the literal content of the current Motif
                    if literal_ptr is None:
                        literal_ptr = self.get_literal_pointer(child)
            else:
                valid_children.append(child)

        self.nodes.append(Node(
            node_id=current_id,
            motif=motif,
            literal_pointer=literal_ptr
        ))

        # Process children and create edges
        child_ids = []
        for child in valid_children:
            child_id = self.process_ast(child)
            if child_id != -1:
                child_ids.append(child_id)

        # Heuristic edge routing based on Motif type
        if motif in (MotifType.SEQUENCE, MotifType.BOUNDARY):
            # Sequence: chain children with EXECUTION edges
            # e.g., child 0 -> child 1 -> child 2
            # Connect parent to first child
            if child_ids:
                self.edges.append(Edge(
                    source_node=current_id,
                    target_node=child_ids[0],
                    edge_type=EdgeType.EXECUTION,
                    input_index=0
                ))
            for i in range(len(child_ids) - 1):
                self.edges.append(Edge(
                    source_node=child_ids[i],
                    target_node=child_ids[i+1],
                    edge_type=EdgeType.EXECUTION,
                    input_index=0
                ))
        else:
            # Condition, Loop, Message, State: connect children via DATA edges
            for i, child_id in enumerate(child_ids):
                self.edges.append(Edge(
                    source_node=child_id,
                    target_node=current_id,
                    edge_type=EdgeType.DATA,
                    input_index=i
                ))

        return current_id

def compress_dataset(input_file: str, output_file: str, limit: int = None):
    print(f"Compressing {input_file} into Universal Motifs...")
    
    count = 0
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if not line.strip(): continue
            
            data = json.loads(line)
            ast_json = json.loads(data["ast_json"])
            
            compressor = ASTCompressor()
            compressor.process_ast(ast_json)
            
            graph = ExecutionGraph(
                nodes=compressor.nodes,
                edges=compressor.edges,
                literal_pool=compressor.literal_pool
            )
            
            out_record = {
                "id": data["id"],
                "repo_name": data["repo_name"],
                "file_path": data["file_path"],
                "method_name": data["method_name"],
                "compressed_graph": graph.model_dump()
            }
            
            outfile.write(json.dumps(out_record) + "\n")
            count += 1
            if limit and count >= limit:
                break
                
    print(f"Successfully compressed {count} graphs.")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    import os
    os.makedirs("dataset/compressed", exist_ok=True)
    
    input_path = "dataset_paired_data_example.jsonl"
    output_path = "dataset/compressed/compressed_motifs.jsonl"
    
    if os.path.exists(input_path):
        # We will process the full file
        compress_dataset(input_path, output_path, limit=None)
    else:
        print(f"Input file {input_path} not found.")
