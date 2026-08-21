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

    def process_ast(self, ast_dict):
        if not ast_dict or not isinstance(ast_dict, dict) or "type" not in ast_dict:
            return -1, [], False

        current_id = self.node_counter
        self.node_counter += 1
        
        ruby_type = ast_dict["type"]
        motif = get_motif(ruby_type)
        
        # Check if the node inherently represents a literal value or identifier
        literal_ptr = None
        children = ast_dict.get("children", [])

        # Bound primitive bool/nil nodes: pool-valid values, interpreter-aware
        if ruby_type == "nil":
            literal_ptr = self.get_literal_pointer("nil")
        elif ruby_type == "true":
            literal_ptr = self.get_literal_pointer(True)
        elif ruby_type == "false":
            literal_ptr = self.get_literal_pointer(False)
        
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
        child_exits = []   # open continuation points of each child subtree
        child_has_branch = []
        for child in valid_children:
            child_id, exits, has_branch = self.process_ast(child)
            if child_id != -1:
                child_ids.append(child_id)
                child_exits.append(exits)
                child_has_branch.append(has_branch)
        has_branch = ruby_type in ("if", "while", "until") or any(child_has_branch)

        # Heuristic edge routing based on Motif type
        if motif in (MotifType.SEQUENCE, MotifType.BOUNDARY):
            # Sequence: chain children with EXECUTION edges. Continuation
            # attaches to the OPEN EXITS of the previous child, not the child
            # root — so branch nodes (if/while) get join semantics.
            if child_ids:
                self.edges.append(Edge(
                    source_node=current_id,
                    target_node=child_ids[0],
                    edge_type=EdgeType.EXECUTION,
                    input_index=0
                ))
            for i in range(len(child_ids) - 1):
                for (exit_node, out_idx) in child_exits[i]:
                    self.edges.append(Edge(
                        source_node=exit_node,
                        target_node=child_ids[i + 1],
                        edge_type=EdgeType.EXECUTION,
                        input_index=out_idx
                    ))
        elif ruby_type == "if":
            # C1.2: {0,1} True/False EXEC branch pair; cond is DATA-in.
            # Continuation = open exits of BOTH subtrees (join semantics).
            if child_ids:
                self.edges.append(Edge(
                    source_node=child_ids[0],
                    target_node=current_id,
                    edge_type=EdgeType.DATA,
                    input_index=0
                ))
            missing_exits = []
            def _branch(child_idx, out_idx):
                if child_idx < len(child_ids):
                    self.edges.append(Edge(
                        source_node=current_id,
                        target_node=child_ids[child_idx],
                        edge_type=EdgeType.EXECUTION,
                        input_index=out_idx
                    ))
                else:
                    nil_id = self.node_counter
                    self.node_counter += 1
                    self.nodes.append(Node(
                        node_id=nil_id, motif=MotifType.MESSAGE,
                        literal_pointer=self.get_literal_pointer("nil")))
                    self.edges.append(Edge(
                        source_node=current_id, target_node=nil_id,
                        edge_type=EdgeType.EXECUTION, input_index=out_idx))
                    missing_exits.append((nil_id, 0))
            _branch(1, 0)
            _branch(2, 1)
            # JOIN: open exits = union of then-exits and else-exits
            join_exits = []
            if len(child_exits) > 1:
                join_exits.extend(child_exits[1])
            if len(child_exits) > 2:
                join_exits.extend(child_exits[2])
            join_exits.extend(missing_exits)
            if not join_exits:
                join_exits = [(current_id, 0)]
            return current_id, join_exits, True
        elif ruby_type in ("while", "until"):
            # cond DATA-in; body EXEC-out idx 0. Open exits: (body_end, 0) for
            # the run-through path and (self, 1) for the loop-exit path.
            if child_ids:
                self.edges.append(Edge(
                    source_node=child_ids[0],
                    target_node=current_id,
                    edge_type=EdgeType.DATA,
                    input_index=0
                ))
            if len(child_ids) > 1:
                self.edges.append(Edge(
                    source_node=current_id,
                    target_node=child_ids[1],
                    edge_type=EdgeType.EXECUTION,
                    input_index=0
                ))
            exits = [(current_id, 1)]  # loop-exit path (parent joins here)
            if len(child_exits) > 1:
                exits.extend(child_exits[1])  # run-through path
            return current_id, exits, True
        else:
            # Message, State: connect children via DATA edges
            for i, child_id in enumerate(child_ids):
                self.edges.append(Edge(
                    source_node=child_id,
                    target_node=current_id,
                    edge_type=EdgeType.DATA,
                    input_index=i
                ))

        # Open exits of THIS node:
        # - branch subtrees: exits = last child's open exits (join propagation)
        # - branch-free subtrees: [(self, 0)] — EXACT old chaining behavior
        # - return/yield: TERMINAL — no continuation (method ends here)
        if ruby_type in ("return", "yield"):
            return current_id, [], has_branch
        if has_branch and motif in (MotifType.SEQUENCE, MotifType.BOUNDARY):
            if child_exits:
                return current_id, child_exits[-1], True
        return current_id, [(current_id, 0)], has_branch

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
