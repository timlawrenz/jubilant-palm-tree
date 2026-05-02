from typing import Any, Dict
from src.execution_engine.schema import ExecutionGraph, MotifType, EdgeType

class GraphInterpreter:
    def __init__(self, graph: ExecutionGraph):
        self.graph = graph
        self.memory: Dict[str, Any] = {}
        
        # Build lookup tables for O(1) traversal
        self.nodes = {n.node_id: n for n in graph.nodes}
        
        self.exec_out = {}
        self.data_in = {}
        
        for e in graph.edges:
            if e.edge_type == EdgeType.EXECUTION:
                if e.source_node not in self.exec_out:
                    self.exec_out[e.source_node] = {}
                self.exec_out[e.source_node][e.input_index] = e.target_node
            elif e.edge_type == EdgeType.DATA:
                if e.target_node not in self.data_in:
                    self.data_in[e.target_node] = {}
                self.data_in[e.target_node][e.input_index] = e.source_node

    def _resolve_data(self, node_id: int, resolving_set=None) -> Any:
        """Recursively resolves data dependencies for a given node."""
        if resolving_set is None:
            resolving_set = set()
            
        if node_id in resolving_set:
            raise RuntimeError(f"Cyclic data dependency detected involving node {node_id}")
            
        resolving_set.add(node_id)
        
        node = self.nodes[node_id]
        
        # If it's a STATE node, it reads from memory
        if node.motif == MotifType.STATE:
            var_name = self.graph.literal_pool[node.literal_pointer]
            resolving_set.remove(node_id)
            return self.memory.get(var_name, None)
            
        # If it's a CONSTANT (represented by a MESSAGE motif with no inputs)
        if node.motif == MotifType.MESSAGE and node_id not in self.data_in:
            resolving_set.remove(node_id)
            return self.graph.literal_pool[node.literal_pointer]
            
        # If it's a MESSAGE with inputs, execute it to get the value
        if node.motif == MotifType.MESSAGE:
            args = [self._resolve_data(self.data_in[node_id][i], resolving_set) for i in sorted(self.data_in[node_id].keys())]
            resolving_set.remove(node_id)
            return self._execute_message(node.literal_pointer, args)

        resolving_set.remove(node_id)
        raise RuntimeError(f"Cannot resolve data for Motif: {node.motif}")

    def _execute_message(self, literal_pointer: int, args: list) -> Any:
        """Executes a hardcoded function based on the literal pool string."""
        func_name = self.graph.literal_pool[literal_pointer]
        
        # The MVP standard library
        if func_name == "+":
            return args[0] + args[1]
        elif func_name == "<":
            return args[0] < args[1]
        elif func_name == "-":
            return args[0] - args[1]
        elif func_name == "print":
            print(f"> {args[0]}")
            return None
        else:
            raise NotImplementedError(f"Function '{func_name}' not in MVP standard library.")

    def run(self, max_steps: int = 10000):
        # Find the starting BOUNDARY node
        start_nodes = [n.node_id for n in self.graph.nodes if n.motif == MotifType.BOUNDARY]
        if not start_nodes:
            raise ValueError("Graph has no BOUNDARY node to start execution.")
            
        # Assume the first boundary is the entry point
        current_node_id = start_nodes[0]
        step_count = 0
        
        while current_node_id is not None:
            if step_count > max_steps:
                raise RuntimeError(f"Max execution steps ({max_steps}) exceeded. Possible infinite loop.")
            step_count += 1
            
            node = self.nodes[current_node_id]
            next_node_id = None
            
            if node.motif == MotifType.BOUNDARY:
                if current_node_id in self.exec_out:
                    next_node_id = self.exec_out[current_node_id].get(0)
                else:
                    # End of program
                    break
                    
            elif node.motif == MotifType.SEQUENCE:
                if current_node_id not in self.exec_out or 0 not in self.exec_out[current_node_id]:
                    raise RuntimeError(f"SEQUENCE node {current_node_id} is missing an outgoing execution edge.")
                next_node_id = self.exec_out[current_node_id][0]
                
            elif node.motif == MotifType.STATE:
                # STATE node in the execution path is a WRITE operation
                var_name = self.graph.literal_pool[node.literal_pointer]
                if current_node_id not in self.data_in or 0 not in self.data_in[current_node_id]:
                    raise RuntimeError(f"STATE node {current_node_id} (write) is missing an incoming data edge.")
                # Get the value to write from the incoming data edge
                val_to_write = self._resolve_data(self.data_in[current_node_id][0])
                self.memory[var_name] = val_to_write
                
                if current_node_id not in self.exec_out or 0 not in self.exec_out[current_node_id]:
                    raise RuntimeError(f"STATE node {current_node_id} is missing an outgoing execution edge.")
                next_node_id = self.exec_out[current_node_id][0]
                
            elif node.motif == MotifType.CONDITION:
                if current_node_id not in self.data_in or 0 not in self.data_in[current_node_id]:
                    raise RuntimeError(f"CONDITION node {current_node_id} is missing a condition data edge.")
                # Evaluate the boolean from incoming data
                condition_val = self._resolve_data(self.data_in[current_node_id][0])
                # Route execution: 0 for True path, 1 for False path
                path_index = 0 if condition_val else 1
                if current_node_id in self.exec_out and path_index in self.exec_out[current_node_id]:
                    next_node_id = self.exec_out[current_node_id][path_index]
                else:
                    # If path doesn't exist, exit gracefully
                    break
                    
            elif node.motif == MotifType.LOOP:
                if current_node_id not in self.data_in or 0 not in self.data_in[current_node_id]:
                    raise RuntimeError(f"LOOP node {current_node_id} is missing a condition data edge.")
                # A LOOP node evaluates condition, then routes to body (0) or exit (1)
                condition_val = self._resolve_data(self.data_in[current_node_id][0])
                path_index = 0 if condition_val else 1
                if current_node_id in self.exec_out:
                    next_node_id = self.exec_out[current_node_id].get(path_index)
                
            elif node.motif == MotifType.MESSAGE:
                # Execution path message (like print)
                if current_node_id in self.data_in:
                    args = [self._resolve_data(self.data_in[current_node_id][i]) for i in sorted(self.data_in[current_node_id].keys())]
                else:
                    args = []
                self._execute_message(node.literal_pointer, args)
                if current_node_id in self.exec_out:
                    next_node_id = self.exec_out[current_node_id].get(0)

            current_node_id = next_node_id
            
        return self.memory
