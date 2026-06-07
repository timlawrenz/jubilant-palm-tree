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

        # Fallback for unexpected data sources
        resolving_set.remove(node_id)
        return None

    def _execute_message(self, literal_pointer: int, args: list):
        """Executes an external function or math operation."""
        op = self.graph.literal_pool[literal_pointer]
        if op == "safe_op":
            return 1
        
        # Original functionality would go here (e.g. looking up a function by name)
        raise NotImplementedError(f"Operation {op} not implemented in MVP interpreter.")

    def run_with_limit(self, max_steps: int = 1000):
        """
        Executes the graph with an instruction limit to prevent infinite loops.
        Returns a dict with 'status' and 'steps' taken.
        """
        # Find entry boundary: BOUNDARY motif with NO incoming EXECUTION edges
        entry_nodes = [
            n_id for n_id, n in self.nodes.items() 
            if n.motif == MotifType.BOUNDARY and n_id not in self.exec_out and not any(n_id == tgt for edges in self.exec_out.values() for tgt in edges.values())
        ]
        
        if not entry_nodes:
            return {"status": "error_no_entry", "steps": 0}
            
        current = entry_nodes[0]
        steps = 0
        
        while current is not None:
            if steps >= max_steps:
                return {"status": "infinite_loop", "steps": steps}
                
            steps += 1
            node = self.nodes[current]
            
            # If we hit a boundary node AFTER the first step, it's an exit. We halted!
            if node.motif == MotifType.BOUNDARY and steps > 1:
                return {"status": "halted", "steps": steps}
                
            out_edges = self.exec_out.get(current, {})
            if not out_edges:
                return {"status": "error_unexpected_sink", "steps": steps}
                
            if node.motif in (MotifType.CONDITION, MotifType.LOOP):
                # Dummy evaluation: alternate branches based on step count to avoid getting stuck
                # while exploring the topological routing
                idx = steps % 2 if len(out_edges) > 1 else list(out_edges.keys())[0]
                current = out_edges.get(idx)
            else:
                # Sequence, State, Message -> follow the single exec out edge
                current = out_edges.get(0, list(out_edges.values())[0] if out_edges else None)

        return {"status": "halted", "steps": steps}

    def run(self, max_steps: int = 10000):
        # Find the starting BOUNDARY node
        start_nodes = [n.node_id for n in self.graph.nodes if n.motif == MotifType.BOUNDARY]
        if not start_nodes:
            raise ValueError("Graph has no BOUNDARY node to start execution.")
            
        current = start_nodes[0]
        step_count = 0
        
        while current is not None:
            if step_count > max_steps:
                raise RuntimeError(f"Execution exceeded max steps ({max_steps})")
                
            node = self.nodes[current]
            
            # If we hit another boundary, execution ends
            if node.motif == MotifType.BOUNDARY and step_count > 0:
                break
                
            out_edges = self.exec_out.get(current, {})
            
            if node.motif == MotifType.CONDITION:
                if 0 not in out_edges or 1 not in out_edges:
                    raise RuntimeError(f"Condition node {current} missing True(0) or False(1) exec edges.")
                # Needs 1 data input (the condition to evaluate)
                cond_val = self._resolve_data(current)
                next_node = out_edges[0] if cond_val else out_edges[1]
                
            elif node.motif == MotifType.LOOP:
                if 0 not in out_edges or 1 not in out_edges:
                    raise RuntimeError(f"Loop node {current} missing Loop(0) or Exit(1) exec edges.")
                cond_val = self._resolve_data(current)
                next_node = out_edges[0] if cond_val else out_edges[1]
                
            elif node.motif == MotifType.STATE:
                # State writes (if it has exec out) evaluate their data_in and update memory
                if len(out_edges) > 0:
                    val = self._resolve_data(current)
                    if node.literal_pointer is not None:
                        var_name = str(self.graph.literal_pool[node.literal_pointer])
                        self.memory[var_name] = val
                    next_node = out_edges.get(0, None)
                else:
                    next_node = None
                    
            elif node.motif in (MotifType.SEQUENCE, MotifType.MESSAGE, MotifType.BOUNDARY):
                if node.motif == MotifType.MESSAGE:
                    # Execute the message for side effects (e.g. print)
                    args = [self._resolve_data(self.data_in.get(current, {})[i]) for i in sorted(self.data_in.get(current, {}).keys())]
                    if node.literal_pointer is not None:
                        self._execute_message(node.literal_pointer, args)
                    
                next_node = out_edges.get(0, None)
                
            else:
                raise NotImplementedError(f"Execution logic for {node.motif} not implemented")
                
            current = next_node
            step_count += 1
            
        return self.memory
