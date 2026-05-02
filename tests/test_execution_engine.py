import pytest
import json
from src.execution_engine.schema import ExecutionGraph, Node, Edge, MotifType, EdgeType
from src.execution_engine.interpreter import GraphInterpreter

def test_fibonacci_execution():
    # Setup matching the demo graph
    pool = {0: "a", 1: "b", 2: "count", 3: "temp", 4: 0, 5: 1, 6: 10, 7: "+", 8: "<", 9: "print"}
    nodes = [
        Node(node_id=0, motif=MotifType.BOUNDARY),
        Node(node_id=1, motif=MotifType.MESSAGE, literal_pointer=4),
        Node(node_id=2, motif=MotifType.STATE, literal_pointer=0),
        Node(node_id=3, motif=MotifType.MESSAGE, literal_pointer=5),
        Node(node_id=4, motif=MotifType.STATE, literal_pointer=1),
        Node(node_id=5, motif=MotifType.STATE, literal_pointer=2),
        Node(node_id=6, motif=MotifType.STATE, literal_pointer=2),
        Node(node_id=7, motif=MotifType.MESSAGE, literal_pointer=6),
        Node(node_id=8, motif=MotifType.MESSAGE, literal_pointer=8),
        Node(node_id=9, motif=MotifType.LOOP),
        Node(node_id=10, motif=MotifType.STATE, literal_pointer=0),
        Node(node_id=11, motif=MotifType.STATE, literal_pointer=1),
        Node(node_id=12, motif=MotifType.MESSAGE, literal_pointer=7),
        Node(node_id=13, motif=MotifType.STATE, literal_pointer=3),
        Node(node_id=14, motif=MotifType.STATE, literal_pointer=1),
        Node(node_id=15, motif=MotifType.STATE, literal_pointer=0),
        Node(node_id=16, motif=MotifType.STATE, literal_pointer=3),
        Node(node_id=17, motif=MotifType.STATE, literal_pointer=1),
        Node(node_id=18, motif=MotifType.STATE, literal_pointer=2),
        Node(node_id=19, motif=MotifType.MESSAGE, literal_pointer=5),
        Node(node_id=20, motif=MotifType.MESSAGE, literal_pointer=7),
        Node(node_id=21, motif=MotifType.STATE, literal_pointer=2),
        Node(node_id=22, motif=MotifType.STATE, literal_pointer=0),
        Node(node_id=23, motif=MotifType.MESSAGE, literal_pointer=9),
        Node(node_id=24, motif=MotifType.BOUNDARY)
    ]
    edges = [
        Edge(source_node=0, target_node=2, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=2, target_node=4, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=4, target_node=5, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=5, target_node=9, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=1, target_node=2, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=3, target_node=4, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=1, target_node=5, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=6, target_node=8, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=7, target_node=8, edge_type=EdgeType.DATA, input_index=1),
        Edge(source_node=8, target_node=9, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=9, target_node=13, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=9, target_node=23, edge_type=EdgeType.EXECUTION, input_index=1),
        Edge(source_node=10, target_node=12, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=11, target_node=12, edge_type=EdgeType.DATA, input_index=1),
        Edge(source_node=12, target_node=13, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=14, target_node=15, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=16, target_node=17, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=18, target_node=20, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=19, target_node=20, edge_type=EdgeType.DATA, input_index=1),
        Edge(source_node=20, target_node=21, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=13, target_node=15, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=15, target_node=17, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=17, target_node=21, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=21, target_node=9, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=22, target_node=23, edge_type=EdgeType.DATA, input_index=0),
        Edge(source_node=23, target_node=24, edge_type=EdgeType.EXECUTION, input_index=0)
    ]
    graph = ExecutionGraph(nodes=nodes, edges=edges, literal_pool=pool)
    interpreter = GraphInterpreter(graph)
    memory = interpreter.run(max_steps=1000)
    assert memory["a"] == 55
    
def test_interpreter_cyclic_data_guard():
    # Node 0 is a MESSAGE that needs input from Node 0 (itself)
    pool = {0: "add"}
    nodes = [
        Node(node_id=0, motif=MotifType.MESSAGE, literal_pointer=0)
    ]
    edges = [
        Edge(source_node=0, target_node=0, edge_type=EdgeType.DATA, input_index=0)
    ]
    graph = ExecutionGraph(nodes=nodes, edges=edges, literal_pool=pool)
    interpreter = GraphInterpreter(graph)
    with pytest.raises(RuntimeError, match="Cyclic data dependency"):
        interpreter._resolve_data(0)

def test_interpreter_infinite_loop_guard():
    pool = {0: True}
    nodes = [
        Node(node_id=0, motif=MotifType.BOUNDARY),
        Node(node_id=1, motif=MotifType.LOOP),
        Node(node_id=2, motif=MotifType.MESSAGE, literal_pointer=0) # True provider
    ]
    edges = [
        Edge(source_node=0, target_node=1, edge_type=EdgeType.EXECUTION, input_index=0),
        Edge(source_node=2, target_node=1, edge_type=EdgeType.DATA, input_index=0),
        # Loop routes to itself on True
        Edge(source_node=1, target_node=1, edge_type=EdgeType.EXECUTION, input_index=0)
    ]
    graph = ExecutionGraph(nodes=nodes, edges=edges, literal_pool=pool)
    interpreter = GraphInterpreter(graph)
    with pytest.raises(RuntimeError, match="Max execution steps"):
        interpreter.run(max_steps=50)
