import pytest
from src.execution_engine.schema import MotifType
from scripts.dataset_prep.compress_ast import ASTCompressor

def test_ast_compression_extracts_motifs_and_literals():
    compressor = ASTCompressor()
    
    # Mocking a basic ruby AST chunk:  storage = empty?
    ast_json = {
        "type": "lvasgn",
        "children": [
            "storage",
            {
                "type": "send",
                "children": [None, "empty?"]
            }
        ]
    }
    
    root_id = compressor.process_ast(ast_json)
    
    # Nodes should be processed
    assert len(compressor.nodes) == 2
    # The literal pool should contain 'storage' and 'empty?'
    assert "storage" in compressor.literal_pool.values()
    assert "empty?" in compressor.literal_pool.values()
    
    # The root node (lvasgn) is MotifType.STATE
    root_node = next(n for n in compressor.nodes if n.node_id == root_id)
    assert root_node.motif == MotifType.STATE
    assert compressor.literal_pool[root_node.literal_pointer] == "storage"
    
    # Check edges
    assert len(compressor.edges) == 1
    edge = compressor.edges[0]
    assert edge.target_node == root_id
    assert edge.edge_type == 1 # DATA edge
