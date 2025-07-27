#!/usr/bin/env python3
"""
Test for the extract_sample_graph function fix.

This test validates that extract_sample_graph correctly extracts 
individual sample graphs from batched graph data.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from train_autoregressive import extract_sample_graph


def test_basic_extraction():
    """Test basic graph extraction with simple batched data."""
    print("Testing basic extraction...")
    
    mock_partial_graphs = {
        'x': [[1.0] * 74, [2.0] * 74, [3.0] * 74, [4.0] * 74],
        'edge_index': [[0, 1, 2], [1, 2, 3]],  # edges: 0->1, 1->2, 2->3
        'batch': [0, 0, 1, 1],  # first 2 nodes in sample 0, last 2 in sample 1
        'num_graphs': 2
    }
    
    # Test sample 0
    result_0 = extract_sample_graph(mock_partial_graphs, 0, 2)
    assert len(result_0['x']) == 2, f"Sample 0 should have 2 nodes, got {len(result_0['x'])}"
    assert result_0['x'][0][0] == 1.0, "Sample 0 first node should have feature value 1.0"
    assert result_0['x'][1][0] == 2.0, "Sample 0 second node should have feature value 2.0"
    assert result_0['edge_index'] == [[0], [1]], f"Sample 0 should have edge 0->1, got {result_0['edge_index']}"
    assert result_0['batch'] == [0, 0], f"Sample 0 batch should be [0, 0], got {result_0['batch']}"
    
    # Test sample 1
    result_1 = extract_sample_graph(mock_partial_graphs, 1, 2)
    assert len(result_1['x']) == 2, f"Sample 1 should have 2 nodes, got {len(result_1['x'])}"
    assert result_1['x'][0][0] == 3.0, "Sample 1 first node should have feature value 3.0"
    assert result_1['x'][1][0] == 4.0, "Sample 1 second node should have feature value 4.0"
    assert result_1['edge_index'] == [[0], [1]], f"Sample 1 should have edge 0->1, got {result_1['edge_index']}"
    assert result_1['batch'] == [0, 0], f"Sample 1 batch should be [0, 0], got {result_1['batch']}"
    
    print("✅ Basic extraction test passed")
    return True


def test_complex_edge_patterns():
    """Test with more complex edge patterns and cross-sample edges."""
    print("Testing complex edge patterns...")
    
    mock_partial_graphs = {
        'x': [[1.0] * 74, [2.0] * 74, [3.0] * 74, [4.0] * 74, [5.0] * 74],
        'edge_index': [[0, 1, 0, 2, 3, 4], [1, 0, 2, 4, 4, 3]],  # complex pattern
        'batch': [0, 0, 0, 1, 1],  # first 3 nodes in sample 0, last 2 in sample 1
        'num_graphs': 2
    }
    
    # Test sample 0 (should get internal edges only)
    result_0 = extract_sample_graph(mock_partial_graphs, 0, 2)
    assert len(result_0['x']) == 3, f"Sample 0 should have 3 nodes, got {len(result_0['x'])}"
    expected_edges_0 = [[0, 1, 0], [1, 0, 2]]  # edges 0->1, 1->0, 0->2
    assert result_0['edge_index'] == expected_edges_0, f"Sample 0 edges incorrect: {result_0['edge_index']}"
    
    # Test sample 1 (should get internal edges with remapped indices)
    result_1 = extract_sample_graph(mock_partial_graphs, 1, 2)
    assert len(result_1['x']) == 2, f"Sample 1 should have 2 nodes, got {len(result_1['x'])}"
    expected_edges_1 = [[0, 1], [1, 0]]  # edges 3->4, 4->3 remapped to 0->1, 1->0
    assert result_1['edge_index'] == expected_edges_1, f"Sample 1 edges incorrect: {result_1['edge_index']}"
    
    print("✅ Complex edge patterns test passed")
    return True


def test_edge_cases():
    """Test edge cases like empty graphs, single nodes, etc."""
    print("Testing edge cases...")
    
    # Test empty graph
    empty_graphs = {'x': [], 'edge_index': [[], []], 'batch': []}
    result_empty = extract_sample_graph(empty_graphs, 0, 1)
    assert result_empty == {'x': [], 'edge_index': [[], []], 'batch': []}, "Empty graph test failed"
    
    # Test single node, no edges
    single_node = {
        'x': [[1.0] * 74],
        'edge_index': [[], []],
        'batch': [0],
        'num_graphs': 1
    }
    result_single = extract_sample_graph(single_node, 0, 1)
    assert len(result_single['x']) == 1, "Single node should be extracted"
    assert result_single['edge_index'] == [[], []], "Single node should have no edges"
    assert result_single['batch'] == [0], "Single node batch should be [0]"
    
    # Test sample that doesn't exist
    result_nonexistent = extract_sample_graph(single_node, 1, 2)
    assert result_nonexistent == {'x': [], 'edge_index': [[], []], 'batch': []}, "Non-existent sample should return empty"
    
    print("✅ Edge cases test passed")
    return True


def test_no_batch_info():
    """Test fallback behavior when batch info is missing."""
    print("Testing fallback for missing batch info...")
    
    graphs_no_batch = {
        'x': [[1.0] * 74, [2.0] * 74],
        'edge_index': [[0], [1]],
        # No 'batch' key
    }
    
    result = extract_sample_graph(graphs_no_batch, 0, 1)
    assert len(result['x']) == 2, "Should return all nodes when no batch info"
    assert result['edge_index'] == [[0], [1]], "Should return all edges when no batch info"
    assert result['batch'] == [0, 0], "Should create batch indices when missing"
    
    print("✅ Missing batch info test passed")
    return True


def test_isolated_nodes():
    """Test extraction with isolated nodes (no edges)."""
    print("Testing isolated nodes...")
    
    mock_partial_graphs = {
        'x': [[1.0] * 74, [2.0] * 74, [3.0] * 74],
        'edge_index': [[], []],  # No edges
        'batch': [0, 1, 1],  # sample 0 has 1 node, sample 1 has 2 nodes
        'num_graphs': 2
    }
    
    # Test sample 0 (1 isolated node)
    result_0 = extract_sample_graph(mock_partial_graphs, 0, 2)
    assert len(result_0['x']) == 1, "Sample 0 should have 1 node"
    assert result_0['edge_index'] == [[], []], "Sample 0 should have no edges"
    
    # Test sample 1 (2 isolated nodes)
    result_1 = extract_sample_graph(mock_partial_graphs, 1, 2)
    assert len(result_1['x']) == 2, "Sample 1 should have 2 nodes"
    assert result_1['edge_index'] == [[], []], "Sample 1 should have no edges"
    
    print("✅ Isolated nodes test passed")
    return True


def main():
    """Run all tests."""
    print("🧪 Testing extract_sample_graph function fix")
    print("=" * 50)
    
    tests = [
        test_basic_extraction,
        test_complex_edge_patterns,
        test_edge_cases,
        test_no_batch_info,
        test_isolated_nodes,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! extract_sample_graph function is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)