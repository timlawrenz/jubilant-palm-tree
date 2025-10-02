# Memory Optimization Guide for evaluate_autoencoder_consolidated.ipynb

## Overview

The `evaluate_autoencoder_consolidated.ipynb` notebook has been enhanced with memory optimization features to handle large datasets without running out of memory. This guide explains how to use these features effectively.

## Problem Solved

**Before**: The notebook would run out of memory when evaluating large datasets because:
- All evaluation results were stored in memory simultaneously
- JSONL file was read completely for each sample
- No memory cleanup between samples
- No batch processing to control memory usage

**After**: Memory-efficient processing that:
- Processes samples in configurable batches
- Caches JSONL data efficiently
- Monitors and cleans up memory automatically
- Provides streaming evaluation for large datasets

## New Configuration Options

Add these to your `CONFIG` dictionary in the notebook:

```python
CONFIG = {
    # Existing options...
    'num_samples': 1000,
    'enable_ruby_conversion': True,
    
    # New memory optimization options
    'batch_size': 50,                    # Process samples in batches
    'enable_memory_optimization': True,   # Enable all optimizations
    'max_memory_mb': 2048,               # Memory limit before cleanup
    'cache_jsonl_data': True,            # Cache JSONL for faster access
    'save_intermediate_results': False,   # Save results between batches
}
```

## Configuration Guidelines

### Batch Size (`batch_size`)
- **Small datasets (< 100 samples)**: Use batch_size = 10-20
- **Medium datasets (100-1000 samples)**: Use batch_size = 50-100  
- **Large datasets (> 1000 samples)**: Use batch_size = 100-200
- **Very large datasets (> 5000 samples)**: Use batch_size = 200-500

**Memory impact**: Larger batches use more memory but have less overhead.

### Memory Limit (`max_memory_mb`)
- **8GB RAM systems**: Set to 4000-6000 MB
- **16GB RAM systems**: Set to 8000-12000 MB
- **32GB+ RAM systems**: Set to 16000+ MB

**Note**: Leave room for system processes and other applications.

### JSONL Caching (`cache_jsonl_data`)
- **True**: Faster processing, uses more memory upfront
- **False**: Slower processing, uses less memory

**Recommendation**: Use `True` unless you have severe memory constraints.

## Memory Usage Examples

### Example 1: Small Evaluation (< 100 samples)
```python
CONFIG = {
    'num_samples': 50,
    'batch_size': 10,
    'enable_memory_optimization': True,
    'max_memory_mb': 2048,
    'cache_jsonl_data': True,
    'enable_ruby_conversion': True,
}
```

### Example 2: Medium Evaluation (100-1000 samples)
```python
CONFIG = {
    'num_samples': 500,
    'batch_size': 50,
    'enable_memory_optimization': True,
    'max_memory_mb': 4096,
    'cache_jsonl_data': True,
    'enable_ruby_conversion': True,
}
```

### Example 3: Large Evaluation (> 1000 samples)
```python
CONFIG = {
    'num_samples': 2000,
    'batch_size': 100,
    'enable_memory_optimization': True,
    'max_memory_mb': 8192,
    'cache_jsonl_data': True,
    'enable_ruby_conversion': False,  # Disable for speed
    'save_intermediate_results': True,
}
```

### Example 4: Memory-Constrained System
```python
CONFIG = {
    'num_samples': 1000,
    'batch_size': 20,               # Smaller batches
    'enable_memory_optimization': True,
    'max_memory_mb': 1024,          # Lower limit
    'cache_jsonl_data': False,      # Disable caching
    'enable_ruby_conversion': False, # Disable for memory savings
}
```

## Monitoring Memory Usage

The optimized notebook will display:
- Initial memory usage
- Memory usage per batch
- Peak memory during processing
- Memory savings achieved

Example output:
```
Configuration:
  batch_size: 50
  enable_memory_optimization: True
  max_memory_mb: 2048

Initial memory usage: 856.2 MB

Processing batch 1/20 (samples 1-50)
Memory usage high (2100.5 MB), cleaning up...
Batch 1 complete, memory usage: 1204.7 MB

Processing batch 2/20 (samples 51-100)
Batch 2 complete, memory usage: 1456.3 MB
```

## Troubleshooting

### Still Running Out of Memory?
1. **Reduce batch size**: Try half the current value
2. **Disable JSONL caching**: Set `cache_jsonl_data: False`
3. **Lower memory limit**: Reduce `max_memory_mb` to trigger cleanup earlier
4. **Disable Ruby conversion**: Set `enable_ruby_conversion: False`
5. **Enable intermediate saving**: Set `save_intermediate_results: True`

### Processing Too Slow?
1. **Increase batch size**: Double the current value (if memory allows)
2. **Enable JSONL caching**: Set `cache_jsonl_data: True`
3. **Disable Ruby conversion**: Set `enable_ruby_conversion: False`
4. **Increase memory limit**: Set higher `max_memory_mb`

### Inconsistent Results?
1. **Check sample selection**: Random seed should be consistent
2. **Verify batch processing**: Results should be identical regardless of batch size
3. **Test with small dataset**: Validate with known samples first

## Advanced Features

### Intermediate Result Saving
For very large evaluations, enable intermediate saving:
```python
CONFIG['save_intermediate_results'] = True
```

This saves results after each batch to `../output/intermediate_results_batch_N.json`.

### Fallback Mode
If optimization fails, the notebook automatically falls back to the original implementation:
```python
CONFIG['enable_memory_optimization'] = False
```

### Custom Memory Monitoring
Monitor memory usage in real-time:
```python
print(f"Current memory: {get_memory_usage_mb():.1f} MB")
cleanup_memory()  # Force cleanup
```

## Performance Comparison

| Method | Memory Usage | Processing Speed | Scalability |
|--------|-------------|------------------|-------------|
| Original | High (grows linearly) | Fast | Limited by RAM |
| Optimized | Low (constant per batch) | Fast | Unlimited |

## Best Practices

1. **Start small**: Test with small datasets first
2. **Monitor memory**: Watch the memory usage output
3. **Tune batch size**: Find the optimal balance for your system
4. **Save results**: Enable intermediate saving for large evaluations
5. **Use caching**: Enable JSONL caching unless memory-constrained
6. **Disable Ruby**: Turn off Ruby conversion for faster processing

## Support

If you encounter issues:
1. Check the configuration examples above
2. Review the troubleshooting section
3. Run the test scripts: `python test_memory_optimization.py`
4. Use the demo: `python demo_memory_optimization.py`

The memory optimization maintains full backward compatibility - existing notebooks will work unchanged with the new features available as opt-in enhancements.