# Tasks: Fix Hierarchical Dataset String Nodes

## Phase 1: Preparation (5 min)

- [ ] **T1.1** Backup current dataset
  ```bash
  cp -r dataset/hierarchical dataset/hierarchical.backup
  cp -r models/hierarchical models/hierarchical.backup
  ```

- [ ] **T1.2** Verify current dataset statistics
  ```bash
  python3 -c "import sys; sys.path.insert(0, 'src'); from data_processing import *; # Count unknown nodes"
  ```

## Phase 2: Fix Dataset Creation Script (1 hour)

- [ ] **T2.1** Modify `build_partial_ast()` in `scripts/create_ast_level_dataset.py`
  - Add `isinstance(child, dict)` check before processing children
  - Add `'type' in child` check to filter non-AST nodes
  - Skip string children silently

- [ ] **T2.2** Test modification with sample data
  ```bash
  python3 scripts/create_ast_level_dataset.py --levels 1 --test-mode
  ```

- [ ] **T2.3** Verify unknown node percentage <5%
  ```python
  # Load generated test data and count node types
  ```

## Phase 3: Regenerate Full Dataset (15 min)

- [ ] **T3.1** Delete old hierarchical dataset
  ```bash
  rm -rf dataset/hierarchical/*.jsonl
  ```

- [ ] **T3.2** Regenerate all 20 levels
  ```bash
  python3 scripts/create_ast_level_dataset.py
  ```

- [ ] **T3.3** Verify dataset statistics
  - Check file sizes (should be ~50% smaller)
  - Check unknown node percentage across levels
  - Verify node count reduction

## Phase 4: Retrain Models (5-6 hours)

- [ ] **T4.1** Delete old trained models
  ```bash
  rm -rf models/hierarchical/*.pt
  ```

- [ ] **T4.2** Retrain all levels
  ```bash
  python3 train_hierarchical.py
  ```

- [ ] **T4.3** Monitor training metrics
  - Verify loss decreases at each level
  - Check type prediction diversity
  - Ensure no NaN/Inf losses

## Phase 5: Validation (10 min)

- [ ] **T5.1** Run hierarchical evaluation
  ```bash
  python3 generate_hierarchical.py --max-samples 50 --output hierarchical_eval_fixed.txt
  ```

- [ ] **T5.2** Check syntactic validity
  - Should be >0% (any improvement from current 0%)
  - Check for diverse node types (not 90%+ unknown)

- [ ] **T5.3** Analyze generated ASTs
  - Verify structural correctness
  - Check node type distribution
  - Compare with expected Ruby AST patterns

## Phase 6: Cleanup & Documentation (5 min)

- [ ] **T6.1** Document results
  - Record unknown node percentages (before/after)
  - Record syntactic validity (before/after)
  - Record training time and convergence

- [ ] **T6.2** Commit changes
  ```bash
  git add scripts/create_ast_level_dataset.py
  git commit -m "Fix hierarchical dataset: filter string nodes"
  ```

- [ ] **T6.3** Update change status
  ```bash
  openspec apply fix-hierarchical-dataset-strings
  ```

## Rollback Procedure

If validation fails (T5.2 shows no improvement):

```bash
# Restore backups
rm -rf dataset/hierarchical
mv dataset/hierarchical.backup dataset/hierarchical
rm -rf models/hierarchical
mv models/hierarchical.backup models/hierarchical

# Revert code
git checkout scripts/create_ast_level_dataset.py
```

## Success Criteria

- ✅ Unknown nodes <5% in new dataset (down from 50%)
- ✅ Model predicts diverse types (>20 unique types)
- ✅ Syntactic validity >0% on evaluation
- ✅ Generated ASTs have proper structure
- ✅ Training converges for all 20 levels

## Estimated Time

- Preparation: 5 min
- Code changes: 1 hour
- Dataset regeneration: 15 min
- Model retraining: 5-6 hours (mostly unattended)
- Validation: 10 min
- **Total: ~7 hours**
