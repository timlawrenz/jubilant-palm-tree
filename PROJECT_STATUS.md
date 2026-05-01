# Project Status Tracker

**Last Updated**: 2025-11-20 13:06 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 5 - NEURAL UNIVERSAL MACHINE MVP PROVEN 🚀

### What's Happening Right Now?
The research has successfully pivoted away from human-readable syntax to generating pure topological execution graphs. We implemented a 6-Motif Universal schema, compressed the Ruby AST dataset, and trained a Permuted Dense DiT using Optimal Transport Flow Matching.

### Training Progress & Findings
- **Core Hypothesis Proven**: The DiT natively learned to route Directed Acyclic logic graphs (achieving near 100% pathing/acyclic pass rates out of pure noise).
- **The Sticking Point**: Syntactic Validity Rate (SVR) is currently bottlenecked by the "Continuous Index Channel." The model successfully draws the general graph topology, but suffers index collisions because it is trying to predict discrete branching paths (e.g., True vs False edges) using a continuous MSE regression target.

### Next Immediate Action
- **Architectural Refinement**: Transition the DiT's `input_index` channel from continuous MSE regression to a discrete Categorical Classification head (CrossEntropyLoss) to natively enforce mutually exclusive edge routing without rounding collisions.

---

## 📋 NEXT ACTION CHECKLIST

### 🤔 DECISION POINT: HIERARCHICAL APPROACH FAILED

The hierarchical decoder has fundamental issues:
1. **Trained 100%** but achieves **0% validity**
2. Generates repetitive, nonsensical patterns
3. Cannot capture semantic meaning from text alone
4. Architecture may be fundamentally flawed for this task

### ⚡ OPTIONS MOVING FORWARD:

**OPTION A: Pivot to Autoregressive Decoder (RECOMMENDED)**
```bash
# Start Phase 7: Sequence-based autoregressive generation
# See docs/README_phase7.md
python train_autoregressive.py

# WHY: Transformer-based autoregressive models have proven success in code generation
# and may work better than the hierarchical graph-based approach
```

**OPTION B: Debug Hierarchical Architecture**
Investigate why the model can't learn:
- Add attention mechanisms
- Try different loss functions
- Increase model capacity
- Use pre-trained code embeddings

**WHY**: Maybe fixable with architecture changes, but uncertain payoff.

**OPTION C: Simplify the Problem**
Train on trivial code patterns first:
- Only generate simple expressions (no functions/classes)
- Limit vocabulary to 10 most common node types
- Start with depth-1 trees only

**WHY**: Validate if approach can work on easier tasks before scaling up.

**OPTION D: Use Pre-trained Models**
Fine-tune existing models like CodeT5, CodeGen:
- Leverage pre-training on massive code corpuses
- Skip custom architecture development
- Focus on Ruby-specific fine-tuning

**WHY**: Fastest path to working solution, but less research novelty.

### ✅ IF YOU WANT TO ANALYZE TRAINING:
```bash
# Check if training is actually learning anything
python -c "
import torch
import matplotlib.pyplot as plt

# Load and examine model weights to see if they're changing
for i in range(5):
    model = torch.load(f'models/hierarchical/hierarchical_decoder_level_{i}.pt', map_location='cpu')
    print(f'Level {i} - Params: {sum(p.numel() for p in model.values())}')
"

# TODO: Add training loss logging to train_hierarchical.py to track progress
```

**WHY**: Understand if the model is learning or if architecture needs fixing.

### ✅ IF YOU WANT TO PIVOT STRATEGY:
See "Strategic Options" section below.

---

## 🗺️ PROJECT PHASE MAP

### Phase 1-3: Data & Complexity Prediction ✅ **COMPLETE & WORKING**
- **Status**: Fully functional
- **Performance**: 4.64 MAE (46.6% better than baseline)
- **Models**: `models/best_model.pt`
- **Next Action**: None needed - this is solid

### Phase 4: Original Autoencoder ❌ **FAILED & ABANDONED**
- **Status**: Non-functional (0% validity)
- **Next Action**: Superseded by Phase 4b

### Phase 4b: Hierarchical Decoder ❌ **TRAINED BUT NON-FUNCTIONAL**
- **Status**: Training complete (20/20 levels), 0% validity
- **Performance**: Generates nonsensical repetitive patterns
- **Models**: `models/hierarchical/hierarchical_decoder_level_[0-19].pt`
- **Issue**: Architecture cannot learn semantic code generation from text embeddings
- **Next Action**: ⚡ **PIVOT TO ALTERNATIVE APPROACH (Phase 7)** ⚡

### Phase 5: Text-Code Alignment ❌ **COMPLETE BUT NON-FUNCTIONAL**
- **Status**: Trained but performs at random levels
- **Performance**: 0.34% Recall@10 (near random)
- **Models**: `models/best_alignment_model.pt`
- **Next Action**: Needs architectural rethink OR wait for Phase 4b to work first

### Phase 6: Text-to-Code Generation ❌ **BLOCKED BY PHASES 4B & 5**
- **Status**: Non-functional (depends on broken upstream models)
- **Performance**: Generates trivial code
- **Next Action**: Wait for Phase 4b and Phase 5 to work

### Phase 7: Autoregressive Decoder 📋 **PLANNED**
- **Status**: Not started
- **Next Action**: Don't start until Phase 4b evaluation is complete

---

## 🎓 DECISION GUIDE

### Question 1: "Should I continue training the hierarchical decoder?"

**YES**, if:
- ✅ You believe more levels might help (current: 5/20)
- ✅ You have GPU time available (overnight training)
- ✅ You want to give the hierarchical approach a fair shot

**NO**, if:
- ❌ You think the 0% validity indicates a fundamental architecture problem
- ❌ You want to analyze current results before proceeding
- ❌ You want to try a different approach

**RECOMMENDED**: ✅ **Continue training to at least 10-12 levels**, then re-validate. The hierarchical approach may need more depth to work.

### Question 2: "What if training to 20 levels still shows 0% validity?"

Then the hierarchical approach has fundamental issues. Options:
1. **Debug the architecture**: Add attention, better loss functions, etc.
2. **Pivot to Phase 7**: Try the autoregressive decoder approach
3. **Simplify the problem**: Train on very simple code patterns first
4. **Use pre-trained models**: Fine-tune existing code generation models

### Question 3: "Should I work on Phase 5 (alignment) instead?"

**NO** - Phase 5 is blocked by Phase 4b. Even if alignment worked perfectly, it needs a working decoder. Fix Phase 4b first.

---

## 🔄 TYPICAL WORKFLOW CYCLE

```
┌─────────────────────────────────────────────────────┐
│ 1. OPEN PROJECT                                     │
│    → Read this file (PROJECT_STATUS.md)             │
│    → Check "NEXT ACTION CHECKLIST" section          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 2. TRAINING PHASE                                   │
│    → Run training script overnight                  │
│    → Update checkpoint state                        │
│    → Update this file with new progress             │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 3. VALIDATION PHASE                                 │
│    → Run validation scripts                         │
│    → Generate VALIDATION_RESULTS_YYYY-MM-DD.md     │
│    → Decide: continue, pivot, or debug?             │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ 4. DECISION POINT                                   │
│    → If good results: Move to next phase            │
│    → If poor results: Analyze, debug, or pivot      │
│    → Update this file with decision                 │
└─────────────────────────────────────────────────────┘
```

---

## 📊 TRAINING CONTINUATION GUIDE

### To Resume Hierarchical Training:

```bash
# 1. Activate environment
source venv/bin/activate  # or your venv path

# 2. Optional: Check GPU availability
nvidia-smi

# 3. Resume training (will auto-detect checkpoint)
python train_hierarchical.py \
    --max-levels 20 \
    --epochs-per-level 5 \
    --batch-size 16 \
    --learning-rate 1e-4

# Training will:
# - Resume from level 5 (reads checkpoint automatically)
# - Train levels 5 through 19
# - Save each level to models/hierarchical/
# - Update training_checkpoint.pt after each level
```

### Training Monitoring:
```bash
# Watch GPU usage in another terminal
watch -n 1 nvidia-smi

# Check progress (training saves after each level)
ls -lah models/hierarchical/*.pt

# After training completes, validate:
python scripts/evaluate_hierarchical_model.py --num_samples 100
```

---

## 🎯 IMMEDIATE RECOMMENDED ACTION (RIGHT NOW)

Based on validation results (0% validity after full training), here's what to do:

### **OPTION A: Pivot to Autoregressive Approach (RECOMMENDED)**
The hierarchical approach has failed after full training. Time to try a proven architecture.

```bash
# Check if autoregressive trainer exists
ls train_autoregressive.py

# If it exists, start training
python train_autoregressive.py --epochs 10 --batch-size 16

# If not, need to implement Phase 7
# See: docs/README_phase7.md or start with existing transformer implementations
```

**Why recommended**: Autoregressive transformers are proven for code generation (GPT, CodeT5, etc.) and may work where hierarchical GNNs failed.

### **OPTION B: Analyze Training Failure**
Understand why hierarchical approach completely failed despite full training.

```bash
# Check if loss actually decreased during training
python -c "
import re
import matplotlib.pyplot as plt

with open('training_log.txt') as f:
    content = f.read()
    
# Extract losses
losses = re.findall(r'Avg\. Loss: ([-\d.]+)', content)
losses = [float(l) for l in losses]

plt.plot(losses)
plt.xlabel('Level × Epoch')
plt.ylabel('Loss')
plt.title('Hierarchical Decoder Training Loss')
plt.savefig('hierarchical_training_loss.png')
print(f'Loss progression: {losses[:5]} ... {losses[-5:]}')
"
```

**Next step**: If loss didn't decrease → broken architecture. If it did → issue is elsewhere.

### **OPTION C: Document and Move On**
Accept that this approach didn't work and document findings.

Create a validation summary, then proceed to next phase or different project.

---

## 📝 WHEN TO UPDATE THIS FILE

Update the "CURRENT STATUS" and "NEXT ACTION" sections when:
- ✅ You complete a training session
- ✅ You run a validation
- ✅ You make a strategic decision to pivot
- ✅ You start a new phase

This keeps the file as a reliable "project memory" for quick orientation.

---

## 📚 QUICK REFERENCE

### Key Files
- **This file**: Overall project status and next actions
- **VALIDATION_RESULTS_YYYY-MM-DD.md**: Detailed validation results
- **docs/README_phase*.md**: Phase-specific documentation
- **docs/README_assessment.md**: Original assessment methodology

### Key Commands
```bash
# Training
python train_hierarchical.py --max-levels 20

# Validation
python scripts/evaluate_hierarchical_model.py --num_samples 100
python validate_alignment_model.py
python validate_complexity_model.py

# Generation test
python generate_code.py "a method that adds two numbers"
```

### Key Questions to Ask AI Assistant
- "What phase am I in?" → Check "CURRENT STATUS" section above
- "What should I do next?" → Check "NEXT ACTION CHECKLIST" section
- "Is training complete?" → Check "Training Progress" section
- "What are my options?" → Check "DECISION GUIDE" section

---

**Remember**: This project has a solid foundation (Phases 1-3 work great!). The generative parts (4b, 5, 6) are the challenge. Take it one step at a time.
