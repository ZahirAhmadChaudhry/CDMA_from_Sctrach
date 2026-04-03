# Module 4 Implementation Report (Chapter 7 End-to-End CDMA)

## Scope and Current Status
This report documents how Module 4 is implemented in code and how it maps to the thesis specification for Chapter 7.

Current status:
- Implementation complete.
- Local training/evaluation runs were intentionally not executed, per instruction.
- Kaggle GPU run is the intended validation path.

Main implementation files:
- src/cdma/module4_cdma.py
- run_module4.py

## Thesis-to-Code Mapping

### 1) IT-MLA stage (Eq 6.1 / 6.2)
Implemented in ITMLALayer.

Code behavior:
- Computes frame mean a_k for each frame.
- Computes cosine similarity between each x_i and a_k.
- Applies x_i* = x_i * (1 + cos(theta_i)).

Status: aligned.

### 2) LSTM1 stage (frame-level)
Implemented in LSTM1Layer and CDMAModel._encode_stream.

Code behavior:
- LSTM over 128 vectors in each frame.
- Mean pooling over hidden states to context vector.
- Sigmoid(Linear(context)) gives per-frame probability.
- For Module 4, p_c and p_o are differentiable masked means over frame probabilities.

Status: aligned with Module 4 requirement (differentiable p_c/p_o).

### 3) CT-GA stage (Eq 7.2 / 7.3) + self-GA (Eq 7.8 / 7.9)
Implemented in CTGALayer.

Code behavior:
- Cross-GA: attends RT sequence using IT mean, and IT sequence using RT mean.
- Self-GA: attends each stream with its own mean.
- Attention weight uses softmax over cosine similarities.
- Uses sequence masks for variable-length validity.

Status: aligned.

### 4) LSTM2 stage (sequence-level)
Implemented in LSTM2Layer.

Code behavior:
- Uses pack_padded_sequence for variable-length participant sequences.
- Unpacks outputs and applies masked mean pooling.
- Sigmoid(Linear(pooled)) gives p_t or p_d.

Status: aligned.

### 5) CTF stage (Eq 7.4 / 7.5)
Implemented in CTFLayer.

Code behavior:
- f = r + s (pre-attention fusion) -> p_f1.
- f* = r* + s* (post-attention fusion) -> p_f2.

Status: aligned.

### 6) Aggregation (Eq 7.6)
Implemented in CDMAModel.forward.

Code behavior:
- p_hat = mean(active probabilities in mode output set).
- Final prediction at evaluation: p_hat > 0.5.

Status: aligned.

### 7) Loss (Eq 7.7)
Implemented in CombinedBCELoss.

Code behavior:
- BCE with reduction="mean" applied per output probability.
- Sum over all active outputs for the mode.
- This preserves the thesis requirement: sum over outputs, not average over outputs.

Status: aligned.

## 13-Mode Routing Coverage
Implemented through MODE_CONFIGS in module4_cdma.py.

Modes covered:
- ba1_rt, ba1_it
- itmla_rt, itmla_it
- ba2_rt, ba2_it
- ba3_rt, ba3_it
- ctga_rt, ctga_it
- ba4, ba5, full_cdma

Important detail:
- ba2_rt and ba3_rt are RT-only modes (`need_it=False`) and do not require IT input.
- ba2_it and ba3_it are IT-only modes (`need_rt=False`) and do not require RT input.
- ctga_rt and ctga_it require both streams for cross attention and are implemented that way.
- They report only the mode-defined outputs (p_c,p_t for ctga_rt; p_o,p_d for ctga_it).

Status: aligned.

## Training and Evaluation Protocol in Code

### Data and folds
Module 4 uses get_dataloaders from Module 2.

Current project decision retained:
- RT fold assignment is used as the split backbone.
- Participant set is the 110 with both streams (via filtered split).

### Optimization
- Optimizer: RMSProp
- Learning rate: 1e-3
- Batch size: 16
- Epochs default: 300
- Threshold: 0.5
- Seed convention: rep * 42

Status: aligned with project spec.

## Added Reliability Features (Engineering Layer)
These are implementation safety features, not thesis architecture changes.

Implemented:
- Checkpoint save every 50 epochs (configurable) and final epoch.
- Resume from latest fold checkpoint if interrupted.
- Skip completed folds when prediction/report outputs already exist.
- Persistent history append files:
  - results/module4/fold_history.csv
  - results/module4/pooled_history.csv
- Immediate per-fold output persistence:
  - fold report
  - fold predictions CSV
- Terminal output control:
  - loss logs at epoch 1, every N epochs, and final epoch
  - preview first K participants after each fold

Status: implemented as requested for long Kaggle runs.

## Built-In Sanity Checks (No full training required)
Implemented in run_sanity_checks.

Checks included:
- Forward pass succeeds for all 13 modes on dummy data.
- Each mode output keys match expected mode outputs.
- p_hat shape is (B, 1).
- No NaN values in output probabilities.
- Loss scaling check (|B|=6 output mode loss > |B|=1 output mode loss).
- Gradient flow check (parameter update after one optimization step).

Status: implemented.

## Alignment Summary

Fully aligned to thesis/module spec:
- IT-MLA, LSTM1, CT-GA/self-GA, LSTM2, CTF, p_hat aggregation, Eq 7.7 loss behavior.
- 13-mode routing logic and ctga two-stream dependency.

Project-specific and intentional:
- Split protocol uses RT-fold backbone from Module 2 for all Module 4 runs.
- Added checkpoint/resume/history/logging controls for robust long experiments.

Pending empirical validation on Kaggle:
- Full 13 modes x all folds x reps with 300 epochs.
- Relative ordering and pooled metric behavior under k=5 protocol.

## Recommended Kaggle Execution Plan
1. Run a small subset first (for example: ba1_rt, itmla_rt) with 1 rep.
2. Confirm outputs are being written under results/module4 and checkpoints are created.
3. Expand to all 13 modes and target reps.
4. Use appended history CSV files to monitor progress without losing prior results.
