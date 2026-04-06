# CDMA Replication: Independent Verification Protocol

## Purpose

You are an independent auditor. Your job is to verify that the code in the repository correctly implements the architecture described in the thesis. You will NOT change any code. You will only read, compare, and report.

You have access to:
1. The thesis PDF: `2023TaoPhD.pdf`
2. The codebase: current directory structure and files.
3. The raw dataset files: see original dataset directory structure and contents in Androids-Corpus folder and in the data folder we extrated the features only... 

Work through each section below in order. For each check, report PASS, FAIL, or UNCLEAR with a one-line explanation. At the end, produce a summary.

---

## SECTION 1: Dataset Integrity

### 1.1 Feature file counts

Open the `cdma_features/rt/` and `cdma_features/it/` directories.

- [ ] Count .npy files in rt/. Expected: 110.
- [ ] Count .npy files in it/. Expected: 110.
- [ ] Confirm all filenames follow the pattern `XX_YY00_Z_frames.npy` where YY is CF, CM, PF, or PM.

### 1.2 Label extraction

The thesis (Chapter 7, Section 7.2) states: 110 participants, 52 control, 58 depressed. Labels come from the participant ID: CF/CM = control, PF/PM = depressed.

- [ ] Count participants where ID contains CF or CM. Expected: 52.
- [ ] Count participants where ID contains PF or PM. Expected: 58.
- [ ] Verify the code's label extraction matches this rule. Find the function that builds labels from participant IDs. Confirm it checks the second segment of the underscore-split ID for CF/CM/PF/PM.

### 1.3 Feature shapes

The thesis (Section 3.3.2) describes: frames of M=128 vectors, each vector is D=32 features (16 LLDs + 16 deltas). Frames overlap by M/2=64.

- [ ] Load `01_CF56_1_frames.npy` from rt/. Expected shape: (N_frames, 128, 32) where N_frames varies per participant. Confirm the sample has shape (109, 128, 32).
- [ ] Load `01_CF56_1_frames.npy` from it/. Expected shape: (290, 128, 32).
- [ ] Load 3 more random participants. Confirm all have shape (*, 128, 32).
- [ ] Confirm dtype is float32.

### 1.4 Fold structure

The thesis (Section 7.4) uses k=3 custom folds. The corpus provides k=5 folds in fold-lists.csv. Our implementation uses the k=5 corpus folds.

Open fold-lists.csv (read with header=None):
- [ ] Row 0 contains section labels (e.g., "Read", "Interview").
- [ ] Row 1 contains fold labels: fold1, fold2, fold3, fold4, fold5 (possibly with quotes).
- [ ] RT folds are in columns 0 through 4. IT folds are in columns 7 through 11. Columns 5 and 6 are empty separators.
- [ ] RT fold participant counts (after filtering to 110 with both streams): should be approximately 22, 23, 22, 21, 22 (total 110).
- [ ] No participant appears in more than one RT fold.
- [ ] Verify the code uses RT fold assignments for all cross-validation splits, not IT fold assignments.

### 1.5 Fold class balance

Compute the depressed percentage for each RT fold (after filtering to the 110 participants with both RT and IT files):

- [ ] Fold 2 should have approximately 78% depressed.
- [ ] Fold 5 should have approximately 32% depressed.
- [ ] These are the two structurally imbalanced folds that cause high per-fold variance.

---

## SECTION 2: Hyperparameters

The thesis (Section 7.4) specifies exact hyperparameters. Find where each is defined in the code and confirm the value.

| Parameter | Thesis Value | Thesis Location | Code Value | Match? |
|-----------|-------------|-----------------|------------|--------|
| LSTM hidden size | 32 | Section 7.4 | | |
| Feature dimension | 32 | Section 3.3.1 | | |
| Frame size (M) | 128 | Section 3.3.2 | | |
| Frame step | 64 (M/2) | Section 3.3.2 | | |
| Learning rate | 1e-3 | Section 7.4 | | |
| Optimizer | RMSProp | Section 7.4 | | |
| Epochs | 300 | Section 7.4 | | |
| Batch size | 16 | Section 7.4 | | |
| Repetitions | 10 | Section 7.4 | | |
| Classification threshold | 0.5 | Section 7.3.3 (Eq 7.1) | | |

- [ ] Confirm there is NO early stopping (thesis trains for exactly 300 epochs).
- [ ] Confirm the optimizer is RMSProp, NOT Adam.
- [ ] Confirm batch size is 16, NOT 256 or any other value.

---

## SECTION 3: Architecture Verification (Equation by Equation)

### 3.1 IT-MLA Layer (Eq 6.1, 6.2)

Thesis (Section 6.3.2):
```
x_i* = x_i + cos(theta_i) * x_i        (Eq 6.1)
cos(theta_i) = (a_k . x_i) / (||a_k|| * ||x_i||)   (Eq 6.2)
```
where a_k is the mean of all vectors in frame k.

Equivalent: `x_i* = x_i * (1 + cosine_similarity(x_i, a_k))`

Find the IT-MLA implementation in the code.

- [ ] Confirm it computes the frame mean (mean over dim=1 for input of shape (batch, 128, 32)).
- [ ] Confirm it computes cosine similarity between each vector and the frame mean.
- [ ] Confirm the output formula is `x * (1 + cos)`, NOT `x + cos * x` applied differently.
- [ ] Confirm IT-MLA has ZERO learnable parameters.
- [ ] Confirm input shape (batch, 128, 32) produces output shape (batch, 128, 32).

### 3.2 LSTM1 Layer (Frame-Level)

Thesis (Section 7.3.3): Each frame is fed to an LSTM. The LSTM produces hidden states. The mean of hidden states is the context vector c_k. A classification head (Linear + sigmoid) produces a per-frame probability.

Find the LSTM1 implementation.

- [ ] LSTM input_size = 32 (FEATURE_DIM).
- [ ] LSTM hidden_size = 32 (LSTM_HIDDEN).
- [ ] LSTM is single-layer (num_layers=1, the PyTorch default).
- [ ] batch_first=True.
- [ ] Context vector = mean of all hidden states (mean pooling over the sequence dimension).
- [ ] Classification head = Linear(32, 1) followed by sigmoid.
- [ ] Returns both the context vector (for CT-GA) and the probability.

### 3.3 p_c and p_o Computation (Eq 7.1)

Thesis (Section 7.3.3, Eq 7.1): "p(d) = n(d)/N, where n(d) is the number of frames assigned to class depression and N is the total number of frames."

This is a counting/voting formula. In the end-to-end differentiable pipeline (Module 4), p_c is implemented as the masked mean of per-frame sigmoid probabilities (a continuous relaxation of the counting formula).

Find how p_c and p_o are computed in the CDMAModel forward pass.

- [ ] Confirm p_c is the masked mean of frame-level sigmoid probabilities for the RT stream.
- [ ] Confirm p_o is the masked mean of frame-level sigmoid probabilities for the IT stream.
- [ ] Confirm the mask correctly excludes padded frames (only real frames contribute).
- [ ] Confirm the denominator uses the count of real frames, not the total padded length.

### 3.4 CT-GA Layer (Eq 7.2, 7.3)

Thesis (Section 7.3.4):
```
c_k* = c_k + [exp(cos(alpha_k)) / sum_j(exp(cos(alpha_j)))] * c_k    (Eq 7.2)
cos(alpha_k) = (s . c_k) / (||s|| * ||c_k||)                          (Eq 7.3)
```
where s is the mean of sequence O (IT), and the attention is applied to sequence C (RT).

This is: `c_k* = c_k * (1 + softmax(cosine_similarities)[k])`

Find the CT-GA implementation.

- [ ] Confirm it computes masked means: r = mean(C), s = mean(O) (accounting for variable-length sequences).
- [ ] Confirm cross-GA: C is attended using s (IT mean) as reference. O is attended using r (RT mean) as reference.
- [ ] Confirm the attention weight uses softmax over cosine similarities.
- [ ] Confirm masked positions are excluded from the softmax (set to -inf or a very large negative number before softmax).
- [ ] Confirm the output formula matches Eq 7.2: `X * (1 + softmax(cos))`, NOT `X + softmax(cos) * X` if implemented differently.
- [ ] Confirm CT-GA has ZERO learnable parameters.
- [ ] Confirm the layer also computes r_star = mean(C_star) and s_star = mean(O_star).

### 3.5 Self-GA (Eq 7.8, 7.9) for BA3 conditions

Thesis (Section 7.4.1, Eq 7.8/7.9): Same formula as CT-GA, but the reference vector is the stream's own mean instead of the other stream's mean. C is attended with mean(C), O is attended with mean(O).

- [ ] Confirm the code has a self-GA path where C is attended using r (its own mean) instead of s.
- [ ] Confirm the same softmax-over-cosine formula is used.
- [ ] Confirm self-GA is activated only for ba3_rt and ba3_it conditions.

### 3.6 LSTM2 Layer (Sequence-Level)

Thesis (Section 7.3.4): The transformed sequences C* and O* are fed to two LSTMs (LSTM2). The average of hidden states is fed to a softmax layer for classification.

Find the LSTM2 implementation.

- [ ] LSTM input_size = 32 (LSTM_HIDDEN, since input is context vectors from LSTM1).
- [ ] LSTM hidden_size = 32.
- [ ] Uses pack_padded_sequence for variable-length inputs.
- [ ] Uses pad_packed_sequence to recover outputs.
- [ ] Masked mean pooling of hidden states (only real positions, not padded).
- [ ] Classification head = Linear(32, 1) followed by sigmoid.
- [ ] Separate LSTM2 instances for RT (produces p_t) and IT (produces p_d).

### 3.7 CTF Layer (Eq 7.4, 7.5)

Thesis (Section 7.3.5):
```
f = r + s          (Eq 7.4)
f* = r* + s*       (Eq 7.5)
```
Then f and f* are each fed to a "softmax layer" (Linear + sigmoid) to produce p_f1 and p_f2.

Find the CTF implementation.

- [ ] Confirm f = r + s (element-wise addition of pre-attention means).
- [ ] Confirm f_star = r_star + s_star (element-wise addition of post-attention means).
- [ ] Confirm p_f1 = sigmoid(Linear(32, 1)(f)).
- [ ] Confirm p_f2 = sigmoid(Linear(32, 1)(f_star)).
- [ ] Confirm there are TWO separate Linear heads (not shared).

### 3.8 Aggregation (Eq 7.6)

Thesis (Section 7.3.6): p_hat = (1/|B|) * sum(p_v for v in B), where B is the set of active probability outputs.

- [ ] Confirm p_hat is the mean of all active probability outputs for the current mode.
- [ ] Confirm p_hat is NOT included in the loss computation.
- [ ] Confirm classification at evaluation uses p_hat > 0.5.

### 3.9 Loss Function (Eq 7.7)

Thesis (Section 7.3.6):
```
L = -(1/K) * SUM_{v in B} [y*log(p_v) + (1-y)*log(1-p_v)]
```
This is a SUM of BCE losses over all active probability outputs, divided by K (number of training samples).

Find the loss function implementation.

- [ ] Confirm BCE is computed for each active probability output separately.
- [ ] Confirm the individual BCE losses are SUMMED (not averaged) across outputs.
- [ ] Confirm nn.BCELoss uses reduction='mean' (which handles the 1/K per batch).
- [ ] Confirm p_hat is excluded from the loss.
- [ ] For ba1_rt (|B|=1): loss = 1 BCE term. For full_cdma (|B|=6): loss = 6 BCE terms summed.

---

## SECTION 4: 13-Condition Routing Verification

The thesis Table 7.2 defines 13 conditions with specific active stages and probability outputs. Verify that the code's mode routing matches exactly.

For each condition, trace the forward pass in the code and confirm which streams are processed, which stages are active, and which probability outputs are produced.

### 4.1 Part I: Single-stream, no LSTM2

| Condition | Need RT? | Need IT? | MLA? | LSTM2? | GA? | CTF? | Expected Outputs |
|-----------|----------|----------|------|--------|-----|------|-----------------|
| ba1_rt | Yes | No | No | No | No | No | p_c |
| ba1_it | No | Yes | No | No | No | No | p_o |
| itmla_rt | Yes | No | Yes | No | No | No | p_c |
| itmla_it | No | Yes | Yes | No | No | No | p_o |

- [ ] Verify ba1_rt: only RT stream through LSTM1, no MLA, output is p_c only.
- [ ] Verify ba1_it: only IT stream through LSTM1, no MLA, output is p_o only.
- [ ] Verify itmla_rt: only RT stream through MLA then LSTM1, output is p_c only.
- [ ] Verify itmla_it: only IT stream through MLA then LSTM1, output is p_o only.
- [ ] Confirm none of these conditions activate LSTM2, GA, or CTF.

### 4.2 Part II: With LSTM2

| Condition | Need RT? | Need IT? | MLA? | GA type | LSTM2? | Expected Outputs |
|-----------|----------|----------|------|---------|--------|-----------------|
| ba2_rt | Yes | No | Yes | None | Yes | p_c, p_t |
| ba2_it | No | Yes | Yes | None | Yes | p_o, p_d |
| ba3_rt | Yes | No | Yes | Self | Yes | p_c, p_t |
| ba3_it | No | Yes | Yes | Self | Yes | p_o, p_d |
| ctga_rt | Yes | YES | Yes | Cross | Yes | p_c, p_t |
| ctga_it | YES | Yes | Yes | Cross | Yes | p_o, p_d |

- [ ] Verify ba2_rt: RT through MLA + LSTM1, then C directly to LSTM2 (NO attention), outputs p_c and p_t.
- [ ] Verify ba2_it: IT through MLA + LSTM1, then O directly to LSTM2 (NO attention), outputs p_o and p_d.
- [ ] Verify ba3_rt: RT through MLA + LSTM1, self-GA (attend C with mean(C)), then LSTM2, outputs p_c and p_t.
- [ ] Verify ba3_it: IT through MLA + LSTM1, self-GA (attend O with mean(O)), then LSTM2, outputs p_o and p_d.
- [ ] CRITICAL: Verify ctga_rt requires BOTH streams through LSTM1 (to get both C and O), then cross-GA, then only RT LSTM2 output reported. Outputs p_c and p_t.
- [ ] CRITICAL: Verify ctga_it requires BOTH streams through LSTM1, then cross-GA, then only IT LSTM2 output reported. Outputs p_o and p_d.
- [ ] Verify that ba2 and ba3 do NOT process the opposite stream (ba2_rt should NOT run the IT stream through LSTM1).

### 4.3 Part III: With CTF

| Condition | Expected Outputs |
|-----------|-----------------|
| ba4 | p_c, p_o, p_t, p_d, p_f1 |
| ba5 | p_c, p_o, p_t, p_d, p_f2 |
| full_cdma | p_c, p_o, p_t, p_d, p_f1, p_f2 |

- [ ] Verify ba4: full pipeline but CTF produces only p_f1 (from f = r + s), NOT p_f2.
- [ ] Verify ba5: full pipeline but CTF produces only p_f2 (from f* = r* + s*), NOT p_f1.
- [ ] Verify full_cdma: full pipeline, CTF produces both p_f1 and p_f2. All 6 outputs active.

---

## SECTION 5: Data Pipeline Verification

### 5.1 Normalization

Thesis does not explicitly specify normalization, but z-score is standard for LLD features in this literature.

- [ ] Confirm the normalizer computes mean and std from training participants only.
- [ ] Confirm it uses ALL frames from BOTH RT and IT streams of training participants for fitting.
- [ ] Confirm test data is normalized using training statistics (no leakage).
- [ ] Confirm the formula is (x - mean) / (std + epsilon) where epsilon is small (e.g., 1e-8).

### 5.2 Padding and Masking

Different participants have different numbers of frames. The code must pad sequences within a batch and use masks.

- [ ] Confirm the collate function pads RT frames to the maximum N in the batch.
- [ ] Confirm the collate function pads IT frames to the maximum L in the batch.
- [ ] Confirm padded positions are filled with zeros.
- [ ] Confirm masks are binary (1 for real, 0 for padded).
- [ ] Confirm mask shapes are (batch, max_N) for RT and (batch, max_L) for IT.

### 5.3 Person Independence

Thesis (Section 7.4): "the same person is never represented in both training and test set."

- [ ] Confirm the code splits participants by fold, not by frame.
- [ ] Confirm there is an assertion or check that train and test participant sets do not overlap.
- [ ] Confirm this holds for all 5 folds.

### 5.4 Seeding

- [ ] Confirm seeds are set before each (condition, fold, rep) combination.
- [ ] Confirm the seed convention is deterministic (e.g., rep * 42).
- [ ] Confirm both torch.manual_seed and torch.cuda.manual_seed_all are called.

---

## SECTION 6: Evaluation Protocol Verification

### 6.1 Pooled Evaluation

The evaluation protocol pools predictions across all 5 folds before computing metrics. Each participant appears in exactly one test fold.

- [ ] Confirm that for each repetition, predictions from all 5 folds are collected.
- [ ] Confirm each participant has exactly one prediction per repetition (no duplicates).
- [ ] Confirm the total prediction count per repetition is 110.
- [ ] Confirm accuracy, precision, recall, and F1 are computed ONCE over all 110 predictions (not averaged across per-fold metrics).

### 6.2 Metric Computation

- [ ] Confirm accuracy = (TP + TN) / (TP + TN + FP + FN).
- [ ] Confirm precision = TP / (TP + FP).
- [ ] Confirm recall = TP / (TP + FN).
- [ ] Confirm F1 = 2 * precision * recall / (precision + recall).
- [ ] Confirm the positive class is "depressed" (label=1), not "control".

---

## SECTION 7: Numerical Spot-Checks

Pick a specific participant and trace one forward pass manually to confirm the code produces the expected intermediate values.

### 7.1 IT-MLA Spot-Check

Take participant `01_CF56_1`, RT stream, first frame (shape 128 x 32).

- [ ] Compute frame mean manually: mean of 128 vectors, result shape (32,).
- [ ] Compute cosine similarity between the first vector and the frame mean.
- [ ] Apply the IT-MLA formula: output_vector = input_vector * (1 + cos_sim).
- [ ] Compare with the code's IT-MLA output for this frame. Values should match to float32 precision.

### 7.2 Loss Scaling Spot-Check

- [ ] Run a forward pass with a dummy batch for ba1_rt (|B|=1) and full_cdma (|B|=6) using the same input data.
- [ ] Confirm the full_cdma loss is approximately 6x the ba1_rt loss (since it sums 6 BCE terms vs 1).

### 7.3 Mask Spot-Check

Create a batch of 2 participants where participant A has 5 RT frames and participant B has 10 RT frames.

- [ ] Confirm rt_frames shape is (2, 10, 128, 32) (padded to max).
- [ ] Confirm rt_mask for participant A is [1,1,1,1,1,0,0,0,0,0].
- [ ] Confirm rt_mask for participant B is [1,1,1,1,1,1,1,1,1,1].
- [ ] Confirm p_c for participant A uses only 5 frame probabilities, not 10.

---

## SECTION 8: Known Discrepancies (Expected, Not Bugs)

These are known differences between our implementation and the thesis. Verify they are present and documented.

### 8.1 k=5 vs k=3 folds

- [ ] Confirm the code uses k=5 corpus-provided folds, not k=3.
- [ ] Confirm this is documented as a known protocol difference.

### 8.2 Part I numbers from Chapter 6

The thesis Table 7.2 Part I numbers (BA1, IT-MLA) are identical to Chapter 6 Table 6.1. Chapter 6 uses frame-level loss and majority vote. Our Module 4 uses the end-to-end aggregated pipeline for all conditions.

- [ ] Confirm the code uses the same pipeline (Module 4, aggregated loss) for all 13 conditions.
- [ ] Confirm there is a separate Module 3 implementing the Chapter 6 frame-level pipeline for comparison.

### 8.3 Participant counts

Chapter 6 uses 116 participants for RT and 112 for IT. Chapter 7 uses 110 (both streams required). Our code uses 110 for all conditions.

- [ ] Confirm the code filters to 110 participants with both RT and IT files.
- [ ] Confirm this is applied consistently across all conditions.

---

## REPORT TEMPLATE

After completing all checks, fill in this summary:

```
VERIFICATION SUMMARY
====================
Date: ___
Codebase version/commit: ___
Thesis reference: Tao 2023 PhD, University of Glasgow

SECTION 1: Dataset Integrity
  Total checks: ___
  PASS: ___
  FAIL: ___
  UNCLEAR: ___
  Notes: ___

SECTION 2: Hyperparameters
  Total checks: ___
  PASS: ___
  FAIL: ___
  Notes: ___

SECTION 3: Architecture (Equations)
  Total checks: ___
  PASS: ___
  FAIL: ___
  CRITICAL FAILURES: ___
  Notes: ___

SECTION 4: 13-Condition Routing
  Total checks: ___
  PASS: ___
  FAIL: ___
  Notes: ___

SECTION 5: Data Pipeline
  Total checks: ___
  PASS: ___
  FAIL: ___
  Notes: ___

SECTION 6: Evaluation Protocol
  Total checks: ___
  PASS: ___
  FAIL: ___
  Notes: ___

SECTION 7: Numerical Spot-Checks
  Total checks: ___
  PASS: ___
  FAIL: ___
  Notes: ___

SECTION 8: Known Discrepancies
  Total checks: ___
  PASS: ___
  FAIL: ___
  Notes: ___

OVERALL VERDICT: [PASS / PASS WITH KNOWN DISCREPANCIES / FAIL]

CRITICAL ISSUES FOUND (if any):
1. ___
2. ___

RECOMMENDATIONS (if any):
1. ___
2. ___
```
