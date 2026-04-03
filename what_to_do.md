# CDMA Replication: Build Plan for Claude Code

## Who you are working with

Zahir Ahmad, Jean Monnet University. Supervisor: Prof. Alessandro Vinciarelli (University of Glasgow). This is a replication of Fuxiang Tao's 2023 PhD thesis, Chapter 7: Cross-Data Multilevel Attention (CDMA) for speech-based depression detection on the Androids Corpus.

## CRITICAL INSTRUCTION

You will implement ONE module at a time. After each module, STOP and wait for Zahir to verify before moving to the next. Do not build ahead. Do not combine modules. Each module must be independently testable.

The thesis PDF is at: `2023TaoPhD.pdf` (in the project files). Reference it when needed.

complete work will be implemented and tested locally.....
in the Androds-Corpus folder original data is as per open source repo.
in the data folder, we have the pre-extracted features (npy files) and fold-lists.csv. We extracted them in the previous attempts, as this time everything becomes vague, and we are starting from sctrach, we will develope the modules also for data analysis, loading, and preprocessing, and finally feature extraction. We will use the same openSMILE configuration (Androids.conf) to ensure feature consistency with the thesis.

.venvCDMA is the virtual environment for this project. Activate it before running any code. I am using uv instead of pip for package management. add the requirements.txt file with the necessary dependencies. in this system, we do not have GPU, but, you need to develop the code in a way that it can run on GPU if available. Use torch.device to handle this. 
---

## THE CORE PROBLEM WITH PREVIOUS ATTEMPTS

We have already attempted this implementation twice. The results were consistently 5 to 15 percentage points below the thesis across all conditions. The architectural ordering (BA1 < IT-MLA < CT-GA < Full CDMA) was partially confirmed, but the absolute numbers never got close. We also found that adding more stages barely changed which participants were classified correctly (only 1 to 4 participants changed between conditions out of 110).

**The most likely source of the gap** is the computation of p_c and p_o (the IT-MLA stage probabilities). The thesis describes these differently in two places:

1. **Chapter 6** (where BA1 and IT-MLA were originally trained): uses "categorical cross-entropy" as the loss function applied per-frame, with "majority vote" aggregation. Each frame is independently classified. The recording is assigned to the class that most frames belong to. This is Eq 7.1: p(d) = n(d)/N where n(d) = count of frames classified as depressed.

2. **Chapter 7** (CDMA pipeline): describes an end-to-end differentiable pipeline with combined loss Eq 7.7 applied to all probability outputs. p_c appears in the loss, which requires it to be differentiable.

**The key question is: how is p_c actually computed during training?**

There are two interpretations:

**Interpretation A (what we implemented before):** p_c is the masked mean of per-frame sigmoid probabilities. Each frame produces a continuous probability via sigmoid(Linear(context)). p_c = mean of these probabilities, weighted by mask. This is fully differentiable. Loss backpropagates through the mean into each frame's LSTM.

**Interpretation B (what Chapter 6 literally describes):** Each frame is independently classified (sigmoid > 0.5 = depressed). p_c = count(depressed frames) / total frames. This is NOT differentiable. During training, you would need frame-level loss (BCE per frame), and p_c would only be used at inference for aggregation.

**Table 7.2 Part I numbers (BA1, IT-MLA) are byte-identical to Chapter 6 Table 6.1.** The thesis did not re-train these for Chapter 7. It reused the Chapter 6 pipeline results. This means BA1 and IT-MLA were trained with frame-level loss and evaluated with majority vote.

For Part II and Part III, the thesis likely trains end-to-end with Eq 7.7, where p_c is computed as Interpretation A (differentiable mean) to allow backpropagation.

**What this build plan does:** We will implement BOTH approaches and test them side by side. Module 3 implements the Chapter 6 pipeline (frame-level). Module 4 implements the Chapter 7 pipeline (end-to-end). We compare BA1 results from both to determine which matches the thesis.

---

## DATA STRUCTURE (CONFIRMED, DO NOT CHANGE)

- **Corpus:** 118 total participants. 110 have both RT and IT data.
- **Chapter 6 uses:** 116 for RT (all who did reading task), 112 for IT (all who did interview task).
- **Chapter 7 uses:** 110 (both RT and IT required). 52 control (HC), 58 depressed (PT).
- **Labels from participant ID:** CF/CM = control (0), PF/PM = patient (1). Example: `01_CF56_1`, `01_PM58_2`.
- **Features:** 32-dim LLD vectors per 10ms frame. 16 LLDs + 16 deltas. Extracted with openSMILE + Androids.conf.
- **Framing:** M=128 vectors per frame, step=64 (50% overlap). Each frame = 1.28 seconds of speech.
- **Sample shapes:** participant `01_CF56_1` has RT=(109 frames, 128, 32), IT=(290 frames, 128, 32).
- **Data location:** Google Drive zip (id=1LJenZ-VXktBbroTI3btVSkRRq-glSWCb). Extracts to `cdma_features/rt/` and `cdma_features/it/` with files like `01_CF56_1_frames.npy`. Plus `fold-lists.csv`.
- **fold-lists.csv:** header=None. Row 0: section labels. Row 1: fold labels (fold1..fold5). Rows 2+: participant IDs (may have single quotes). RT folds: columns 0-4. IT folds: columns 7-11. Columns 5-6 are empty separators.
- **Fold sizes:** F1=23, F2=23, F3=22, F4=22, F5=22. Total=110 (for k=5 protocol, which we use).
- **Fold imbalance:** Fold 2 = 78% depressed test set. Fold 5 = 32%. Overall corpus = 53%.
- **Thesis used k=3 custom unpublished folds.** We use k=5 corpus-provided folds. This means our numbers will differ from the thesis. The point is to confirm the relative ordering between conditions, not to match absolute numbers exactly.

## HYPERPARAMETERS (DO NOT CHANGE UNDER ANY CIRCUMSTANCES)

| Parameter | Value | Source |
|-----------|-------|--------|
| FEATURE_DIM | 32 | Section 3.3.1 |
| LSTM_HIDDEN | 32 | Section 7.4 |
| FRAME_SIZE (M) | 128 | Section 3.3.2 |
| FRAME_STEP | 64 (M/2) | Section 3.3.2 |
| EPOCHS | 300 | Section 7.4 (no early stopping) |
| LEARNING_RATE | 1e-3 | Section 7.4 |
| OPTIMIZER | RMSProp | Section 7.4 |
| BATCH_SIZE | 16 | Section 7.4 (NOT 256, that was a previous bug) |
| N_REPS | 10 | Section 7.4 |
| THRESHOLD | 0.5 | Section 7.3.3 |
| SEED | rep * 42 | Our convention |

**Chapter 6 had different settings for some experiments:** epochs=100, lr=0.0005 for Chapter 5 correlation experiments. But Chapter 6 MLA/BLLSTM used 300 epochs, lr=1e-3. Chapter 7 CDMA used the same. So all our experiments use 300/1e-3.

## ALL 13 CONDITIONS FROM TABLE 7.2

Read the column headers in Table 7.2:
```
Stages: Read | Spont | MLA | LSTM1 | GA-self | GA-cross | LSTM2 | CTF-f1 | CTF-f2
```

### Part I: IT-MLA stage only (single-stream, no LSTM2, no cross-stream)

| # | Name | Active stages | Probability outputs | Thesis F1 |
|---|------|--------------|---------------------|-----------|
| 1 | BA1 (Read) | LSTM1 on RT | p_c | 84.7 |
| 2 | BA1 (Spont.) | LSTM1 on IT | p_o | 83.3 |
| 3 | IT-MLA (Read) | MLA + LSTM1 on RT | p_c | 89.0 |
| 4 | IT-MLA (Spont.) | MLA + LSTM1 on IT | p_o | 87.4 |

**NOTE:** These Part I numbers come from Chapter 6 (frame-level pipeline). See the core problem section above.

### Part II: CT-GA stage (adds LSTM2, tests global attention variants)

| # | Name | Active stages | Probability outputs | Thesis F1 |
|---|------|--------------|---------------------|-----------|
| 5 | BA2 (Read) | MLA + LSTM1 + LSTM2 on RT, **no GA** | p_c, p_t | 89.9 |
| 6 | BA2 (Interview) | MLA + LSTM1 + LSTM2 on IT, **no GA** | p_o, p_d | 87.3 |
| 7 | BA3 (Read) | MLA + LSTM1 + **self-GA** + LSTM2 on RT | p_c, p_t | 90.2 |
| 8 | BA3 (Spont.) | MLA + LSTM1 + **self-GA** + LSTM2 on IT | p_o, p_d | 89.7 |
| 9 | CT-GA (Read) | MLA + LSTM1 (both streams) + **cross-GA** + LSTM2 on RT | p_c, p_t | 90.7 |
| 10 | CT-GA (Spont.) | MLA + LSTM1 (both streams) + **cross-GA** + LSTM2 on IT | p_o, p_d | 89.8 |

**BA2:** C goes directly to LSTM2, no attention transform at all. Tests whether LSTM2 alone helps.
**BA3:** Self-GA means attend C with mean(C), not with mean(O). Same formula as cross-GA (Eq 7.2/7.3) but reference vector is the stream's own mean. Defined in Eq 7.8/7.9.
**CT-GA:** Cross-GA means attend C with mean(O), O with mean(C). BOTH streams must go through LSTM1 to produce C and O, even though only one stream's probability (p_t or p_d) is reported.

### Part III: CTF stage (dual-stream, tests fusion)

| # | Name | Active stages | Probability outputs | Thesis F1 |
|---|------|--------------|---------------------|-----------|
| 11 | BA4 | Full CDMA but CTF uses only f=r+s (p_f1 only) | p_c, p_o, p_t, p_d, p_f1 | 90.1 |
| 12 | BA5 | Full CDMA but CTF uses only f*=r*+s* (p_f2 only) | p_c, p_o, p_t, p_d, p_f2 | 90.7 |
| 13 | CDMA | All stages, all outputs | p_c, p_o, p_t, p_d, p_f1, p_f2 | 92.5 |

---

## ARCHITECTURE SPECIFICATION (from thesis, equation by equation)

### Stage 1: IT-MLA (Eq 6.1, 6.2) -- Zero learnable parameters

For each frame I_k of M=128 feature vectors:
1. Compute frame average: a_k = mean(x_1, ..., x_M)
2. For each vector x_i in the frame:
   - cos(theta_i) = cosine_similarity(a_k, x_i)
   - x_i* = x_i + cos(theta_i) * x_i = x_i * (1 + cos(theta_i))

Effect: vectors aligned with frame average get amplified (up to sqrt(2) for the average itself). Orthogonal vectors go to zero.

Input: (batch, M=128, D=32)
Output: (batch, M=128, D=32)

### Stage 2: LSTM1 -- Frame-level LSTM

Each frame processed independently by a shared LSTM.

LSTM(input_size=32, hidden_size=32, batch_first=True)

For each frame:
1. Feed 128 vectors to LSTM, get 128 hidden states
2. Context c_k = mean of all 128 hidden states (mean pooling)
3. Classification: sigmoid(Linear(32, 1)(c_k)) gives per-frame probability

The context vectors c_k form sequence C = {c_1, ..., c_N} for RT (or O = {o_1, ..., o_L} for IT).

**p_c computation (Eq 7.1):**
Thesis says: p(d) = n(d)/N = count of frames classified as depressed / total frames.
This is a counting operation. At inference, threshold each frame's sigmoid at 0.5, count, divide.

For the differentiable version (needed for end-to-end training): p_c = masked_mean(sigmoid_outputs).

### Stage 3: CT-GA (Eq 7.2, 7.3) -- Zero learnable parameters

Given sequences C (from RT) and O (from IT):
1. Compute means: r = mean(C), s = mean(O)
2. For each c_k in C:
   - cos(alpha_k) = cosine_similarity(s, c_k)
   - c_k* = c_k + [exp(cos(alpha_k)) / sum_j(exp(cos(alpha_j)))] * c_k
   - This is: c_k* = c_k + softmax(cosine_similarities)[k] * c_k
3. Same for O using r as reference
4. C* and O* are the attended sequences
5. r* = mean(C*), s* = mean(O*)

**Self-GA variant (BA3, Eq 7.8/7.9):** Same formula but attend C with mean(C) and O with mean(O) instead of cross-stream.

### Stage 4: LSTM2 -- Sequence-level LSTM

Processes the sequence of frame-level context vectors (C* or O*).

LSTM(input_size=32, hidden_size=32, batch_first=True)

Uses pack_padded_sequence because sequences have variable length (different participants have different numbers of frames).

1. Pack and feed to LSTM
2. Masked mean pooling of hidden states
3. sigmoid(Linear(32, 1)(pooled)) gives p_t (for RT) or p_d (for IT)

### Stage 5: CTF (Eq 7.4, 7.5)

1. f = r + s (sum of pre-attention means)
2. f* = r* + s* (sum of post-attention means)
3. p_f1 = sigmoid(Linear(32, 1)(f))
4. p_f2 = sigmoid(Linear(32, 1)(f*))

### Aggregation (Eq 7.6)

p_hat = (1/|B|) * sum(all active probability outputs)

Classify as depressed if p_hat > 0.5.

### Loss (Eq 7.7)

L = -(1/K) * SUM_{v in B} [y*log(p_v) + (1-y)*log(1-p_v)]

This is a SUM of BCE over each probability output (not mean). K = number of training samples. nn.BCELoss(reduction='mean') handles the 1/K per batch. The aggregation over outputs in B must be .sum().

p_hat is NOT included in the loss. Only the individual probability outputs are.

### Normalization

Z-score normalization. Fit mean and std on ALL frames from ALL training participants (both RT and IT). Apply to test data using training statistics. No leakage.

---

## BUILD MODULES (implement one at a time, verify before moving on)

### Module 1: Data Loading and Validation

**What to build:**
- Download and extract data from Google Drive
- Parse fold-lists.csv
- Build participant info and label map
- Load .npy frame files
- Verify shapes, counts, labels

**Verification checklist:**
- [ ] 110 RT files, 110 IT files found
- [ ] 110 participants with both RT and IT
- [ ] 52 control, 58 depressed
- [ ] Fold 1=23, F2=23, F3=22, F4=22, F5=22 participants
- [ ] No participant appears in more than one fold
- [ ] All fold participants exist in the loaded data
- [ ] Sample participant `01_CF56_1`: RT shape = (109, 128, 32), IT shape = (290, 128, 32)
- [ ] Feature values are float32, not all zeros, not all NaN
- [ ] Print fold class balance: fold 2 should be ~78% depressed, fold 5 ~32%

**STOP after this module. Wait for verification.**

### Module 2: Dataset, Normalizer, DataLoader

**What to build:**
- FeatureNormalizer: z-score, fit on training participants only, both RT+IT
- AndroidsDataset: loads all participant data into memory
- collate_fn: pad variable-length sequences, create masks
- get_dataloaders: split by fold, create train/test loaders

**Verification checklist:**
- [ ] Normalizer fit on fold 1 training set (88 participants): mean and std are shape (32,), not zero
- [ ] After normalization, fold 1 training features have approximately mean=0, std=1
- [ ] Test features (fold 1, 22 participants) use training stats, NOT their own
- [ ] collate_fn output shapes correct for a batch of 4 participants:
  - rt_frames: (4, max_N, 128, 32)
  - it_frames: (4, max_L, 128, 32)
  - rt_mask: (4, max_N) with 1s for valid positions
  - it_mask: (4, max_L)
  - n_rt: (4,) integer counts
  - n_it: (4,)
  - labels: (4,)
  - pids: list of 4 strings
- [ ] Padded positions in rt_frames are all zeros
- [ ] Mask sums match n_rt and n_it
- [ ] Batch size is 16. With 88 training participants, that is 5-6 batches per epoch
- [ ] Person independence: no overlap between train and test participant sets

**STOP after this module. Wait for verification.**

### Module 3: Chapter 6 Pipeline (Frame-Level) -- BA1 and IT-MLA

This is the pipeline that produced the BA1 and IT-MLA numbers in Table 7.2 Part I. It was described in Chapter 6.

**What to build:**
- ITMLALayer (Eq 6.1/6.2)
- LSTM1Layer that outputs per-frame classifications
- Frame-level BCE loss (each frame gets the participant's label)
- Majority vote evaluation (count frames classified as depressed / total)
- Training loop for single-stream conditions
- Support for 4 conditions: ba1_rt, ba1_it, itmla_rt, itmla_it

**Architecture for this module:**
```
Input: participant's frames (N, 128, 32)
For each frame k:
    if MLA: frame = IT-MLA(frame)           # (128, 32) -> (128, 32)
    hidden_states = LSTM(frame)             # (128, 32) -> (128, 32)
    context = mean(hidden_states)           # (128, 32) -> (32,)
    prob_k = sigmoid(Linear(32,1)(context)) # (32,) -> scalar

Loss: BCE averaged over all N frames (each frame's prob vs participant label)
Evaluation: p_c = count(prob_k > 0.5 for k in frames) / N. Classify depressed if p_c > 0.5.
```

**Key differences from the Chapter 7 pipeline:**
1. Loss is PER-FRAME, not per-participant
2. Evaluation uses COUNTING (majority vote), not mean of probabilities
3. Single-stream only (no cross-stream interaction)
4. During training, each frame is a separate training example with the participant's label

**Verification checklist:**
- [ ] ITMLALayer: input (batch, 128, 32), output same shape
- [ ] ITMLALayer: output = input * (1 + cosine_similarity(input, mean(input, dim=1)))
- [ ] LSTM1Layer: input (128, 32), output context (32,) and prob (1,)
- [ ] Run ba1_rt for 1 rep on fold 1 (300 epochs). Check:
  - Loss decreases over epochs
  - Final accuracy on fold 1 test set is above 60% (not random)
  - Training takes roughly 5-8 minutes on T4 GPU
- [ ] Run ba1_rt for 1 rep, all 5 folds, pooled evaluation:
  - All 110 participants have predictions
  - No participant appears twice
  - Pooled accuracy is in the range 75-90%
- [ ] Run itmla_rt same way. MLA should improve over BA1 by at least 1-2 points on average
- [ ] Compare with thesis targets (BA1 RT: 84.7 F1, IT-MLA RT: 89.0 F1)

**STOP after this module. Wait for verification.**

### Module 4: Chapter 7 Pipeline (End-to-End) -- All 13 Conditions

This is the full CDMA pipeline with end-to-end training.

**What to build:**
- CDMAModel class supporting all 13 modes
- CombinedBCELoss (Eq 7.7: SUM of BCE over outputs, not mean)
- CTGALayer (Eq 7.2/7.3) with support for self-GA (Eq 7.8/7.9) and cross-GA
- LSTM2Layer (sequence-level with pack_padded_sequence)
- CTFLayer (Eq 7.4/7.5)
- Aggregation (Eq 7.6: mean of all active probabilities)
- Full forward pass for all 13 conditions

**p_c computation in this pipeline:**
Use the differentiable version: p_c = masked_mean(per-frame sigmoid probabilities). This allows backpropagation through the loss. At evaluation, use p_hat > 0.5 for final classification.

**Condition routing logic:**
```
Mode           | need_rt | need_it | use_mla | ga_type | lstm2 | ctf_f1 | ctf_f2 | outputs
---------------|---------|---------|---------|---------|-------|--------|--------|--------
ba1_rt         | yes     | no      | no      | none    | no    | no     | no     | p_c
ba1_it         | no      | yes     | no      | none    | no    | no     | no     | p_o
itmla_rt       | yes     | no      | yes     | none    | no    | no     | no     | p_c
itmla_it       | no      | yes     | yes     | none    | no    | no     | no     | p_o
ba2_rt         | yes     | no      | yes     | none    | yes   | no     | no     | p_c, p_t
ba2_it         | no      | yes     | yes     | none    | yes   | no     | no     | p_o, p_d
ba3_rt         | yes     | no      | yes     | self    | yes   | no     | no     | p_c, p_t
ba3_it         | no      | yes     | yes     | self    | yes   | no     | no     | p_o, p_d
ctga_rt        | yes     | YES     | yes     | cross   | yes   | no     | no     | p_c, p_t
ctga_it        | YES     | yes     | yes     | cross   | yes   | no     | no     | p_o, p_d
ba4            | yes     | yes     | yes     | cross   | yes   | yes    | no     | p_c,p_o,p_t,p_d,p_f1
ba5            | yes     | yes     | yes     | cross   | yes   | no     | yes    | p_c,p_o,p_t,p_d,p_f2
full_cdma      | yes     | yes     | yes     | cross   | yes   | yes    | yes    | p_c,p_o,p_t,p_d,p_f1,p_f2
```

CRITICAL: ctga_rt and ctga_it need BOTH streams through LSTM1 because cross-GA requires both C and O. But they only REPORT one stream's probability outputs.

**Verification checklist:**
- [ ] Forward pass works for all 13 modes on dummy data (batch=2)
- [ ] Each mode produces exactly the expected probability outputs
- [ ] p_hat shape is (B, 1) for all modes
- [ ] No NaN in any output
- [ ] Loss scales correctly: for |B|=1 (ba1), loss is one BCE term. For |B|=6 (full_cdma), loss is 6x larger
- [ ] Gradient flows: run 1 training step, check that model parameters change
- [ ] Run ba1_rt with this pipeline for 1 rep, compare with Module 3 results (they will differ because different loss/evaluation)
- [ ] Run full_cdma for 1 rep, 1 fold: verify it produces predictions for all test participants

**STOP after this module. Wait for verification.**

### Module 5: Experiment Runner with Pooled Evaluation

**What to build:**
- run_condition function: loops over reps (outer) and folds (inner)
- Pooled evaluation: collect raw predictions from all 5 folds per rep, compute metrics once over all 110 participants
- Resume support: save fold_predictions.csv after every fold, skip completed (condition, fold, rep) combos
- pooled_results.csv: one row per (condition, rep) with accuracy, precision, recall, f1
- Results summary printing

**Verification checklist:**
- [ ] Run ba1_rt for 1 rep: fold_predictions.csv has 110 rows, pooled_results.csv has 1 row
- [ ] Pooled metrics computed from fold_predictions match the pooled_results row
- [ ] Kill and restart: completed folds are skipped, new folds resume correctly
- [ ] Run ba1_rt for 3 reps: pooled_results has 3 rows, fold_predictions has 330 rows
- [ ] No duplicate (condition, fold, rep, pid) combinations in fold_predictions

**STOP after this module. Wait for verification.**

### Module 6: Full 13-Condition Experiment

**What to build:**
- Command-line interface: --test, --conditions, --reps, --results-dir
- Validation mode (--test): quick check of all 13 modes
- Full run: all 13 conditions, 10 reps each
- Final comparison table against thesis Table 7.2

**This module just wires everything together. No new logic.**

---

## KNOWN BUGS FROM PREVIOUS ATTEMPTS (never reintroduce)

1. **Batch size 256:** Produces only 1 gradient update per epoch. Must be 16.
2. **LSTM output unpacking with generators:** Silently mangles tensor shapes. Use explicit .view() calls.
3. **DataParallel:** Overhead is not worth it at batch size 16. Do not use.
4. **Loss .mean() over probability outputs:** Eq 7.7 is a SUM, not mean. Using .mean() divides gradient by |B|, reducing effective learning rate for multi-output conditions.
5. **Requiring both RT+IT files for single-stream conditions:** BA1(Read) should only need RT files. But since the thesis Chapter 7 uses 110 participants for all conditions (including BA1 in the combined pipeline), requiring both is actually correct for the Chapter 7 pipeline. For the Chapter 6 pipeline (Module 3), use the stream-specific participant set.

## EVALUATION PROTOCOL

**Experiment 2 protocol (per Prof. Vinciarelli):**
- Use all k=5 corpus-provided folds
- For each repetition (different random seed):
  - Train on folds 2-5, test on fold 1. Save predictions.
  - Train on folds 1,3-5, test on fold 2. Save predictions.
  - ... for all 5 folds
  - Pool all 110 predictions. Compute one set of metrics.
- Repeat for N_REPS=10 repetitions
- Report mean and std of pooled metrics across repetitions

This eliminates fold-composition variance. The standard deviation reflects only initialization variance.

## STYLE RULES

- Use explicit variable names. No single-letter variables except loop counters.
- Use logger.info for epoch loss output (not debug, not print).
- Save results after every single fold completion (resume-safe).
- No DataParallel.
- No generator expressions for tensor operations.
- Test mode (--test) must complete in under 30 seconds.