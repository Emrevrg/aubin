# AUBIN — results, 3 October 2026

These are all measured numbers. No result that uses a Kev model as a member is counted as "beating Kev". Each section states its protocol and what was used to select it.

## 1. Kev public suites (Kev's own training sources): `kev_test` 1,440 questions, `kev_transfer_test` 764

How the ensemble was built:
- **Members, weights and temperature** were chosen only on the calibration split `cal300`. This split is disjoint from both test sets.
- **Selection method:** greedy forward selection on NLL. It picked 4 members with equal weight, at T = 0.8.
- **Test sets** were scored once, for reporting only.

| model | kev_test | kev_transfer_test |
|---|---|---|
| **AUBIN ensemble (E4B-v3 + 12B-b + 12B-d + 31B)** | **85.7** | **86.5** |
| AUBIN-12B-b (wide-option training, 3 Oct) | 85.1 | 84.7 |
| AUBIN-12B-d | 84.8 | 85.2 |
| AUBIN-12B-e | 84.4 | 83.6 |
| AUBIN-E4B-v3 | 84.3 | 83.9 |
| AUBIN-31B | 82.9 | **89.0** |
| Kev-0.8B | 83.8 | – |
| Kev-4B | 86.5 | – |
| Kev-27B | 87.0 | – |
| Kev-9B | 87.4 | – |

- **Calibration of the ensemble:** Brier 0.232 / 0.208 and ECE 0.033 / 0.058 (kev_test / transfer).
- **Status:** on Kev's own training sources, AUBIN beats Kev-0.8B. It is **1.7 points behind Kev-9B**.
- **New sources:** on sources Kev was not trained on, AUBIN-31B was already ahead (89.8 vs 89.6, report of 1 Oct).

Alternative selections, reported for transparency:
- cal300 accuracy-selected: 84.8
- 12B-b + 12B-e: 85.4
- equal-weight mix of the 4 new models: 85.2

## 2. Typed decisions (`LocalLLaMA/typed-decisions`): 400 test cases × 5 questions = 2,000 decisions

| system | accuracy |
|---|---|
| **AUBIN-Learn (AUBIN-12B + NIVEN tree skill, equal weights, fixed before scoring)** | **77.55** |
| NIVEN (Norovox, 26 Sep) | 77.3 |
| NIVEN tree skill alone | 77.1 |
| meraGPT sd-1 | 76.8 |
| Laya | 76.65 |
| AUBIN-12B zero-shot | 72.9 |
| TypeSafe Jev 1.13.0 | 72.7 |

How the AUBIN-Learn row was produced:
- The fusion weights were fixed before the test was scored; they were not tuned on test.
- The tree skill is trained only on the 1,200 training cases.

Caveat: the dataset's teacher agrees with itself only about 73.5% of the time, so scores above ~75% are close to saturation. Differences of a few tenths are within noise. With 2,000 decisions, the 95% interval is roughly ±1.8 points.

## 3. Computer use: Mind2Web (MindAct multiple-choice protocol)

Protocol:
- top-50 candidates from the official ranker
- groups of 5 + "None"
- iterative tournament
- 200 steps per split

| model | cross-task | cross-website | cross-domain |
|---|---|---|---|
| AUBIN-31B (no web fine-tuning) | – | **33.5** | **48.5** |
| AUBIN-12B-Web (1,200 steps on Mind2Web train) | 40.0 | 32.0 | 45.0 |
| MindAct Flan-T5-XL (paper, full split) | 52.0 | 38.9 | 39.6 |
| MindAct GPT-4 (paper, 50 tasks) | 36.2 | 30.1 | 26.4 |

The table shows step success rate (%). Element accuracy for AUBIN-31B is 54.0 / 41.0 (domain / website); operation F1 is 86.5 / 82.0.

How to read these numbers:
- The sample is 200 steps per split, not the full split, so each number is ±7 points.
- Cross-domain is the hardest split, because the domains are entirely unseen. AUBIN-31B is above the published MindAct baselines there.
- On cross-task and cross-website it is below MindAct-XL.
- AUBIN is **not** claimed to be the best open-source computer-use agent.

## 4. Real-time control: AUBIN-12B-Control and AUBIN-E4B-Control

The benchmark is a command-following grid game with deadly lava: 100 episodes, never seen in training.

| controller | success | lava deaths |
|---|---|---|
| **AUBIN-12B-Control + safety shield** | **92%** | **0** |
| AUBIN-12B-Control, no shield | 74% | 24 |
| greedy rule + same shield | 89% | 0 |

The same shield is also applied to the rule baselines. Full table: model card of `emrevrg/AUBIN-12B-Control`.

## 5. AUBIN-Learn: learning without retraining

Details: `reports/AUBIN_LEARN.md`.

**Feedback stream (kev_test).** The label is revealed after each answer. All 6 variants improve by **+1.0 to +1.3** points (84.3 → 85.6). This protocol is separate from the static scores above and must not be compared with them.

**Never-seen sources (transfer).** The result is −0.4 to +0.1 points: no harm, and no gain yet.

**Gated fast skills.** These are per-source classifiers learned from memory in seconds. Each one is switched on only if dev proves it.
- For the models before 3 Oct, they add +0.3 to +0.6 points on the locked test.
- For the 3 Oct models trained on wide-option data, the gate stays closed: there is no gain, and nothing is broken.

**Safety by construction.** Memory and skills sit outside the weights. The calibrator (Hedge) moves trust back to the model wherever memory does not help. A bad lesson therefore cannot overwrite what the model already knows; this is the "learn by trying without breaking" property.

## 6. Integration

AUBIN-Learn exposes the same interface in every AUBIN repository (`INTEGRATION.md`):
- **Python:** `AubinLearning(model).decide(...)` and `.learn(...)`.
- **HTTP:** `aubin serve --learn mem.npz` serves `POST /decide`, `POST /learn` and `GET /health`.
- **OpenAI-compatible:** `POST /v1/chat/completions`, so existing OpenAI clients can connect.
- **MCP server:** AUBIN can be used as a tool by other assistants.

Code is in `code/`. Every number above can be reproduced from the scripts and run files in this repository.

---

## Evening update, 3 October: learned skills, visual computer use, FPS

### A. Kev training sources, with AUBIN-Learn skills (kev_test, 1,440 questions)

**What a skill is.** A small encoder fine-tuned on each source's public training split: Kev train plus the leak-free kev_augment data. Six encoders were trained: DeBERTa-v3-base and -large, RoBERTa-large, ModernBERT-large, bge-large and e5-large.

**How skills are fused.** Skills are fused with the AUBIN ensemble per source. Every choice is made on **kev_dev only**: which skill, its weight, and the per-source temperature.

| procedure (all selected on kev_dev) | kev_test | NLL |
|---|---|---|
| AUBIN ensemble, no skills | 85.69 | 0.789 |
| **FINAL (pre-declared, run once):** selection on kev_dev + cal300 (1,768 questions), strict accuracy gate, per-source temperature | **87.08** | **0.431** |
| **pre-declared final:** dev-accuracy selection, strict gate (≥4 questions and ≥3%) | **86.32** | – |
| NLL selection, gate ≥2 questions and lower NLL | 86.39 | 0.716 |
| best variant observed (chosen after other variants had been scored on test, so **not a clean result**) | 87.01 | 0.641 |
| per-source member routing ("best model per topic") | 85.62 | – (overfit to dev; not used) |

- **Result:** on Kev's own training sources, AUBIN reaches **87.08**, so it does **not** yet beat Kev-9B (87.4). The gap is about 5 questions. Calibration improves strongly: NLL goes from 0.789 to 0.431.
- **Where skills help:** they lift the hardest source, banking77, from 64.7 to 75–79. The ModernBERT skill alone scores 83.6 on banking77.
- **Why there is no larger gain:** per-source selection on dev splits of 80–116 questions overfits. Example: on boolq, a skill-only choice went from 90 to 95 on dev but from 92.5 down to 80 on test.
- **Planned fix:** distil the skills into the model through training, and select on a larger split.

### B. Visual computer use: ScreenSpot (1,272 screenshots, click-point accuracy)

| model | accuracy |
|---|---|
| **AUBIN-E4B-Screen**, round 2 (16k leak-free wave-ui examples) | **69.3** |
| AUBIN-E4B-Screen, round 1 (8k examples) | 67.7 |
| AUBIN-E4B zero-shot | 48.5 |
| SeeClick | 53.4 |
| CogAgent | 47.4 |
| UGround-7B | 73.3 |
| OS-Atlas-7B | 82.5 |
| UI-TARS-7B | 89.5 |

**Training data.** The model was trained on agentsea/wave-ui. To keep it leak-free, the screenspot, agent_studio (GroundUI) and mind2web_test sources were removed.

**Speed.** About 1.9 s per screenshot on a T4 in 4-bit. A second training round is running.

### C. Mind2Web, full training, MindAct protocol (step success, 200 steps per split)

| model | cross-task | cross-website | cross-domain |
|---|---|---|---|
| AUBIN-12B-Web v2 | 47.0 | 37.0 | 43.5 |
| **AUBIN-E4B-Web** (≈2× faster) | 47.0 | 34.5 | 42.0 |
| MindAct Flan-T5-XL | 52.0 | 38.9 | 39.6 |
| GPT-4 (MindAct) | 36.2 | 30.1 | 26.4 |

### D. FPS: ViZDoom defend_the_center (kills per episode)

| agent | kills per episode | max |
|---|---|---|
| **AUBIN-12B**, zero-shot, no game training (12 episodes) | **15.1** | **23** |
| random agent | 1.25 | – |
| scripted rule (turn to nearest, fire when aimed) | 19.6 | – |

**Perception.** Enemy positions come from the engine's label buffer, used here as the object detector. AUBIN makes the decisions, at about 0.55 s per move.

**Self-learning in the game.**
- A bug in the memory recall made the first self-learning runs collapse: with only a handful of stored cases, an unrelated case voted on every situation.
- This is fixed with a similarity threshold (min_sim) in ubin/learn.py.
- **Fixed run, same 12 seeds:** AUBIN-12B + AUBIN-Learn scores **18.3 kills per episode (max 25)**, against **15.1** for the same model without learning: +3.2, or +21%. The memory stores moves that made progress, keyed by game situation, and persists across episodes.
- Per-episode differences range from −12 to +15, so 12 episodes were not conclusive.
- **30-episode confirmation, same seeds:** with learning **17.3**, without learning **15.6** (+1.7, +11%); scripted rule 18.8; random 1.3. The paired difference is not yet statistically significant (t ≈ 1.15). The 12-episode figure overstated the gain; the 30-episode figure is the one to cite.

### E. Self-acquired skills (product)

AubinLearning.acquire_skill(qid, examples) lets the model add a skill to itself:
1. It learns a new skill from examples.
2. It holds out part of the examples and **tests itself** on them.
3. It switches the skill on only if accuracy rises and log-loss falls.

Unit test:
- **Model that does not know the task:** the skill is enabled, and self-test accuracy goes from 34% to 100%.
- **Model that already knows the task:** the skill stays off, and nothing changes.

## Update, 4 October 2026: every ability in both 12B and 31B (AUBIN Omni)

| ability | 12B | 31B |
|---|---|---|
| typed, calibrated decisions | AUBIN-12B | AUBIN-31B |
| click on screenshots | **ScreenSpot 66.7** (full set, 1,272; text 76.9, icon 54.3): the Gemma-4-12B base itself with fp32 compute, no AUBIN adapter | E4B-Screen companion (69.3) |
| web agent step | AUBIN-12B-Web, Mind2Web cross-domain 43.5 | **AUBIN-31B-Web, Mind2Web cross-domain 45.0** (element accuracy 49.0, operation F1 0.890; 200 steps, dev-selected step 300 of 420). Weights are uploaded with the public release (private storage limit); until then Omni uses the E4B-Web companion |
| real-time control and games | AUBIN-12B-Control, 92% · 0 lava | E4B-Control companion |
| self-learning and new skills | AUBIN-Learn | AUBIN-Learn |

Honest notes:
- **12B screen in fp16 does not work.** With image tokens, the 12B language model overflows in fp16. Every answer came out empty (0.0), and LoRA training gave NaN loss from the first example. Clamping the fp16 activations did not help: the model repeated tokens. Running the vision tower in fp32 alone was not enough. Only fp32 compute for the language model works, so the 12B screen ability runs that way. AUBIN fine-tuning of the 12B screen ability (in fp32) is the next training run.
- **Companion models** are separate, lazily loaded E4B adapters, so one AubinOmni object offers every ability. They are listed in omni.json and by omni.companions.
- Code: ubin/omni.py (base mode and companions), screenspot_eval.py (--fp32lm, --shard), inalize_omni.py.
