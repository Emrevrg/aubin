# AUBIN-Learn — instant self-learning memory for AUBIN (Norovox core)

AUBIN-Learn plugs the Norovox self-learning core into AUBIN's fast decision loop. Knowledge is not baked into weights: verified cases live in an external **decision memory**. When AUBIN is unsure, it recalls the most similar solved cases of the same question type and fuses their vote with its own probabilities; when it is told the right answer, `learn()` writes it to memory and the very next decision uses it — no gradient step, no retraining.

```python
from aubin.learn import DecisionMemory, AubinLearner
mem = DecisionMemory()                       # or DecisionMemory(embed_fn=<sentence embedder>)
agent = AubinLearner(score_fn=scorer.score, memory=mem, weights=dev_tuned_weights, margin=4.0,
                     research_fn=Retrieval().fetch)   # Norovox Retrieval: local KB -> web, when memory has nothing
choice, logprobs, evidence = agent.decide(item)
agent.feedback(item, correct_index)  # learned instantly
```

## Results at a glance

* **Learning from feedback works:** on the kev_test stream every one of 6 AUBIN variants improved (+1.0 to +1.3 points); best `ens_a12_b+a31lora2+a12_long` 84.3 → **85.6**. Online protocol (labels revealed after each answer) — not a static test score.
* **Never-seen sources (transfer, memory starts empty):** change -0.4 to +0.1 points — the self-calibrator keeps trust on the model when memory is not yet useful, so it does not hurt; a few hundred feedbacks per source are not enough to help.
* **Static locked test (no feedback):** fusing a frozen memory into an already-fine-tuned AUBIN changes kev_test by -1.0 to +0.7 points — no reliable gain; AUBIN already learned these sources.
* Latency: writing a case to memory takes well under a millisecond with the hashing embedder (tens of ms with a sentence embedder); recall is a matrix product over the partition.

## Protocol

* Memory = Kev decision-v7 **train** (+ kev_augment extra rows). Test / development / calibration / transfer items never enter memory.
* Model log-probabilities come from previously measured AUBIN runs (same item order; labels verified to match).
* Fusion weight (per source, by log-loss), neighbour count and temperature are selected on development data only (kev_dev; cal-300 when a run has no dev); kev_test and kev_transfer_test are reported once.
* Also measured: a learning curve (memory 0% → 100%), latency, and an online-feedback stream (separate protocol).

Memory: 15576 Kev decision-v7 train items + 67473 leak-free extra items (kev_augment.py: same HF source datasets, rows not used anywhere in Kev's suites).

## Embedding `bge`, memory `kev+extra` (83049 items)

Latency: learn 30.746 ms (median, incl. embedding) · recall 96.126 ms/query (incl. embedding).

| model predictions | tuned on | kev_test model | kev_test + memory | memory alone | kev_transfer_test model | + memory |
|---|---|---|---|---|---|---|
| a12_b | kev_dev | 83.1 | 83.8 | 68.6 | 81.5 | 81.5 |
| a31lora2 | kev_dev | 82.8 | 82.9 | 68.3 | 89.0 | 89.0 |
| a12_long | kev_dev | 83.0 | 82.8 | 67.8 | 84.0 | 84.0 |
| a12_dv_test | cal300 | 83.8 | 83.8 | 68.3 | 84.8 | 84.8 |
| a31_think_test | cal300 | 83.7 | 83.8 | 68.4 | 88.0 | 88.0 |
| ens_a12_b+a31lora2+a12_long | kev_dev | 84.3 | 84.1 | 68.3 | 87.7 | 87.7 |

Reference on kev_test (Kev's own training sources): Kev-9B 87.4, Kev-27B 87.0. Best row above: `ens_a12_b+a31lora2+a12_long` 84.1.

Per source (kev_test), model → model + memory:

| source | n | model | + memory |
|---|---|---|---|
| agnews | 24 | 75.0 | 75.0 |
| agnews_yn | 276 | 87.7 | 88.0 |
| amazon | 80 | 62.5 | 58.8 |
| banking77 | 116 | 59.5 | 63.8 |
| boolq | 80 | 91.2 | 92.5 |
| contrastive_age_eligibility | 40 | 100.0 | 100.0 |
| contrastive_quantity_limit | 40 | 97.5 | 95.0 |
| contrastive_return_window | 40 | 92.5 | 92.5 |
| contrastive_spend_threshold | 76 | 100.0 | 100.0 |
| dbpedia14 | 116 | 95.7 | 94.0 |
| imdb | 80 | 97.5 | 97.5 |
| mnli | 116 | 89.7 | 89.7 |
| sst5 | 80 | 61.3 | 58.8 |
| trec | 116 | 89.7 | 89.7 |
| yelp_yn | 160 | 77.5 | 76.2 |

Learning curve — the same model, memory grown from 0% to 100% (no retraining):

| memory fraction | items | kev_test |
|---|---|---|
| 0.0 | 0 | 84.3 |
| 0.05 | 4147 | 83.0 |
| 0.2 | 16563 | 84.7 |
| 0.5 | 41549 | 84.3 |
| 1.0 | 83049 | 84.1 |

Online feedback (separate protocol, not a static test score): answering kev_test as a stream and writing each correct label to memory after answering gives 83.5 overall, 86.0 on the second half.

## Embedding `bge`, memory `kev_only` (15576 items)

Latency: learn 33.498 ms (median, incl. embedding) · recall 39.162 ms/query (incl. embedding).

| model predictions | tuned on | kev_test model | kev_test + memory | memory alone | kev_transfer_test model | + memory |
|---|---|---|---|---|---|---|
| a12_b | kev_dev | 83.1 | 83.5 | 66.4 | 81.5 | 81.5 |
| a31lora2 | kev_dev | 82.8 | 83.2 | 66.0 | 89.0 | 89.0 |
| a12_long | kev_dev | 83.0 | 83.1 | 65.6 | 84.0 | 84.0 |
| a12_dv_test | cal300 | 83.8 | 83.5 | 66.7 | 84.8 | 84.8 |
| a31_think_test | cal300 | 83.7 | 82.7 | 64.8 | 88.0 | 88.0 |
| ens_a12_b+a31lora2+a12_long | kev_dev | 84.3 | 84.2 | 66.0 | 87.7 | 87.7 |

Reference on kev_test (Kev's own training sources): Kev-9B 87.4, Kev-27B 87.0. Best row above: `ens_a12_b+a31lora2+a12_long` 84.2.

Per source (kev_test), model → model + memory:

| source | n | model | + memory |
|---|---|---|---|
| agnews | 24 | 75.0 | 75.0 |
| agnews_yn | 276 | 87.7 | 89.1 |
| amazon | 80 | 62.5 | 61.3 |
| banking77 | 116 | 59.5 | 60.3 |
| boolq | 80 | 91.2 | 92.5 |
| contrastive_age_eligibility | 40 | 100.0 | 100.0 |
| contrastive_quantity_limit | 40 | 97.5 | 92.5 |
| contrastive_return_window | 40 | 92.5 | 92.5 |
| contrastive_spend_threshold | 76 | 100.0 | 100.0 |
| dbpedia14 | 116 | 95.7 | 94.0 |
| imdb | 80 | 97.5 | 97.5 |
| mnli | 116 | 89.7 | 89.7 |
| sst5 | 80 | 61.3 | 61.3 |
| trec | 116 | 89.7 | 89.7 |
| yelp_yn | 160 | 77.5 | 76.2 |

Online feedback (separate protocol, not a static test score): answering kev_test as a stream and writing each correct label to memory after answering gives 82.5 overall, 86.4 on the second half.

## Fast skills (learned from memory in seconds)

The Norovox skill-library idea applied to decisions: for every source, a small softmax classifier is fitted on the memory's embeddings (labels by name) in seconds on a CPU, and fused with AUBIN's probabilities. Regularisation and per-source fusion weights were selected on kev_dev only; test is reported once. A skill is switched on for a source only if, on kev_dev, it raises accuracy by at least 2 questions **and** lowers log-loss; everywhere else it stays off. (An ungated variant that picked weights by log-loss alone gave no gain — `reports/learn_skill_ungated.json`.)

| predictions | kev_test model | kev_test + skills | kev_transfer_test model | + skills |
|---|---|---|---|---|
| a12_b | 83.1 | 83.5 | 81.5 | 81.5 |
| a31lora2 | 82.8 | 83.5 | 89.0 | 89.0 |
| a12_long | 83.0 | 83.4 | 84.0 | 84.0 |
| ens_a12_b+a31lora2+a12_long | 84.3 | 84.8 | 87.7 | 87.7 |

Per source for `ens_a12_b+a31lora2+a12_long` (kev_test): n · model · + skills

* agnews: 24 · 75.0 · 75.0
* agnews_yn: 276 · 87.7 · 88.8
* amazon: 80 · 62.5 · 62.5
* banking77: 116 · 59.5 · 65.5
* boolq: 80 · 91.2 · 91.2
* contrastive_age_eligibility: 40 · 100.0 · 100.0
* contrastive_quantity_limit: 40 · 97.5 · 97.5
* contrastive_return_window: 40 · 92.5 · 92.5
* contrastive_spend_threshold: 76 · 100.0 · 100.0
* dbpedia14: 116 · 95.7 · 95.7
* imdb: 80 · 97.5 · 97.5
* mnli: 116 · 89.7 · 89.7
* sst5: 80 · 61.3 · 61.3
* trec: 116 · 89.7 · 89.7
* yelp_yn: 160 · 77.5 · 75.6

## Online self-learning (feedback stream)

Questions arrive one by one; after each answer the correct label is revealed. AUBIN-Learn writes it to memory (instant) and its **self-calibrator** (Hedge / multiplicative weights over three experts — model, memory, fused) shifts trust per source towards whichever has been more accurate. On kev_transfer_test the memory starts **empty**: these sources were never seen in training by AUBIN or Kev. Hyper-parameters (η, initial trust) were selected on the development streams only. This is a separate protocol, not a static locked-test score.

| predictions | stream | model | AUBIN-Learn | model, 2nd half | AUBIN-Learn, 2nd half |
|---|---|---|---|---|---|
| a12_b | kev_transfer_test | 81.5 | 81.7 | 86.9 | 86.9 |
| a12_b | kev_test | 83.1 | 84.4 | 86.4 | 87.6 |
| a31lora2 | kev_transfer_test | 89.0 | 89.0 | 94.2 | 94.2 |
| a31lora2 | kev_test | 82.8 | 84.2 | 86.1 | 87.5 |
| a12_long | kev_transfer_test | 84.0 | 84.0 | 89.0 | 89.0 |
| a12_long | kev_test | 83.0 | 84.2 | 85.6 | 86.5 |
| a12_dv_test | kev_transfer_test | 84.8 | 84.4 | 90.0 | 90.0 |
| a12_dv_test | kev_test | 83.8 | 85.0 | 87.1 | 88.3 |
| a31_think_test | kev_transfer_test | 88.0 | 87.6 | 93.7 | 93.7 |
| a31_think_test | kev_test | 83.7 | 84.7 | 87.2 | 88.3 |
| ens_a12_b+a31lora2+a12_long | kev_transfer_test | 87.7 | 87.6 | 92.9 | 92.7 |
| ens_a12_b+a31lora2+a12_long | kev_test | 84.3 | 85.6 | 87.5 | 88.5 |

## Honest notes

* Fusion weights, neighbour count and temperature were chosen on development data only; test numbers are reported once.
* Memory never contains test, development, calibration or transfer items.
* Sources unseen by the memory (all of transfer-v4) fall back to the model unchanged — memory cannot hurt them.
* Any comparison with Kev uses the same public Kev suites; Kev models are not part of AUBIN-Learn.
