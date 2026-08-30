# Fraud/Risk Flag Agent — Razorpay AI Buildathon 2026
### Track 02: AI Risk Manager

**"Stop the merchant losing money to fraud, returns and chargebacks."**

A three-agent fraud detection system for merchant transactions, built directly against Track 02's stated bar: *"Honest metrics including false-positive cost. Strictly defense-only: anything offense-capable is disqualified."* Every design decision in this repo traces back to that sentence.

---

## What this is, in one paragraph

Most fraud-detection prototypes stop at a trained model and a headline accuracy number. This system adds an independent **Reviewer Agent** that checks whether a score's confidence is actually earned by its evidence before any decision is finalized, a **Pattern/Evasion Agent** that flags statistical anomalies and threshold-gaming attempts (defensive only — never modeling fraud techniques), and a decision layer whose thresholds are chosen against a real, cited cost model, not a default or a guess.

---

## Architecture

```
Real transaction data (Razorpay test-mode customers/orders, or the
credit_card_fraud_2026.csv training set)
        |
        v
Feature pipeline -- 23 features, thresholds derived from actual data
        |
        v
Agent 1 -- Pattern/Evasion Agent
        |   Two modes: per-entity historical drift (compares a
        |   transaction to that specific customer's own history, when
        |   available) and population-level anomaly + threshold-evasion
        |   detection (fallback). Defensive framing only -- see
        |   "Defensive-use statement" below.
        v
Agent 2 -- Detection & Scoring Agent
        |   Random Forest (300 estimators, class_weight=balanced),
        |   trained on a held-out 80/20 split. Evidence generated via
        |   SHAP TreeExplainer -- real per-transaction feature
        |   contributions, not hand-picked rule weights.
        v
Agent 3 -- Reviewer Agent
        |   Independent evidence-strength check -- NOT a re-score.
        |   Checks evidence count vs. score magnitude, evidence
        |   diversity across signal categories, and contradiction
        |   strength. Consumes {score, evidence}, model-agnostic by
        |   design -- Agent 2 could be swapped for any upstream scorer.
        v
Decision Router
        |   auto_approve / escalate_to_human / auto_reject. Thresholds
        |   chosen by sweeping a real, cited cost model (see below) --
        |   plus a hard rule: a downgraded or insufficient-evidence
        |   Reviewer verdict can never auto-reject, regardless of the
        |   raw score.
        v
Audit Log -- append-only, every stage logged with timestamp and actor
```

### Why three agents, not one model + SHAP?

SHAP explains *what* drove a score. It doesn't tell you whether the score itself is trustworthy given how the evidence combines. A high score resting on one weak signal, or on several redundant signals from the same category, or on evidence that contradicts other facts about the transaction, can all produce a SHAP explanation that looks reasonable while the underlying confidence isn't earned. Agent 3 exists specifically to catch that.

---

## Measured results, held-out test set

Served live at `GET /metrics`, not just claimed in this README:

| Metric | Value |
|---|---|
| Model | RandomForestClassifier (n_estimators=300, class_weight=balanced) |
| Held-out test set | 4,000 transactions (20% split, stratified, random_state=42) |
| AUC | 0.9266 |
| Cost-optimal threshold | 0.01 |
| Recall at optimal threshold | 100% (0 missed fraud cases) |
| Precision at optimal threshold | 4.55% |
| Total cost at optimal threshold | ₹133,950 |
| Total cost at default (0.5) threshold | ₹2,366,536 |
| Cost reduction | 94% |

### The cost model — sourced, not assumed

- **False negative (missed fraud) = ₹34,802** — average value of card/internet banking fraud in India, FY22, from official Lok Sabha data as reported by Business Standard. Specific to card/internet fraud, not the larger loan/advances category that dominates more recent aggregate RBI figures.
- **False positive (unnecessary manual review) = ₹94** — ~15 minutes at the average Indian fraud analyst hourly rate (₹377/hr, ERI SalaryExpert compensation data).
- **Ratio: ~370:1.** Missing one fraud case costs roughly 370x more than one unnecessary review — which is why the system is tuned for recall, and why the resulting false-positive volume is a deliberate, cost-justified trade-off, not an oversight. See `scripts/train_model.py` for full citations and the threshold sweep.

---

## Real Razorpay integration

Beyond the training dataset, this system was tested against **genuinely real Razorpay test-mode data** — 8 real customers and 70 real orders, created via the `razorpay-python` SDK against Razorpay's actual test-mode API (`scripts/generate_razorpay_data.py`), verifiable live in the Razorpay Dashboard. This real customer/order linkage is what powers Agent 1's per-entity drift mode: a real customer's transaction is compared against their own real transaction history, not a population-wide average.

Honest scope note: Razorpay's Orders API doesn't track fraud-relevant behavioral fields (velocity, merchant risk score, etc.) -- those don't exist on a real Order object. For the entity-drift demo, real customer/order identifiers and timestamps are paired with behavioral feature values sampled from the training dataset's actual distributions, clearly labeled as simulated wherever referenced (`scripts/build_entity_demo_data.py`). Nothing about this is presented as more real than it is.

---

## Engineering notes

A few things worth calling out from the build process, not because things went wrong, but because catching them is the point:

- **The rule-based Phase 1 scorer and the trained model disagreed on which signal mattered most for the same transaction** -- the hand-picked rule weights leaned heavily on `merchant_risk_score`; the trained model's SHAP output didn't rank it in its top contributors at all for that case. Neither is "wrong" -- it's a concrete demonstration of why training on real data surfaces things intuition-based weighting misses.
- **A SHAP evidence-strength calibration bug** was caught by reading actual output values, not just checking pass/fail on tests: normalizing SHAP contribution strength relative to the top signal *within* a transaction was inflating minor noise into misleadingly large "contradicting evidence." Fixed by using raw contribution magnitude instead, comparable across transactions.
- **Full automated test suite (18 tests)** covers all three agents and the full pipeline, rewritten against the real ML scorer's actual behavior, not hardcoded to expected-but-unverified numbers -- one early test assertion was itself wrong (expected evidence that didn't actually cross the real threshold), caught and fixed the same way.

Full build log, including every bug and how long each took to resolve, is in `debug-log.md`.

---

## Repository structure

```
app/
|-- main.py                       FastAPI app, mounts all routers
|-- models.py                      Pydantic schemas matched to the real dataset columns
|-- storage.py                     In-memory store (Phase 1/2; real DB planned)
|-- features.py                    Feature pipeline, data-derived thresholds
|-- decision.py                     Decision Router
|-- audit.py                        Append-only audit logging (in-memory + JSONL)
|-- ml_artifacts/
|   |-- model.pkl                   Trained RandomForestClassifier
|   |-- encoders.pkl                Fitted LabelEncoders
|   `-- training_results.txt        Real metrics, read live by /metrics
|-- agents/
|   |-- pattern_agent.py            Agent 1 -- entity-drift + population fallback
|   |-- scoring_agent.py            Agent 2 -- Random Forest + SHAP evidence
|   `-- reviewer_agent.py           Agent 3
`-- routers/                        ingest, score, review, decisions, audit_log,
                                     metrics, health, process, entity_drift_demo
data/
|-- credit_card_fraud_2026.csv      Training dataset, 20k txns, 1.7% fraud rate
`-- demo/
    |-- razorpay_real_test_data.json     Real Razorpay customers/orders
    `-- merged_entity_demo.json          Real linkage + simulated features
scripts/
|-- train_model.py                  Model training, cost-optimal threshold sweep
|-- generate_razorpay_data.py       Real Razorpay test-mode data generation
`-- build_entity_demo_data.py       Merges real linkage with simulated features
tests/                               18 tests, unit + end-to-end
debug-log.md                         Full build log: every bug, time cost, lesson
```

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/transactions/ingest` | Ingest a transaction, run feature pipeline |
| `POST` | `/transactions/{id}/score` | Run Agent 1 + Agent 2 |
| `POST` | `/transactions/{id}/review` | Run Agent 3 |
| `POST` | `/decisions/{id}` | Run the Decision Router |
| `POST` | `/transactions/{id}/process` | Run the full pipeline in one call |
| `GET` | `/audit-log` | Retrieve audit trail, filterable by transaction |
| `GET` | `/metrics` | Real precision/recall/AUC/cost data, sourced |
| `GET` | `/demo/entity-drift/customers` | List real Razorpay demo customers |
| `POST` | `/demo/entity-drift/{customer_id}` | Run Agent 1's entity-drift mode on real customer data |
| `GET` | `/health` | Liveness check |

---

## What's fully built vs. planned

**Built and verified -- live, not just claimed:**
- Full three-agent pipeline: rule-based Phase 1 baseline + trained ML Phase 2, both working
- Real Random Forest model with SHAP-based, per-transaction evidence
- Cost-optimal decision threshold, sourced from two independent Indian industry figures
- Real Razorpay test-mode customer/order data, both agent modes demonstrated against it
- Append-only audit trail, all pipeline stages confirmed logged
- 18 automated tests, full pipeline + all three agents
- `/metrics` serving real, cited numbers live

**Planned:**
- Frontend/dashboard UI
- Real database (currently in-memory, resets on restart)
- Live integration testing against a running `razorpay-mcp-server` -- deliberately out of scope for the build; the Python SDK integration (real customers/orders) covers the actual data-access need

---

## Non-negotiables

1. Measured precision/recall on a held-out set -- met (`/metrics`)
2. False-positive cost reasoning tied to the threshold choice, honestly sourced -- met
3. Explicit defense-only framing for Agent 1 -- met (see below)
4. Honest "built vs. designed" distinction -- met, this section

---

## Defensive-use statement

Agent 1 performs statistical drift and evasion detection only. It does not model, simulate, or output fraud techniques, evasion strategies, or attacker behavior in any form. Stated here and in the module docstring (`app/agents/pattern_agent.py`), consistent with Track 02's requirement that anything offense-capable is disqualified.

---

## Disclaimer

This is a buildathon submission. The primary training dataset is synthetic; the Razorpay integration uses real test-mode data with clearly-labeled simulated behavioral features layered on top, since fraud-relevant signals aren't part of a real Order object. Cost figures are sourced from named public studies, not Razorpay-internal data. This is not a production fraud-detection system and has not been evaluated against live payment data.
