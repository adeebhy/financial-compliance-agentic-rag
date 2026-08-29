# Financial Compliance Agentic RAG & Credit Risk Model

An end-to-end framework combining a Quantitative Credit Scorecard with an Agentic RAG system for regulatory compliance (SR 11-7, ECOA Reg B).

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```
## Overview
An end-to-end framework combining an interpretable **Credit Risk Scorecard** (Probability of Default) with an **Agentic Regulatory Compliance Audit Engine**. 

The system leverages Small Language Models (SLMs) and FAISS-based Retrieval-Augmented Generation (RAG) to automatically audit quantitative credit models against key banking regulations:
- **Federal Reserve SR 11-7:** Model Risk Management, conceptual soundness, outcomes analysis, and stability benchmarking.
- **CFPB ECOA / Regulation B (12 CFR Part 1002):** Fair lending checks and disparate impact testing (4/5ths rule).
- **Basel III/IV Internal Ratings-Based (IRB):** Probability of default calibration and discriminatory power minimums.

---

## Architecture

1. **Quantitative Modeling Engine (`src/credit_engine.py`)**
   - Baseline: L2-Regularized Logistic Regression on UCI Default of Credit Card Clients data.
   - Metrics Suite: ROC-AUC, Kolmogorov-Smirnov (KS) statistic, Brier calibration score.
   - Population Stability Index (PSI) computation to detect covariate drift.
   - Disparate impact ratio calculation across protected demographic segments.

2. **Regulatory Vector Store (`src/vector_store.py`)**
   - Local FAISS vector index chunking regulatory policy documents.
   - Semantic retrieval of relevant regulatory clauses using lightweight embeddings (`all-MiniLM-L6-v2`).

3. **Agentic Compliance Orchestrator (`src/compliance_agent.py` & `src/tools.py`)**
   - Deterministic rule checks paired with an SLM reasoning layer.
   - Automated generation of a formal **Model Risk Management (MRM) Validation Memorandum**.

