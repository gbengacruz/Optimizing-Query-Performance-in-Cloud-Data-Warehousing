**A Comparative Analysis of Microsoft Azure Synapse, Amazon Redshift, and Oracle Autonomous Data Warehouse**

---

## 📌 Project Overview

Cloud data warehouses are now central to modern analytics, yet organisations—particularly SMEs—often struggle to choose the most suitable platform due to conflicting benchmark results and unclear performance guidance.

This research provides a **neutral, controlled, and empirical comparison** of three leading cloud data warehousing platforms:

* **Microsoft Azure Synapse Analytics**
* **Amazon Redshift**
* **Oracle Autonomous Data Warehouse (ADW)**

Using industry-standard **TPC-DS and TPC-H benchmarks (10GB scale)**, the study evaluates baseline performance, optimization effectiveness, and concurrency behaviour under SME-representative constraints.

---

## 🎯 Research Objectives

* Benchmark Azure Synapse, Amazon Redshift, and Oracle ADW under identical conditions
* Evaluate platform-specific optimization strategies (compression, distribution keys, materialized views)
* Analyse scalability and concurrency behaviour
* Provide an **evidence-based decision framework** for SMEs selecting cloud data warehouses

---

## 🔍 Research Gap Addressed

* Conflicting benchmark results in existing literature
* Lack of neutral, tri-platform comparisons under controlled configurations
* Limited practical guidance for SMEs balancing cost, performance, and operational complexity

---

## 🧪 Methodology

A **seven-phase experimental pipeline** was adopted, featuring:

* **Benchmarks:**

  * TPC-DS (99 queries, 24 tables)
  * TPC-H (22 queries, 8 tables)
* **Scale Factor:** 10GB (SME-representative)
* **Execution Model:** Cold and warm run separation
* **Iterations:** 20 executions per query

### 📊 Statistical Analysis

* **Kruskal–Wallis H-test** (α = 0.05)
* **Dunn’s post-hoc pairwise comparisons**
* **Cliff’s Delta effect sizes** for practical significance

---

## 🚀 Key Findings

* **Oracle ADW** delivered the lowest median latency and exceptional consistency, with **12.2× concurrency scaling**.
* **Amazon Redshift** showed competitive performance, particularly on TPC-H workloads, but required manual DBA tuning.
* **Azure Synapse (DW200c)** proved inadequate for production workloads, with optimization often degrading performance due to resource constraints.

**Key Insight:** Optimization effectiveness is **highly platform- and resource-dependent**. Autonomous optimization consistently outperformed manual tuning under multi-user workloads.

---

## 🧠 Contributions

✔ Neutral tri-platform benchmarking under controlled conditions
✔ Explicit measurement of optimization marginal effects
✔ Telemetry-driven performance root cause analysis
✔ Practical, SME-focused decision framework

---

## ⚠️ Limitations

* 10GB scale factor limits enterprise-scale generalization
* Single-region deployment (US-East)
* Temporal validity (October 2025 snapshot)

---

## 🔮 Future Work

* Larger scale factors (SF100–SF1000)
* Multi-region and cost-per-query analysis
* Real-world workload traces
* Evaluation of Azure Synapse Spark pools

---

## 📂 Repository Structure

```
├── datasets/          # TPC-DS and TPC-H data generation scripts
├── sql/               # Benchmark and optimization SQL queries
├── results/           # Raw and aggregated performance results
├── analysis/          # Statistical analysis notebooks
├── figures/           # Charts and plots used in dissertation
└── README.md          # Project overview
```

---

## 📖 Citation

If you use this work, please cite it appropriately as part of academic or research outputs.

---

⭐ *This repository supports MSc-level research into cloud data warehousing performance and optimization.*
