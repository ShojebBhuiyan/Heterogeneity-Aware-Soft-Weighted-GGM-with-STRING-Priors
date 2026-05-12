# Project Details

## Project Title

Heterogeneity-Aware Gene Interaction Inference using Soft-Weighted Gaussian Graphical Models with Uncertain Knowledge Graph Priors

---

# Project Overview

This project focuses on probabilistic graphical models (PGMs) for gene interaction inference using single-cell RNA-seq (scRNA-seq) data.

The primary goal is to develop a heterogeneity-aware framework capable of learning biologically meaningful gene interaction networks by combining:

1. Soft-weighted Gaussian Graphical Models (GGMs)
2. Single-cell heterogeneity modeling
3. Uncertain biological knowledge graph priors

The project is intended to be:

- research-oriented
- modular
- reproducible
- extensible
- publication-quality

This is NOT just a course assignment implementation. The codebase should resemble a serious academic/research repository.

---

# Research Motivation

Traditional gene network inference methods often assume:

- homogeneous data distributions
- equal contribution from all samples
- no prior biological knowledge

These assumptions fail in modern single-cell biological datasets because:

- different cell types exhibit different interaction structures
- biological states are continuous rather than discrete
- purely data-driven inference is noisy

The project aims to address these limitations using:

- soft clustering
- weighted covariance estimation
- knowledge-guided regularization

---

# Core Research Question

How can probabilistic graphical models be extended to:

1. handle biological heterogeneity in single-cell data
2. integrate uncertain biological prior knowledge
3. improve stability and biological plausibility of inferred gene interaction networks

---

# Core Methodology

## 1. Data Processing

Use single-cell RNA-seq datasets:

- primarily 10x Genomics PBMC data
- optionally GEO datasets for validation

Preprocessing steps:

- quality control
- filtering
- normalization
- log transformation
- highly variable gene selection
- scaling
- PCA

---

## 2. Heterogeneity Modeling

Model latent biological states using:

- Gaussian Mixture Models (GMMs)

Instead of hard cluster assignment:

- each cell receives probabilistic memberships

Example:

Cell A:

- 70% state 1
- 20% state 2
- 10% state 3

This creates soft cluster memberships.

---

## 3. Soft-Weighted Gaussian Graphical Model

Traditional GGM:

- learns one global precision matrix

Proposed approach:

- estimate weighted covariance matrices per latent state
- use soft membership probabilities as weights

Weighted covariance intuition:

Each cell contributes proportionally to multiple latent states instead of belonging to a single cluster.

This enables:

- heterogeneous interaction modeling
- state-specific gene interaction networks

---

## 4. Knowledge Graph Integration

Incorporate uncertain biological priors from STRING.

STRING provides:

- gene/protein interaction edges
- confidence scores

These confidence scores are used as:

- soft regularization priors
- NOT hard constraints

The KG guides the model toward biologically plausible interactions while preserving discovery of novel edges.

---

# Knowledge Graph Source

Use:

- STRING API

DO NOT download full STRING datasets locally.

Instead:

- dynamically retrieve interactions only for selected genes

Use:
https://string-db.org/help/api/

Species:

- Human (9606)

---

# Gene Selection Strategy

DO NOT model all genes.

Instead:

- select top highly variable genes

Recommended range:

- 100–500 genes

Reason:

- computational feasibility
- numerical stability
- interpretability
- manageable graph estimation

---

# Datasets

## Primary Dataset

10x Genomics PBMC scRNA-seq dataset

Useful because:

- contains heterogeneous immune cell populations
- widely used benchmark
- suitable size for experimentation

---

## Optional Secondary Dataset

GEO datasets for:

- validation
- robustness testing

---

# Why Single-cell RNA-seq Instead of Bulk RNA-seq

Bulk RNA-seq:

- averages expression across cells
- destroys heterogeneity information

Single-cell RNA-seq:

- preserves cell-level variability
- enables state-specific interaction modeling
- supports soft clustering

This is essential for the proposed methodology.

---

# Expected Outputs

The project should produce:

1. Preprocessed scRNA-seq datasets
2. Exploratory Data Analysis reports
3. Soft cluster assignments
4. Global GGM baseline networks
5. Soft-weighted GGM networks
6. Knowledge-guided networks
7. Evaluation metrics
8. Publication-quality visualizations
9. Reproducible checkpoints
10. End-to-end executable notebooks

---

# Evaluation Goals

Evaluate models using:

## 1. Network Stability

- bootstrap consistency
- reproducibility

## 2. Biological Plausibility

- overlap with STRING interactions

## 3. Network Diversity

- compare latent-state-specific networks

## 4. Sparsity Analysis

- compare graph structures

---

# Expected Research Findings

Expected outcomes include:

- improved stability compared to global GGMs
- biologically meaningful network structures
- distinct latent-state interaction patterns
- improved overlap with known biological interactions

---

# Important Research Assumptions

## Gaussian Assumption

The model assumes approximately Gaussian-transformed gene expression after preprocessing.

## Soft Heterogeneity Assumption

Cells may belong to multiple latent biological states simultaneously.

## Probabilistic Prior Assumption

Knowledge graph edges are uncertain and should influence regularization probabilistically.

---

# Important Constraints

The implementation should:

- prioritize reproducibility
- support checkpoint recovery
- avoid recomputation
- remain memory efficient
- support modular experimentation
- avoid monolithic notebook logic

---

# Engineering Requirements

The implementation MUST:

- be modular
- use logging
- use checkpointing
- use configuration files
- support reproducibility
- include plotting pipelines
- include EDA reports
- support notebook resume functionality
- save intermediate outputs
- include error handling

---

# Notebook Requirements

Create notebooks for:

1. EDA
2. preprocessing
3. clustering
4. global GGM
5. soft-weighted GGM
6. KG integration
7. evaluation
8. final pipeline

Notebook execution should:

- detect checkpoints
- skip completed stages automatically
- support resume workflows

---

# Visualization Requirements

Generate:

- PCA plots
- UMAP plots
- clustering visualizations
- graph heatmaps
- adjacency matrix plots
- edge overlap visualizations
- network sparsity plots
- stability plots
- KG overlap visualizations

All figures should be publication-quality.

---

# Software Engineering Expectations

The codebase should:

- follow clean architecture
- use reusable modules
- separate notebooks from logic
- use type hints
- use docstrings
- support future extensions

---

# Git Requirements

After each major implementation stage:

1. validate outputs
2. execute notebook tests
3. create git commit

Commit messages should clearly describe implemented functionality.

---

# Key References

1. Friedman, Hastie, Tibshirani (2008)
   Sparse inverse covariance estimation with the graphical lasso

2. Mourad et al. (2012)
   Probabilistic graphical models for genetic association studies

3. Wei & Li (2007)
   Markov random field model for genomic data

4. Wang et al. (2003)
   MGraph for microarray data analysis

5. Tang et al. (2023)
   spaCI adaptive graph model

---

# Final Goal

Develop a reproducible, research-grade framework that combines:

- probabilistic graphical models
- soft heterogeneity modeling
- uncertain biological priors

to improve gene interaction inference in single-cell genomic data.
