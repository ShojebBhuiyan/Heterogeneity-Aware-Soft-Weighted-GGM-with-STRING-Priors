You are implementing a research-grade computational biology project focused on probabilistic graphical models for gene interaction inference from single-cell RNA-seq data.

The project goal is to build a modular, reproducible, extensible pipeline for:

1. preprocessing scRNA-seq data
2. exploratory data analysis
3. heterogeneity-aware clustering
4. soft-weighted Gaussian Graphical Model (GGM) inference
5. uncertain knowledge graph integration
6. evaluation and visualization
7. notebook-based experimentation
8. reproducibility and checkpoint recovery

The implementation quality should resemble a serious academic research repository and follow software engineering best practices.

========================================================
PROJECT TITLE
========================================================

Heterogeneity-Aware Gene Interaction Inference using Soft-Weighted Gaussian Graphical Models with Uncertain Knowledge Graph Priors

========================================================
HIGH LEVEL OBJECTIVES
========================================================

Build an end-to-end pipeline that:

- Uses single-cell RNA-seq datasets (10x Genomics PBMC initially)
- Learns gene interaction networks using Gaussian Graphical Models
- Models heterogeneity using soft clustering (Gaussian Mixture Models)
- Integrates uncertain biological knowledge from STRING
- Evaluates network quality and stability
- Produces publication-quality visualizations and reports
- Supports checkpointing and resume functionality
- Maintains modular architecture
- Automatically commits progress to git after every major implementation milestone

========================================================
MANDATORY IMPLEMENTATION REQUIREMENTS
========================================================

The implementation MUST:

- be modular
- follow clean architecture
- use configuration-driven execution
- include extensive logging
- include robust error handling
- include unit-testable components
- support reproducibility
- support checkpoint saving/loading
- support notebook resumability
- generate plots automatically
- generate EDA reports
- save intermediate outputs
- use deterministic seeds where appropriate

========================================================
TECH STACK
========================================================

Language:

- Python 3.11+

Core libraries:

- scanpy
- anndata
- numpy
- scipy
- pandas
- scikit-learn
- networkx
- matplotlib
- plotly
- seaborn
- tqdm
- joblib
- pyarrow
- statsmodels

Optimization / graphical models:

- sklearn.covariance.GraphicalLasso
- optional:
  - skggm
  - cvxpy

Knowledge graph:

- STRING DB interaction files

Notebook:

- Jupyter Notebook
- ipywidgets optional

EDA:

- ydata-profiling or sweetviz

Configuration:

- pydantic or hydra or yaml config system

Logging:

- python logging module
- rich logging preferred

Testing:

- pytest

========================================================
REPOSITORY STRUCTURE
========================================================

Create a professional repository structure:

project_root/
│
├── data/
│ ├── raw/
│ ├── processed/
│ ├── checkpoints/
│ ├── external/
│ └── interim/
│
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_clustering.ipynb
│ ├── 04_global_ggm.ipynb
│ ├── 05_soft_weighted_ggm.ipynb
│ ├── 06_kg_integration.ipynb
│ ├── 07_evaluation.ipynb
│ └── final_pipeline.ipynb
│
├── reports/
│ ├── figures/
│ ├── eda/
│ └── results/
│
├── src/
│ ├── config/
│ ├── data/
│ ├── preprocessing/
│ ├── clustering/
│ ├── models/
│ ├── kg/
│ ├── evaluation/
│ ├── visualization/
│ ├── utils/
│ └── pipelines/
│
├── tests/
│
├── logs/
│
├── configs/
│
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── README.md
└── .gitignore

========================================================
IMPLEMENTATION PHASES
========================================================

Implement the project in incremental phases.

IMPORTANT:
After EACH major implementation phase:

1. run basic validation
2. ensure notebook execution works
3. create a git commit
4. write meaningful commit messages

========================================================
PHASE 1 — PROJECT INITIALIZATION
========================================================

Tasks:

- initialize repository
- setup environment
- setup logging
- setup config system
- create utility helpers
- setup reproducibility helpers
- setup checkpoint utilities
- setup plotting theme
- create README draft

Checkpoint utilities must:

- save intermediate numpy/pandas/anndata/model outputs
- support loading by stage name
- prevent recomputation if checkpoint exists

Notebook helper:

- allow notebook cells to skip execution if checkpoint already exists

Commit:
"Initialize project structure and infrastructure"

========================================================
PHASE 2 — DATA INGESTION
========================================================

Implement:

- automatic download or ingestion of:
  - 10x PBMC dataset
  - optional GEO dataset
- validation checks
- schema checks
- metadata extraction

Generate:

- dataset summary
- missing value analysis
- sparsity analysis

Save:

- raw checkpoints
- processed checkpoints

Commit:
"Implement dataset ingestion and validation pipeline"

========================================================
PHASE 3 — EXPLORATORY DATA ANALYSIS
========================================================

Generate a full EDA pipeline.

Include:

- dimensionality overview
- sparsity heatmaps
- gene expression distributions
- cell quality metrics
- variance analysis
- PCA visualization
- UMAP visualization
- clustering previews
- top variable genes
- correlation structure previews

Automatically generate:

- HTML EDA report
- PNG/PDF figures
- markdown summary

Save all outputs in:
reports/eda/

Commit:
"Add comprehensive EDA and visualization pipeline"

========================================================
PHASE 4 — PREPROCESSING PIPELINE
========================================================

Implement:

- filtering low-quality cells
- filtering low-quality genes
- normalization
- log transformation
- highly variable gene selection
- scaling
- PCA

Support:

- configurable thresholds
- checkpoint recovery

Produce:

- processed AnnData object

Commit:
"Implement preprocessing and feature engineering pipeline"

========================================================
PHASE 5 — HETEROGENEITY MODELING
========================================================

Implement:

- Leiden clustering baseline
- Gaussian Mixture Model soft clustering

Store:

- hard labels
- soft probabilities

Visualize:

- cluster distributions
- latent memberships
- soft assignment entropy

Commit:
"Implement heterogeneity-aware clustering models"

========================================================
PHASE 6 — GLOBAL GGM
========================================================

Implement:

- global Graphical Lasso baseline
- covariance estimation
- precision matrix estimation
- adjacency extraction

Generate:

- network statistics
- sparsity plots
- degree distributions
- heatmaps

Commit:
"Implement baseline global Gaussian graphical model"

========================================================
PHASE 7 — SOFT-WEIGHTED GGM
========================================================

Implement:

- weighted covariance estimation
- per-component precision estimation
- soft membership weighted learning

Support:

- configurable alpha
- configurable component count

Generate:

- separate network per latent state
- network comparison visualizations
- edge overlap analysis

IMPORTANT:
Design implementation to be numerically stable.

Include:

- covariance regularization
- matrix conditioning safeguards
- error handling for singular matrices

Commit:
"Implement soft-weighted Gaussian graphical model"

========================================================
PHASE 8 — KNOWLEDGE GRAPH INTEGRATION
========================================================

Implement:

- STRING data ingestion
- confidence score parsing
- graph prior construction
- prior-weighted regularization

Support:

- filtering by confidence threshold
- prior/no-prior comparison

Visualize:

- overlap with known interactions
- prior influence analysis

Commit:
"Integrate uncertain biological knowledge graph priors"

========================================================
PHASE 9 — EVALUATION FRAMEWORK
========================================================

Implement:

- bootstrap stability analysis
- edge reproducibility
- graph similarity metrics
- network sparsity analysis
- component diversity metrics

Generate:

- comparison tables
- statistical summaries
- plots

Export:

- CSV metrics
- publication-quality figures

Commit:
"Implement evaluation and benchmarking framework"

========================================================
PHASE 10 — FINAL NOTEBOOK
========================================================

Create:
notebooks/final_pipeline.ipynb

The notebook must:

- execute entire pipeline sequentially
- support resume from checkpoints
- avoid recomputation
- include markdown explanations
- include equations where appropriate
- include plots inline
- produce final outputs automatically

Add:

- runtime diagnostics
- execution summaries
- configuration snapshots

Commit:
"Create end-to-end reproducible research notebook"

========================================================
SOFTWARE ENGINEERING REQUIREMENTS
========================================================

The codebase MUST:

- use type hints
- use docstrings
- avoid duplicated logic
- separate business logic from notebooks
- keep notebooks orchestration-focused
- avoid hardcoded paths
- centralize configs
- support future extension

========================================================
LOGGING REQUIREMENTS
========================================================

Every major stage must log:

- start/end
- runtime
- shapes/dimensions
- memory warnings
- checkpoint saves
- failures/recovery

Use structured logs where possible.

========================================================
ERROR HANDLING
========================================================

Handle:

- missing files
- malformed datasets
- singular covariance matrices
- failed model convergence
- memory issues
- invalid configuration

Gracefully recover where possible.

========================================================
PLOTTING REQUIREMENTS
========================================================

Generate publication-quality plots for:

- PCA
- UMAP
- cluster distributions
- graph heatmaps
- adjacency structures
- edge overlap
- network stability
- sparsity comparisons
- prior influence analysis

Save:

- PNG
- SVG
- PDF where feasible

========================================================
CHECKPOINTING REQUIREMENTS
========================================================

Every stage must support:

- save_checkpoint()
- load_checkpoint()

Use:

- joblib
- parquet
- AnnData h5ad
- pickle where appropriate

Notebook cells should:

- detect existing checkpoint
- skip completed computation automatically

========================================================
REPRODUCIBILITY REQUIREMENTS
========================================================

Implement:

- global random seed setup
- environment capture
- config snapshotting
- deterministic execution where feasible

========================================================
FINAL DELIVERABLES
========================================================

Produce:

1. Modular codebase
2. Executable notebooks
3. EDA report
4. Publication-quality plots
5. Final result summaries
6. Saved trained models
7. Reproducible checkpoints
8. Git commit history showing progressive development

========================================================
IMPORTANT IMPLEMENTATION STYLE
========================================================

- Write production-quality research code
- Prioritize clarity and reproducibility
- Use incremental implementation
- Test each phase before proceeding
- Maintain clean git history
- Do not place heavy logic inside notebooks
- Keep notebooks lightweight and reproducible
- Ensure all outputs are saved automatically

Begin implementation from Phase 1 and proceed incrementally.
