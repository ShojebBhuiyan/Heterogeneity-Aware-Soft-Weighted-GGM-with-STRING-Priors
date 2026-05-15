# Method, Results, and Analysis

## 0. Project Brief

This project studies how to infer possible gene interaction networks from single-cell RNA sequencing (scRNA-seq) data. In simple terms, the data records which genes are active in each individual cell. The project asks whether that cell-by-cell information can be used to estimate which genes may be statistically connected, and whether those connections differ across biological cell states.

The core method combines three ideas:

- **Gaussian Graphical Models (GGMs):** statistical models that infer conditional relationships between genes. If two genes are connected in the inferred graph, their expression patterns remain associated after accounting for the other genes in the model.
- **Soft heterogeneity modeling:** instead of forcing each cell into exactly one group, Gaussian Mixture Models (GMMs) assign each cell probabilistic memberships across latent states.
- **STRING knowledge graph priors:** known or suspected protein/gene relationships from STRING are used as uncertain prior information, not as hard rules.

The research motivation is that single-cell data is heterogeneous. A global network over all cells can hide state-specific biology, while purely data-driven inference can be noisy. This project therefore builds a reproducible framework for comparing a global GGM baseline, soft state-specific GGMs, and STRING-guided knowledge-graph GGMs.

## 1. Dataset Information

The primary dataset is the 10x Genomics PBMC 3k dataset. PBMC stands for peripheral blood mononuclear cells, a mixture of immune cells from blood. This is a useful benchmark because it contains multiple immune cell populations rather than one homogeneous cell type.

For a lay reader, the data can be thought of as a large table:

- Each **row** is one cell.
- Each **column** is one gene.
- Each **entry** records how much that gene was observed in that cell.

Most entries are zero because scRNA-seq data is sparse: many genes are not detected in a given cell, either because they are inactive or because of technical dropout. This sparsity is normal for single-cell data, but it makes network inference difficult.

| Dataset Stage            | Cells |  Genes | Notes                                               |
| ------------------------ | ----: | -----: | --------------------------------------------------- |
| Raw ingested PBMC matrix | 2,700 | 32,738 | Full 10x matrix before feature selection            |
| Tuned processed matrix   | 2,698 |    300 | Top highly variable genes selected for GGM modeling |

| Raw Data Quality Signal |          Value | Interpretation                                                |
| ----------------------- | -------------: | ------------------------------------------------------------- |
| Sparsity                |         0.9741 | About 97.4% of matrix entries are zero, typical for scRNA-seq |
| Missing fraction        |            0.0 | No explicit missing values were detected                      |
| Dataset type            | PBMC scRNA-seq | Human immune-cell single-cell expression data                 |

The tuned analysis uses 300 highly variable genes. This follows the project guidance to avoid modeling all genes and to keep the GGM problem computationally feasible, numerically stable, and interpretable.

## 2. EDA Brief and Analysis

The EDA stage confirms that the analysis is working with a sparse, high-dimensional single-cell dataset. The tuned processed object contains 2,698 cells and 300 genes, with an existing PCA representation used for downstream visualization and clustering. The EDA artifacts are saved under `reports/figures/eda/`, with a profiling report at `reports/eda/profile_report.html`.

| EDA Item                             | Current Tuned Run |
| ------------------------------------ | ----------------: |
| Cells used                           |             2,698 |
| Genes used                           |               300 |
| PCA representation                   |  Existing `X_pca` |
| Mean of gene means in current matrix |           -0.0063 |

The EDA result supports the overall modeling direction. PBMC data should contain heterogeneous immune-cell structure, and the availability of PCA and UMAP-style artifacts makes it possible to inspect broad cell-state patterns before fitting graphical models. The strong sparsity of the original matrix also justifies preprocessing steps such as normalization, log transformation, highly variable gene selection, scaling, and PCA.

The main EDA caution is that the data is still a single benchmark dataset. It is suitable for developing and testing the method, but not enough by itself to establish general biological conclusions.

## 3. Full Methodology

The pipeline is configuration-driven and checkpointed. The tuned run uses `configs/tuning_best.yaml`, writes intermediate data under `data/`, saves model checkpoints under `data/checkpoints/tuning/best/`, and writes reports under `reports/`.

### Data Ingestion

The raw 10x PBMC matrix is loaded from the configured 10x matrix directory. Basic validation checks the AnnData object shape and confirms that the dataset has no explicit missing values. A raw/interim AnnData file is written to `data/interim/pbmc3k_raw.h5ad`.

### Preprocessing

The preprocessing stage performs the standard single-cell workflow:

1. Filter low-quality cells and genes.
2. Normalize counts to a target library size.
3. Apply log transformation.
4. Select highly variable genes.
5. Scale expression values.
6. Compute PCA for lower-dimensional modeling and visualization.

The tuned configuration uses:

| Parameter                        |  Value |
| -------------------------------- | -----: |
| Highly variable genes            |    300 |
| PCA components                   |     35 |
| Target normalization sum         | 10,000 |
| Minimum genes per cell           |    200 |
| Minimum cells per gene           |      3 |
| Maximum mitochondrial percentage |     20 |

### Heterogeneity Modeling

The pipeline computes a neighbor graph, Leiden clusters, UMAP coordinates, and a Gaussian Mixture Model over the PCA representation. The GMM produces soft memberships: each cell receives a probability of belonging to each latent state.

The tuned configuration uses 4 GMM components with tied covariance. In principle, this supports the project's soft heterogeneity assumption: cells may partially belong to multiple biological states. In the current result, however, the assignments remain close to hard labels, which is discussed in the analysis section.

### Global GGM Baseline

The global GGM fits one graphical lasso model across all cells and all selected genes. Conceptually, graphical lasso estimates a sparse precision matrix. Off-diagonal nonzero entries in that matrix become edges in the inferred gene network.

The objective can be summarized as:

```text
minimize: -log det(Theta) + trace(S Theta) + alpha * ||Theta||_1
```

Here, `S` is the empirical covariance matrix, `Theta` is the precision matrix, and `alpha` controls sparsity. Larger `alpha` generally produces fewer edges.

### Soft-Weighted State GGMs

For each GMM latent state, the pipeline computes a weighted covariance matrix. A cell contributes more to a state's covariance if that cell has high membership probability for that state. A separate graphical lasso model is then fit for each state.

This creates state-specific gene networks and enables comparison of whether inferred interactions differ across latent biological states.

### STRING Knowledge-Graph Integration

STRING interactions are fetched for the selected genes using the STRING API. Edges above the configured confidence threshold are converted into a prior matrix. The KG-biased model blends this prior into the covariance before graphical lasso fitting.

The tuned configuration uses:

| KG Parameter                | Value          |
| --------------------------- | -------------- |
| Species                     | Human (`9606`) |
| STRING confidence threshold | 0.7            |
| Maximum queried genes       | 300            |
| Prior covariance scale      | 0.02           |

This is intentionally an uncertain prior: STRING can guide the model, but the model is not forced to reproduce STRING exactly.

### Evaluation

The evaluation compares models using:

- Graph sparsity and edge counts.
- Bootstrap stability.
- STRING precision and recall.
- Pairwise similarity between state-specific graphs.
- KG-vs-soft graph differences.
- GMM entropy and component mass diagnostics.

The latest tuned run finished in approximately 42.9 seconds and wrote its selected metrics to `reports/results/tuning/runs/best_metrics.csv`.

## 4. Results

### Pipeline Configuration and Runtime

| Item                  | Value        |
| --------------------- | ------------ |
| Processed cells       | 2,698        |
| Processed genes       | 300          |
| Runtime               | 42.9 seconds |
| HVGs                  | 300          |
| PCA components        | 35           |
| GMM components        | 4            |
| GMM covariance type   | Tied         |
| Graphical lasso alpha | 0.08         |
| STRING threshold      | 0.7          |

### Graph Sparsity and Edge Counts

| Model              | Components |     Edge Count / Range | Mean Degree or Mean Edges |               Sparsity / Union |
| ------------------ | ---------: | ---------------------: | ------------------------: | -----------------------------: |
| Global GGM         |          1 |              116 edges |         Mean degree 0.773 | Upper-triangle sparsity 0.9974 |
| Soft-weighted GGMs |          4 | 38-525 edges per state |     217.5 edges per state |                861 union edges |
| KG-guided GGMs     |          4 | 39-525 edges per state |    218.25 edges per state |                862 union edges |

The tuned model is no longer nearly empty. This is a major improvement over the earlier full run with 2,000 genes, where the global graph contained only a few edges. Reducing the model to 300 highly variable genes made graph estimation more feasible.

### STRING Biological Plausibility

| Model      | STRING Precision | STRING Recall | Interpretation                              |
| ---------- | ---------------: | ------------: | ------------------------------------------- |
| Global GGM |           0.0690 |        0.1127 | Some overlap with known STRING interactions |
| Soft union |           0.0070 |        0.0845 | Many soft-network edges are outside STRING  |
| KG union   |           0.0081 |        0.0986 | Slight recall improvement over soft union   |

There were 71 STRING truth pairs at the configured threshold among the selected genes. The global graph has the strongest precision, while the KG union has slightly better recall than the soft union.

The low precision values should be interpreted carefully. STRING is not a complete ground truth for gene regulatory or expression-conditional relationships, so zero or low overlap does not automatically mean the inferred edges are false. However, STRING overlap is still a useful biological plausibility check, and these values show that biological validation remains a limitation.

### Stability and Heterogeneity Metrics

| Metric                                             |     Value | Interpretation                                                               |
| -------------------------------------------------- | --------: | ---------------------------------------------------------------------------- |
| Global bootstrap mean positive frequency           |    0.1565 | Global edges are moderately unstable across bootstrap samples                |
| Soft component 0 bootstrap mean positive frequency |    0.2540 | Selected soft-state edges are more stable than the global bootstrap baseline |
| Soft pairwise Jaccard mean                         |    0.0059 | State-specific graphs are highly distinct                                    |
| Soft pairwise Jaccard max                          |    0.0199 | Even the most similar states share few edges                                 |
| KG pairwise Jaccard mean                           |    0.0088 | KG-guided state graphs remain highly distinct                                |
| KG-vs-soft mean absolute difference                | 0.0000139 | KG prior changes the graph only slightly                                     |

The soft component bootstrap score is better than the global score, which supports the value of state-specific modeling. However, the very low Jaccard similarities also indicate that state graphs are almost disjoint. Some diversity is expected, but such low overlap may reflect unstable estimation or imbalanced mixture components rather than purely meaningful biological specialization.

### GMM Component Diagnostics

| GMM Diagnostic          |    Value | Interpretation                                            |
| ----------------------- | -------: | --------------------------------------------------------- |
| Mean entropy            |   0.1123 | Assignments have limited softness                         |
| Median entropy          |   0.0098 | Most cells are assigned almost entirely to one component  |
| Mean max probability    |   0.9528 | Average cell is about 95.3% assigned to its top component |
| Minimum component mass  |     14.0 | One component is very small                               |
| Maximum component mass  |  2,031.7 | One component dominates the mixture                       |
| Minimum ESS-style value |    53.45 | Small component has limited effective sample support      |
| Maximum ESS-style value | 2,128.83 | Dominant component has most of the evidence               |

The GMM diagnostics are the biggest warning sign. The project aims to model soft biological states, but the current GMM still behaves mostly like hard clustering and has poor component balance.

## 5. Analysis of the Results

The tuned run is a meaningful improvement over the earlier full-data attempt. The earlier 2,000-HVG run produced an almost empty global graph, weak overlap with STRING, and KG graphs that were effectively identical to non-KG soft graphs. The tuned 300-HVG configuration produces a usable global graph, non-empty soft state networks, and a measurable though still small KG effect.

The strongest positive result is feasibility. The pipeline now operates in the gene range recommended by the project brief, completes quickly, saves reproducible artifacts, and generates interpretable graph-level metrics. The global model recovers 116 edges and reaches a STRING recall of 0.1127, which is much more informative than the near-empty earlier graph.

The second positive result is that soft-state modeling produces richer networks than the global baseline. The soft networks contain 861 unique union edges across four states, and the selected soft bootstrap metric is higher than the global bootstrap metric. This suggests that state-specific weighted modeling can recover additional structure.

The main concern is biological specificity. The soft union has low STRING precision, and the KG union improves recall only modestly. The KG-vs-soft difference is extremely small, meaning the current covariance-inflation prior is not yet strongly changing the inferred networks.

The other major concern is heterogeneity quality. GMM assignments are still nearly hard, and component mass is highly imbalanced. A method intended to model soft biological states should ideally show more balanced components and more cells with meaningful mixed memberships.

Overall, the current results are best interpreted as a solid research prototype and tuned baseline, not as final biological evidence. The pipeline serves the project's engineering and methodological purpose, but the scientific claims need stronger validation.

## 6. Limitations and Future Work

### Limitations

- **Single benchmark dataset:** PBMC 3k is useful, but one dataset is not enough to establish generality.
- **Gaussian assumption:** GGMs assume approximately Gaussian-transformed expression after preprocessing. This is a simplification for sparse scRNA-seq data.
- **Low STRING precision:** The inferred networks only partly overlap with STRING, especially for soft and KG union graphs.
- **Weak KG influence:** The KG prior changes the graph only slightly in the tuned result.
- **Imbalanced GMM components:** One component has very small mass while another dominates, limiting interpretation of state-specific networks.
- **Mostly hard memberships:** The model currently does not fully realize the intended soft heterogeneity assumption.
- **STRING is incomplete:** STRING is not a perfect ground truth for conditional expression networks, so evaluation by STRING overlap is useful but incomplete.

### Future Work

- Replace covariance inflation with adaptive graphical-lasso penalties, where STRING-supported edges receive lower penalties.
- Tune or replace the GMM heterogeneity model to improve component balance and soft memberships.
- Add biological pathway or gene-set enrichment analysis for inferred edges.
- Evaluate on additional PBMC or GEO single-cell datasets.
- Compare against additional baselines such as correlation networks, GENIE3-style methods, or hard-cluster GGMs.
- Expand bootstrap evaluation to all soft and KG components, not just one selected component.
- Add expert biological review of top inferred edges, especially edges not present in STRING.

## 7. Potential Usefulness of the Project

This project is useful as a reproducible research framework for studying gene interaction inference in heterogeneous single-cell data. It does not only produce one graph; it creates a modular pipeline for comparing global, state-specific, and knowledge-guided network estimates.

Potential uses include:

- **Hypothesis generation:** inferred edges can suggest gene pairs or modules for follow-up biological investigation.
- **Method comparison:** the pipeline provides a controlled way to compare global GGMs, soft-weighted GGMs, and KG-guided GGMs.
- **Teaching and demonstration:** the project explains how probabilistic graphical models can be adapted to single-cell data.
- **Reproducible experimentation:** configuration files, checkpoints, saved metrics, and notebook orchestration make it easier to rerun and audit experiments.
- **Foundation for stronger models:** the current implementation can be extended with better priors, better mixture models, and richer validation.

The project is most useful at its current stage as a research prototype and benchmarking scaffold rather than as a finished biological discovery tool.

## 8. Conclusion

The project successfully builds an end-to-end, reproducible pipeline for heterogeneity-aware gene interaction inference from scRNA-seq data. The tuned 300-HVG run improves substantially over the earlier high-dimensional full run: it produces non-empty global and state-specific networks, records stability metrics, integrates STRING-derived priors, and saves all major outputs.

The results are promising but preliminary. The global graph shows some STRING overlap, the soft-state models recover additional network structure, and the KG model slightly improves recall over the non-KG soft union. However, low STRING precision, weak KG-vs-soft differences, imbalanced GMM components, and mostly hard cell assignments limit the strength of the biological claims.

In summary, the project currently serves its purpose as a serious, extensible research framework and tuned baseline. The next step is not more pipeline plumbing, but stronger modeling of heterogeneity and more direct use of biological priors so that the inferred networks become more stable, more biologically interpretable, and more defensible as scientific results.
