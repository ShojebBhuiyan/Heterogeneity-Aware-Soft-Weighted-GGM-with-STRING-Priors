========================================================
STRING API INTEGRATION REQUIREMENTS
========================================================

IMPORTANT:
DO NOT download the full STRING database locally due to size constraints.

Instead, integrate the STRING API dynamically and retrieve only interactions relevant to the genes selected during preprocessing.

Use the official STRING API:
https://string-db.org/help/api/

========================================================
OBJECTIVE
========================================================

Implement a lightweight, scalable knowledge graph retrieval system that:

- queries STRING dynamically
- retrieves only relevant interactions
- supports confidence filtering
- caches responses locally
- avoids redundant API calls
- integrates seamlessly into the KG pipeline

========================================================
API IMPLEMENTATION REQUIREMENTS
========================================================

Create a dedicated module:

src/kg/string_api.py

The module should:

1. Query STRING interactions for selected genes
2. Handle batching for large gene lists
3. Cache API responses locally
4. Retry failed requests gracefully
5. Support configurable confidence thresholds
6. Normalize confidence scores
7. Save processed interaction networks
8. Convert STRING outputs into graph-ready structures

========================================================
TARGET GENE SELECTION
========================================================

DO NOT query all genes.

Instead:

- select top highly variable genes
- default range:
  - 100–500 genes
- configurable via config

The API should only retrieve interactions among selected genes.

========================================================
API ENDPOINT
========================================================

Use:

https://string-db.org/api/tsv/network

Species:

- human (9606)

========================================================
API REQUEST EXAMPLE
========================================================

Example format:

https://string-db.org/api/tsv/network?identifiers=TP53%0dBRCA1%0dMYC&species=9606

Implement proper URL encoding.

========================================================
EXPECTED RETURNED FIELDS
========================================================

Extract and store:

- preferredName_A
- preferredName_B
- score

Optional:

- experimental
- database
- textmining

========================================================
CONFIDENCE HANDLING
========================================================

STRING confidence scores are in range:
0–1

Implement:

- configurable threshold filtering
- default threshold:
  0.7

Allow experimentation with:

- 0.4
- 0.7
- 0.9

========================================================
CACHING REQUIREMENTS
========================================================

Implement local caching to avoid repeated API calls.

Suggested cache structure:

data/external/string_cache/

Requirements:

- hash gene lists into cache keys
- reuse cached responses automatically
- support cache invalidation

========================================================
RATE LIMITING + ERROR HANDLING
========================================================

Implement:

- retry logic
- exponential backoff
- timeout handling
- partial batch recovery
- request logging

Gracefully handle:

- network failures
- malformed responses
- empty interaction returns
- API throttling

========================================================
GRAPH CONSTRUCTION
========================================================

Convert STRING interactions into:

1. pandas edge table
2. networkx graph
3. adjacency matrix
4. weighted prior matrix

The weighted prior matrix will later be used for:
knowledge-guided regularization.

========================================================
VISUALIZATION REQUIREMENTS
========================================================

Generate:

- interaction network plots
- confidence score histograms
- degree distributions
- overlap with inferred GGM edges

Save outputs to:
reports/figures/kg/

========================================================
CHECKPOINTING
========================================================

Save:

- raw API responses
- processed edge tables
- graph objects
- prior matrices

Use:

- parquet
- pickle
- joblib

Notebook execution should:

- skip API requests if cache/checkpoint exists

========================================================
CONFIGURATION SUPPORT
========================================================

Expose configurable parameters:

- species_id
- confidence_threshold
- batch_size
- request_timeout
- max_retries
- cache_enabled
- max_genes_for_query

========================================================
GIT REQUIREMENTS
========================================================

After implementing STRING API integration:

- validate retrieval
- generate sample graph
- test caching
- create git commit

Commit message:
"Implement scalable STRING API knowledge graph integration"

========================================================
IMPORTANT DESIGN CONSTRAINTS
========================================================

- Avoid loading unnecessary genome-scale interactions
- Prioritize memory efficiency
- Design for reproducibility
- Ensure integration is modular and reusable
- Keep API logic separate from modeling logic
- Ensure notebook reproducibility with checkpoint-aware execution

========================================================
FINAL EXPECTED OUTPUT
========================================================

The system should produce:

1. Retrieved STRING interactions for selected genes
2. Confidence-weighted biological prior graph
3. Cached reusable KG artifacts
4. Visualized interaction structures
5. Reproducible API-driven KG pipeline
