# LightRAG Rule Module

## Overview

The `rule` module provides business rules, extraction logic, and workflow management for the electrical_controlled mode in LightRAG.

## Directory Structure

```
lightrag/rule/
├── __init__.py          # Package exports (Workflow classes only)
├── README.md            # This file
├── utils.py             # Common utility functions and helpers
├── tree_override.py     # Annotation memory and tree override rules
├── domain_rules.py      # Domain-specific rule evaluation and application
├── extraction.py        # Knowledge graph extraction and building
├── query_processing.py  # Query processing and response enhancement
└── workflows.py         # High-level workflow classes
```

## Usage

### Recommended: Import from Submodules

```python
# Workflow classes
from lightrag.rule.workflows import SchemaExtractionWorkflow, QueryAugmentationWorkflow

# Utility functions
from lightrag.rule.utils import _normalize_text_key, _stable_test_id

# Extraction functions
from lightrag.rule.extraction import _validate_controlled_payload, _build_controlled_nodes_edges

# Query processing
from lightrag.rule.query_processing import _should_bypass_query_cache, _extract_current_report_scopes

# Domain rules
from lightrag.rule.domain_rules import _evaluate_domain_rule_decisions, _apply_domain_rule_decisions_to_project_context

# Tree override
from lightrag.rule.tree_override import _load_tree_override_rules, _merge_annotation_rules
```

### Alternative: Package-level Imports

```python
# Only Workflow classes are exported at package level
from lightrag.rule import SchemaExtractionWorkflow, QueryAugmentationWorkflow
```

## Module Descriptions

### workflows.py
High-level workflow orchestration classes:
- `SchemaExtractionWorkflow`: Complete extraction pipeline for electrical standards
- `QueryAugmentationWorkflow`: Complete query augmentation pipeline

### utils.py
Common utility functions:
- Text normalization (`_normalize_text_key`)
- Value parsing and extraction (`_extract_*` functions)
- ID generation (`_stable_*_id` functions)
- Data merging (`_merge_*` functions)
- JSON parsing (`_parse_controlled_json_response`)

### tree_override.py
Annotation memory management:
- Loading tree override rules from JSON files
- Merging annotation rules from multiple sources
- Resolving annotation source paths

### domain_rules.py
Domain-specific rule evaluation:
- Loading domain rules from JSON files
- Evaluating rule decisions (split, merge, applicability, count)
- Applying rule decisions to project context

### extraction.py
Knowledge graph construction:
- Validating LLM extraction payloads
- Building controlled nodes and edges
- Upserting nodes and edges into storage

### query_processing.py
Query-time processing:
- Extracting report scopes from queries
- Filtering context by report scope
- Building final test item scope
- Post-processing LLM responses

## Migration from rule.py

The monolithic `rule.py` file has been removed. Update your imports:

```python
# OLD (no longer works)
from lightrag.rule import _normalize_text_key, _validate_controlled_payload

# NEW (import from specific submodule)
from lightrag.rule.utils import _normalize_text_key
from lightrag.rule.extraction import _validate_controlled_payload
```

## Code Statistics

| Module | Lines | Purpose |
|--------|-------|---------|
| utils.py | ~800 | Common utilities |
| tree_override.py | ~600 | Annotation memory |
| domain_rules.py | ~400 | Rule evaluation |
| extraction.py | ~500 | Graph extraction |
| query_processing.py | ~400 | Query processing |
| workflows.py | ~400 | Workflow classes |

**Total**: ~3100 lines (down from 6000+ in original rule.py)
