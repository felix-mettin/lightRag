"""
LightRAG Rule Module

This module provides business rules, extraction logic, and workflow management
for the electrical_controlled mode in LightRAG.

Organization:
- utils.py: Common utility functions and helpers
- tree_override.py: Annotation memory and tree override rules
- domain_rules.py: Domain-specific rule evaluation and application
- extraction.py: Knowledge graph extraction and building
- query_processing.py: Query processing and response enhancement
- workflows.py: High-level workflow classes

Usage:
    # Workflow classes (recommended)
    from lightrag.rule.workflows import SchemaExtractionWorkflow, QueryAugmentationWorkflow
    
    # Utility functions
    from lightrag.rule.utils import _normalize_text_key, _stable_test_id
    
    # Or import all from package
    from lightrag.rule import SchemaExtractionWorkflow, QueryAugmentationWorkflow
"""

# Re-export main classes for convenience
from .workflows import SchemaExtractionWorkflow, QueryAugmentationWorkflow

__all__ = [
    "SchemaExtractionWorkflow",
    "QueryAugmentationWorkflow",
]
