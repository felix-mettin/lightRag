"""Workflow classes for electrical_controlled mode.

This module provides high-level workflow classes that orchestrate
the extraction and query augmentation processes.
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import re
from typing import Any

from lightrag.utils import logger

from .utils import (
    _normalize_text_key,
    _parse_controlled_json_response,
    _json_dumps_compact,
    _stable_test_id,
    _stable_param_id,
    _stable_clause_id,
)
from .extraction import (
    _validate_controlled_payload,
    _build_controlled_nodes_edges,
    _upsert_controlled_node,
    _upsert_controlled_edge,
)


class SchemaExtractionWorkflow:
    """Workflow for controlled schema extraction in electrical standards domain.
    
    This workflow orchestrates the extraction of structured knowledge from
    electrical standards documents, including:
    
    1. Configuration Loading - Load schema config, annotation memory, domain rules
    2. Payload Validation - Validate and normalize LLM extraction results
    3. Tree Override Resolution - Apply human annotation rules from memory.json
    4. Node/Edge Building - Construct knowledge graph nodes and edges
    
    Example:
        workflow = SchemaExtractionWorkflow(global_config)
        nodes, edges = await workflow.process_extraction(
            raw_llm_response, chunk_text, chunk_meta, file_path
        )
    """
    
    def __init__(self, global_config: dict[str, Any]):
        """Initialize the workflow with global configuration."""
        self.global_config = global_config
        self.schema_cfg = self._load_schema_config()
        
    def _load_schema_config(self) -> dict[str, Any]:
        """Load electrical schema configuration from global config."""
        addon_params = self.global_config.get("addon_params", {}) or {}
        return addon_params.get("electrical_schema", {}) or {}
    
    def get_chunk_metadata(self, chunk_id: str, file_path: str) -> dict[str, str]:
        """Build standardized chunk metadata for extraction."""
        std_id = self.schema_cfg.get("standard_id", "")
        std_name = self.schema_cfg.get("standard_name", "")
        
        return {
            "std_id": std_id,
            "std_name": std_name,
            "clause_id": "",
            "clause_title": "",
            "chunk_id": chunk_id,
            "file_path": file_path,
        }
    
    def parse_clause_info(self, chunk_text: str, chunk_meta: dict) -> tuple[str, str]:
        """Extract clause ID and title from chunk text using configured pattern."""
        clause_pattern = self.schema_cfg.get(
            "clause_pattern", r"^(\d+(?:\.\d+)+)\s*(.*)$"
        )
        try:
            clause_regex = re.compile(clause_pattern)
        except re.error:
            clause_regex = re.compile(r"^(\d+(?:\.\d+)+)\s*(.*)$")
        
        first_line = chunk_text.split("\n", 1)[0].strip() if chunk_text else ""
        match = clause_regex.match(first_line)
        
        if match:
            return match.group(1), match.group(2).strip()
        return "", first_line[:120]
    
    def validate_payload(
        self, data: dict, chunk_text: str, chunk_meta: dict
    ) -> dict[str, Any]:
        """Validate and normalize extraction payload."""
        return _validate_controlled_payload(data, chunk_text, chunk_meta)
    
    def build_nodes_edges(
        self,
        payload: dict[str, Any],
        chunk_meta: dict[str, str],
        file_path: str,
        chunk_text: str = "",
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, str, dict]]]:
        """Build knowledge graph nodes and edges from validated payload."""
        return _build_controlled_nodes_edges(
            payload, chunk_meta, file_path, self.schema_cfg, chunk_text
        )
    
    async def process_extraction(
        self,
        raw_response: str,
        chunk_text: str,
        chunk_id: str,
        file_path: str,
    ) -> tuple[list[tuple[str, dict]], list[tuple[str, str, dict]]]:
        """Execute the complete extraction workflow."""
        # Step 1: Parse JSON response
        parsed = _parse_controlled_json_response(raw_response)
        
        # Step 2: Build chunk metadata
        chunk_meta = self.get_chunk_metadata(chunk_id, file_path)
        clause_id, clause_title = self.parse_clause_info(chunk_text, chunk_meta)
        chunk_meta["clause_id"] = clause_id
        chunk_meta["clause_title"] = clause_title
        
        # Step 3: Validate and normalize payload
        payload = self.validate_payload(parsed, chunk_text, chunk_meta)
        
        # Step 4: Build nodes and edges
        nodes, edges = self.build_nodes_edges(payload, chunk_meta, file_path, chunk_text)
        
        logger.debug(
            "SchemaExtractionWorkflow: chunk=%s nodes=%d edges=%d",
            chunk_id, len(nodes), len(edges)
        )
        
        return nodes, edges


class QueryAugmentationWorkflow:
    """Workflow for query-time augmentation in electrical standards domain."""
    
    def __init__(self, global_config: dict[str, Any]):
        """Initialize the workflow with global configuration."""
        self.global_config = global_config
        self.schema_cfg = self._load_schema_config()
        
    def _load_schema_config(self) -> dict[str, Any]:
        """Load electrical schema configuration from global config."""
        addon_params = self.global_config.get("addon_params", {}) or {}
        return addon_params.get("electrical_schema", {}) or {}
    
    def evaluate_domain_rules(self, query: str) -> dict[str, Any]:
        """Evaluate domain rules against the query."""
        from .domain_rules import _evaluate_domain_rule_decisions
        return _evaluate_domain_rule_decisions(query, self.schema_cfg)
    
    def extract_report_scopes(self, query: str) -> list[str]:
        """Extract report type scopes mentioned in the query."""
        from .query_processing import _extract_current_report_scopes
        return _extract_current_report_scopes(query, self.schema_cfg)
    
    def build_project_context(
        self,
        query: str,
        domain_rule_decisions: dict[str, Any],
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]:
        """Build project context with domain rules applied."""
        from .domain_rules import _apply_domain_rule_decisions_to_project_context
        
        param_map: dict[str, list[str]] = {}
        param_value_map: dict[str, dict[str, dict[str, str]]] = {}
        
        configured_test_items = self.schema_cfg.get("test_items", [])
        param_requirements = self.schema_cfg.get("test_item_param_requirements", {})
        
        for test_item in configured_test_items:
            if test_item and test_item not in param_map:
                required_params = param_requirements.get(test_item, [])
                param_map[test_item] = list(required_params) if required_params else []
                param_value_map[test_item] = {}
        
        return _apply_domain_rule_decisions_to_project_context(
            param_map,
            param_value_map,
            domain_rule_decisions,
            self.schema_cfg,
            query,
        )
    
    def filter_by_report_scope(
        self,
        project_param_map: dict[str, list[str]],
        project_param_value_map: dict[str, dict[str, dict[str, str]]],
        report_scopes: list[str],
    ) -> tuple[dict[str, list[str]], dict[str, dict[str, dict[str, str]]]]:
        """Filter project context by report scope whitelist."""
        from .query_processing import _filter_project_context_by_report_scope
        return _filter_project_context_by_report_scope(
            project_param_map, project_param_value_map, report_scopes
        )
    
    def build_final_test_item_scope(
        self,
        project_param_map: dict[str, list[str]],
        domain_rule_decisions: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Build final test item scope after rule application."""
        from .query_processing import _build_final_test_item_scope
        return _build_final_test_item_scope(project_param_map, domain_rule_decisions)
    
    def filter_context_by_scope(
        self,
        entities_context: list[dict],
        relations_context: list[dict],
        removed_test_items: list[str],
    ) -> tuple[list[dict], list[dict]]:
        """Filter retrieved context by final test item scope."""
        from .query_processing import _filter_context_by_final_test_item_scope
        return _filter_context_by_final_test_item_scope(
            entities_context, relations_context, removed_test_items
        )
    
    def build_resolved_rule_overrides(
        self,
        domain_rule_decisions: dict[str, Any],
    ) -> dict[str, Any]:
        """Build resolved rule overrides for prompt enhancement."""
        from .query_processing import _build_resolved_rule_overrides
        return _build_resolved_rule_overrides(domain_rule_decisions)
    
    def build_test_item_display_map(
        self,
        project_param_map: dict[str, list[str]],
    ) -> dict[str, str]:
        """Build display name mapping for test items."""
        from .query_processing import _build_test_item_display_map
        return _build_test_item_display_map(project_param_map)
    
    def postprocess_response(
        self,
        response_text: str,
        raw_data: dict[str, Any] | None,
    ) -> str:
        """Post-process LLM response with context filtering."""
        from .query_processing import _postprocess_electrical_markdown_response
        return _postprocess_electrical_markdown_response(response_text, raw_data)
    
    def should_bypass_cache(self) -> bool:
        """Check if query cache should be bypassed for electrical mode."""
        from .query_processing import _should_bypass_query_cache
        return _should_bypass_query_cache(self.global_config)
    
    async def augment_query(
        self,
        query: str,
        entities_context: list[dict],
        relations_context: list[dict],
        text_chunks: list[str],
    ) -> dict[str, Any]:
        """Execute the complete query augmentation workflow."""
        domain_rule_decisions = self.evaluate_domain_rules(query)
        report_scopes = self.extract_report_scopes(query)
        
        project_param_map, project_param_value_map = self.build_project_context(
            query, domain_rule_decisions
        )
        
        if report_scopes:
            project_param_map, project_param_value_map = self.filter_by_report_scope(
                project_param_map, project_param_value_map, report_scopes
            )
        
        allowed_items, removed_items = self.build_final_test_item_scope(
            project_param_map, domain_rule_decisions
        )
        
        filtered_entities, filtered_relations = self.filter_context_by_scope(
            entities_context, relations_context, removed_items
        )
        
        resolved_overrides = self.build_resolved_rule_overrides(domain_rule_decisions)
        display_map = self.build_test_item_display_map(project_param_map)
        
        return {
            "project_param_map": project_param_map,
            "project_param_value_map": project_param_value_map,
            "allowed_final_test_items": allowed_items,
            "removed_test_items": removed_items,
            "domain_rule_decisions": domain_rule_decisions,
            "resolved_rule_overrides": resolved_overrides,
            "test_item_display_map": display_map,
            "report_scopes": report_scopes,
            "filtered_entities": filtered_entities,
            "filtered_relations": filtered_relations,
            "original_entities": entities_context,
            "original_relations": relations_context,
            "text_chunks": text_chunks,
        }


__all__ = [
    "SchemaExtractionWorkflow",
    "QueryAugmentationWorkflow",
]
