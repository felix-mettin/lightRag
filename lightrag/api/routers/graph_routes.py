"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, Dict, Any
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from lightrag.utils import logger
from lightrag.standards import normalize_standard_type
from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["graph"])


async def _resolve_graph_standard_type(
    standard_type: str | None,
    rag_instances: dict,
    preferred_order: list[str] | None = None,
) -> str:
    if standard_type is None:
        return await _get_workspace_with_data(
            rag_instances,
            preferred_order=preferred_order,
        )

    normalized = normalize_standard_type(standard_type)
    if normalized and normalized in rag_instances:
        return normalized

    raise HTTPException(status_code=400, detail=f"Invalid standard_type: {standard_type}")


async def _get_workspace_with_data(
    rag_instances: dict, preferred_order: list[str] | None = None
) -> str:
    """未指定标准时默认返回 GB，不存在时再按顺序回退。"""
    if preferred_order is None:
        preferred_order = ["GB", "DLT", "IEC", "others"]
    if "GB" in rag_instances:
        return "GB"
    for ws in preferred_order:
        if ws not in rag_instances:
            continue
        rag = rag_instances[ws]
        try:
            labels = await rag.get_graph_labels()
            if labels:
                return ws
        except Exception:
            continue
    if "others" in rag_instances:
        return "others"
    return next(iter(rag_instances))
class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False
    allow_merge: bool = False


class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: Dict[str, Any]


class EntityMergeRequest(BaseModel):
    entities_to_change: list[str] = Field(
        ...,
        description="List of entity names to be merged and deleted. These are typically duplicate or misspelled entities.",
        min_length=1,
        examples=[["Elon Msk", "Ellon Musk"]],
    )
    entity_to_change_into: str = Field(
        ...,
        description="Target entity name that will receive all relationships from the source entities. This entity will be preserved.",
        min_length=1,
        examples=["Elon Musk"],
    )


class EntityCreateRequest(BaseModel):
    entity_name: str = Field(
        ...,
        description="Unique name for the new entity",
        min_length=1,
        examples=["Tesla"],
    )
    entity_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing entity properties. Common fields include 'description' and 'entity_type'.",
        examples=[
            {
                "description": "Electric vehicle manufacturer",
                "entity_type": "ORGANIZATION",
            }
        ],
    )


class RelationCreateRequest(BaseModel):
    source_entity: str = Field(
        ...,
        description="Name of the source entity. This entity must already exist in the knowledge graph.",
        min_length=1,
        examples=["Elon Musk"],
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity. This entity must already exist in the knowledge graph.",
        min_length=1,
        examples=["Tesla"],
    )
    relation_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing relationship properties. Common fields include 'description', 'keywords', and 'weight'.",
        examples=[
            {
                "description": "Elon Musk is the CEO of Tesla",
                "keywords": "CEO, founder",
                "weight": 1.0,
            }
        ],
    )


class EntityDeleteRequest(BaseModel):
    entity_name: str = Field(
        ...,
        description="Name of the entity to delete",
        min_length=1,
        examples=["Tesla"],
    )


class RelationDeleteRequest(BaseModel):
    source_entity: str = Field(
        ...,
        description="Name of the source entity in the relationship",
        min_length=1,
        examples=["Elon Musk"],
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity in the relationship",
        min_length=1,
        examples=["Tesla"],
    )


def create_graph_routes(rag_instances: dict, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graph/workspaces", dependencies=[Depends(combined_auth)])
    async def get_workspaces():
        """
        Get all available workspaces

        Returns:
            Dict: Dictionary with workspace information including data status
        """
        try:
            workspaces = {}
            for ws_name, rag in rag_instances.items():
                try:
                    labels = await rag.get_graph_labels()
                    has_data = len(labels) > 0
                    workspaces[ws_name] = {
                        "name": ws_name,
                        "has_data": has_data,
                        "label_count": len(labels)
                    }
                except Exception as e:
                    logger.warning(f"Error checking workspace {ws_name}: {str(e)}")
                    workspaces[ws_name] = {
                        "name": ws_name,
                        "has_data": False,
                        "label_count": 0
                    }
            return {"workspaces": workspaces}
        except Exception as e:
            logger.error(f"Error getting workspaces: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error getting workspaces: {str(e)}")

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels(standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")):
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            return await rag.get_graph_labels()
        except Exception as e:
            logger.error(f"Error getting graph labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error getting graph labels: {str(e)}")

    @router.get("/graph/label/popular", dependencies=[Depends(combined_auth)])
    async def get_popular_labels(
    limit: int = Query(300, ge=1, le=1000),
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            return await rag.chunk_entity_relation_graph.get_popular_labels(limit)
        except Exception as e:
            logger.error(f"Error getting popular labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error getting popular labels: {str(e)}")

    @router.get("/graph/label/search", dependencies=[Depends(combined_auth)])
    async def search_labels(
    q: str = Query(..., description="Search query string"),
    limit: int = Query(50, ge=1, le=100),
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            return await rag.chunk_entity_relation_graph.search_labels(q, limit)
        except Exception as e:
            logger.error(f"Error searching labels with query '{q}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error searching labels: {str(e)}")

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
    label: str = Query(..., description="Label to get knowledge graph for"),
    max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
    max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            return await rag.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{label}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error getting knowledge graph: {str(e)}")

    @router.get("/graph/entity/exists", dependencies=[Depends(combined_auth)])
    async def check_entity_exists(
    name: str = Query(..., description="Entity name to check"),
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            exists = await rag.chunk_entity_relation_graph.has_node(name)
            return {"exists": exists}
        except Exception as e:
            logger.error(f"Error checking entity existence for '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error checking entity existence: {str(e)}")

    @router.post("/graph/entity/edit", dependencies=[Depends(combined_auth)])
    async def update_entity(request: EntityUpdateRequest, standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")):
        """
        Update an entity's properties in the knowledge graph

        This endpoint allows updating entity properties, including renaming entities.
        When renaming to an existing entity name, the behavior depends on allow_merge:

        Args:
            request (EntityUpdateRequest): Request containing:
                - entity_name (str): Name of the entity to update
                - updated_data (Dict[str, Any]): Dictionary of properties to update
                - allow_rename (bool): Whether to allow entity renaming (default: False)
                - allow_merge (bool): Whether to merge into existing entity when renaming
                                     causes name conflict (default: False)

        Returns:
            Dict with the following structure:
            {
                "status": "success",
                "message": "Entity updated successfully" | "Entity merged successfully into 'target_name'",
                "data": {
                    "entity_name": str,        # Final entity name
                    "description": str,        # Entity description
                    "entity_type": str,        # Entity type
                    "source_id": str,         # Source chunk IDs
                    ...                       # Other entity properties
                },
                "operation_summary": {
                    "merged": bool,           # Whether entity was merged into another
                    "merge_status": str,      # "success" | "failed" | "not_attempted"
                    "merge_error": str | None, # Error message if merge failed
                    "operation_status": str,  # "success" | "partial_success" | "failure"
                    "target_entity": str | None, # Target entity name if renaming/merging
                    "final_entity": str,      # Final entity name after operation
                    "renamed": bool           # Whether entity was renamed
                }
            }

        operation_status values explained:
            - "success": All operations completed successfully
                * For simple updates: entity properties updated
                * For renames: entity renamed successfully
                * For merges: non-name updates applied AND merge completed

            - "partial_success": Update succeeded but merge failed
                * Non-name property updates were applied successfully
                * Merge operation failed (entity not merged)
                * Original entity still exists with updated properties
                * Use merge_error for failure details

            - "failure": Operation failed completely
                * If merge_status == "failed": Merge attempted but both update and merge failed
                * If merge_status == "not_attempted": Regular update failed
                * No changes were applied to the entity

        merge_status values explained:
            - "success": Entity successfully merged into target entity
            - "failed": Merge operation was attempted but failed
            - "not_attempted": No merge was attempted (normal update/rename)

        Behavior when renaming to an existing entity:
            - If allow_merge=False: Raises ValueError with 400 status (default behavior)
            - If allow_merge=True: Automatically merges the source entity into the existing target entity,
                                  preserving all relationships and applying non-name updates first

        Example Request (simple update):
            POST /graph/entity/edit
            {
                "entity_name": "Tesla",
                "updated_data": {"description": "Updated description"},
                "allow_rename": false,
                "allow_merge": false
            }

        Example Response (simple update success):
            {
                "status": "success",
                "message": "Entity updated successfully",
                "data": { ... },
                "operation_summary": {
                    "merged": false,
                    "merge_status": "not_attempted",
                    "merge_error": null,
                    "operation_status": "success",
                    "target_entity": null,
                    "final_entity": "Tesla",
                    "renamed": false
                }
            }

        Example Request (rename with auto-merge):
            POST /graph/entity/edit
            {
                "entity_name": "Elon Msk",
                "updated_data": {
                    "entity_name": "Elon Musk",
                    "description": "Corrected description"
                },
                "allow_rename": true,
                "allow_merge": true
            }

        Example Response (merge success):
            {
                "status": "success",
                "message": "Entity merged successfully into 'Elon Musk'",
                "data": { ... },
                "operation_summary": {
                    "merged": true,
                    "merge_status": "success",
                    "merge_error": null,
                    "operation_status": "success",
                    "target_entity": "Elon Musk",
                    "final_entity": "Elon Musk",
                    "renamed": true
                }
            }

        Example Response (partial success - update succeeded but merge failed):
            {
                "status": "success",
                "message": "Entity updated successfully",
                "data": { ... },  # Data reflects updated "Elon Msk" entity
                "operation_summary": {
                    "merged": false,
                    "merge_status": "failed",
                    "merge_error": "Target entity locked by another operation",
                    "operation_status": "partial_success",
                    "target_entity": "Elon Musk",
                    "final_entity": "Elon Msk",  # Original entity still exists
                    "renamed": true
                }
            }
        """
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            result = await rag.aedit_entity(
                entity_name=request.entity_name,
                updated_data=request.updated_data,
                allow_rename=request.allow_rename,
                allow_merge=request.allow_merge,
            )

            # Extract operation_summary from result, with fallback for backward compatibility
            operation_summary = result.get(
                "operation_summary",
                {
                    "merged": False,
                    "merge_status": "not_attempted",
                    "merge_error": None,
                    "operation_status": "success",
                    "target_entity": None,
                    "final_entity": request.updated_data.get(
                        "entity_name", request.entity_name
                    ),
                    "renamed": request.updated_data.get(
                        "entity_name", request.entity_name
                    )
                    != request.entity_name,
                },
            )

            # Separate entity data from operation_summary for clean response
            entity_data = dict(result)
            entity_data.pop("operation_summary", None)

            # Generate appropriate response message based on merge status
            response_message = (
                f"Entity merged successfully into '{operation_summary['final_entity']}'"
                if operation_summary.get("merged")
                else "Entity updated successfully"
            )
            return {
                "status": "success",
                "message": response_message,
                "data": entity_data,
                "operation_summary": operation_summary,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating entity '{request.entity_name}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error updating entity '{request.entity_name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating entity: {str(e)}"
            )

    @router.post("/graph/relation/edit", dependencies=[Depends(combined_auth)])
    async def update_relation(
    request: RelationUpdateRequest,
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            result = await rag.aedit_relation(
                source_entity=request.source_id,
                target_entity=request.target_id,
                updated_data=request.updated_data,
            )
            return {"status": "success", "message": "Relation updated successfully", "data": result}
        except Exception as e:
            logger.error(f"Error updating relation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error updating relation: {str(e)}")

    @router.post("/graph/entity/create", dependencies=[Depends(combined_auth)])
    async def create_entity(
    request: EntityCreateRequest,
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            result = await rag.acreate_entity(
                entity_name=request.entity_name,
                entity_data=request.entity_data,
            )
            return {"status": "success", "message": f"Entity '{request.entity_name}' created successfully", "data": result}
        except Exception as e:
            logger.error(f"Error creating entity: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error creating entity: {str(e)}")

    @router.post("/graph/relation/create", dependencies=[Depends(combined_auth)])
    async def create_relation(
    request: RelationCreateRequest,
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            result = await rag.acreate_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity,
                relation_data=request.relation_data,
            )
            return {"status": "success", "message": f"Relation created successfully between '{request.source_entity}' and '{request.target_entity}'", "data": result}
        except Exception as e:
            logger.error(f"Error creating relation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error creating relation: {str(e)}")

    @router.post("/graph/entity/delete", dependencies=[Depends(combined_auth)])
    async def delete_entity(
    request: EntityDeleteRequest,
    standard_type: str = Query(..., description="必填的标准类型/workspace")
):
        try:
            resolved_standard_type = await _resolve_graph_standard_type(
                standard_type, rag_instances
            )
            rag = rag_instances[resolved_standard_type]
            result = await rag.adelete_by_entity(request.entity_name)
            return {"status": "success", "message": f"Entity '{request.entity_name}' deleted successfully", "data": result}
        except Exception as e:
            logger.error(f"Error deleting entity: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error deleting entity: {str(e)}")

    @router.post("/graph/relation/delete", dependencies=[Depends(combined_auth)])
    async def delete_relation(
    request: RelationDeleteRequest,
    standard_type: str = Query(..., description="必填的标准类型/workspace")
):
        try:
            resolved_standard_type = await _resolve_graph_standard_type(
                standard_type, rag_instances
            )
            rag = rag_instances[resolved_standard_type]
            result = await rag.adelete_by_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity
            )
            return {"status": "success", "message": f"Relation deleted successfully between '{request.source_entity}' and '{request.target_entity}'", "data": result}
        except Exception as e:
            logger.error(f"Error deleting relation: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error deleting relation: {str(e)}")

    @router.post("/graph/entities/merge", dependencies=[Depends(combined_auth)])
    async def merge_entities(
    request: EntityMergeRequest,
    standard_type: Optional[str] = Query(None, description="可选的标准类型，不填则自动选择")
):
        try:
            std_type = await _resolve_graph_standard_type(standard_type, rag_instances)
            rag = rag_instances[std_type]
            result = await rag.amerge_entities(
                source_entities=request.entities_to_change,
                target_entity=request.entity_to_change_into,
            )
            return {"status": "success", "message": f"Successfully merged {len(request.entities_to_change)} entities into '{request.entity_to_change_into}'", "data": result}
        except Exception as e:
            logger.error(f"Error merging entities: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error merging entities: {str(e)}")

    return router
