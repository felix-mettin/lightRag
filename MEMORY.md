# Conversation Memory (2026-02-06, Updated)

## Context
- User is working in `/Users/df/workspace/python/LightRAG`.
- Goal: build controlled electrical KG with strict hierarchy:
  `报告 -> 试验 -> 试验项目 -> 特征值(参数) -> 标准文件/条款`.
- Main mode: `kg_schema_mode=electrical_controlled`.

## User Reported Current Issues
1. `chunk_size=1000`:
   - Graph structure mostly correct.
   - Many parameter values are missing.
   - QA recall quality is poor with漏召回.
2. `chunk_size=3000`:
   - Extraction can fail with 504 gateway timeout.
   - Logs showed repeated retries and many skipped chunks.
3. Controlled extraction errors:
   - Earlier: `Controlled extraction task failed ... '\n"standard""'`.
   - Later screenshot: repeated `schema validation failed ... root keys mismatch`, then chunks skipped.

## Code Changes Applied

### 1) Mix recall robustness (`lightrag/operate.py`)
- Added `_resolve_chunk_id(...)` with fallback hash ID when vector result lacks `id/chunk_id`.
- Updated mix flow to use resolved chunk IDs in vector tracking/merge.
- Fixed mix early-return condition to check `vector_chunks` instead of `chunk_tracking`.
- Added chunk token budget fallback in `_build_context_str`:
  - If mix budget for chunks is exhausted, drop entities/relations context first to preserve chunk recall.
  - If still non-positive, apply minimum chunk fallback budget.

### 2) Extraction resilience for large chunks (`lightrag/operate.py`)
- In controlled extraction:
  - LLM call exceptions are caught per chunk attempt (no immediate task crash).
  - Added transient error detection for 504/timeout.
  - On transient failure, fallback split current chunk content into smaller parts and retry extraction.
  - At task gather stage, failed tasks are converted to empty result instead of aborting whole batch.

### 3) OpenAI retry extension (`lightrag/llm/openai.py`)
- Added `InternalServerError` to tenacity retry condition for chat completions.

### 4) Controlled JSON parse tolerance (`lightrag/operate.py`)
- Added tolerant parser:
  - remove think tags,
  - strip fenced code blocks,
  - extract primary `{...}`,
  - strict `json.loads` then fallback to `json_repair.loads`.

### 5) Prompt formatting safety (`lightrag/operate.py`)
- Root cause for `'"standard"'` error was prompt template formatting with JSON braces.
- Added safe formatter that:
  - protects known placeholders,
  - escapes remaining braces as literals,
  - restores placeholders.
- This prevents `.format` from treating JSON schema braces as placeholders.

### 6) Controlled payload validation softening (`lightrag/operate.py`)
- Replaced strict root-key equality failure with normalization:
  - missing top-level keys are defaulted,
  - `standard/clause` missing fields are filled,
  - `clause.chunk_id` mismatch is corrected to current chunk.
- Goal: reduce `root keys mismatch` causing chunk skips.

## Config and Prompt Adjustments Made (`config.ini`)
- `ELECTRICAL_SCHEMA` and `PROMPTS` were tuned:
  - Added/expanded aliases for insulation/control-circuit test naming variants.
  - Unified key mappings in `param_map` (e.g., `交流电压` with rated PF withstand key; `正常次数` to `test_count`; added rated/current synonyms).
  - Adjusted requirement entries for specific test items (e.g., BC1/异相接地).
  - Prompt constraints strengthened to force param-key normalization and capture defaults/applicability rules.

## Current Status
- `'"standard"'` template-format error path has been addressed in code.
- Screenshot indicates active issue shifted to model output schema mismatch (`root keys mismatch`), and code has now been updated to normalize instead of fail hard.
- Need user to rerun with updated code actually loaded in runtime container/service.

## Latest Runtime Findings (2026-02-06, later)
- User increased performance params in `.env` (`CHUNK_SIZE=3000`, higher LLM/embedding concurrency).
- New runtime failure appeared: `ollama_embed ... 404 Not Found`.
  - Root cause: effective embedding config was missing/unset, system fell back to default Ollama embedding.
  - Fix applied in `.env` with explicit openai-compatible embedding endpoint:
    - `EMBEDDING_BINDING=openai`
    - `EMBEDDING_BINDING_HOST=http://host.docker.internal:18088/xinference/v1`
    - `EMBEDDING_MODEL=bge-m3`
    - `EMBEDDING_DIM=1024`
    - `EMBEDDING_BINDING_API_KEY=sk-11111`
- After restart, extraction runs and produces nodes/edges; repeated warning observed:
  - `Controlled payload root mismatch, normalizing keys. found=[...]`
  - This indicates model output has extra fields at root (spill), but extraction is still successful (`extracted XX Nodes + XX Edges`).
- Log-noise improvement applied:
  - benign root spill cases are downgraded from `WARNING` to `DEBUG`;
  - only true mismatches (missing required root keys, unexpected extras) remain `WARNING`.

## User Clarification Provided
- "容错" in this context means: normalize malformed-but-usable payload and continue ingestion, not skip chunk.
- Skip only happens on unrecoverable failures; successful extraction lines confirm chunk data is retained in graph.

## Validation Notes
- Local syntax checks passed (`python3 -m py_compile`) for modified files.
- Full integration tests were not run in this session.

## Next Practical Checks for User
1. Restart/redeploy runtime so latest `operate.py`/`openai.py` are loaded.
2. Re-run one document and inspect:
   - skipped chunk count,
   - whether `root keys mismatch` still causes skips,
   - whether mix QA now includes chunk context consistently.

---

## Conversation Memory (2026-02-24, Updated)

## Context
- User focused on local page `electrical_test_tree.html` for tree interaction UX.
- Goal shifted from browser-local persistence to portable JSON import/export.
- Secondary goal: refine LightRAG native QA user prompt for electrical test planning and counting rules.

## Frontend Changes Requested and Delivered
1. Node notes and persistence
   - User asked if each tree node can hold notes and persist across reopen.
   - Initial feasibility confirmed as easy in single-page frontend.
2. Persistence strategy changed
   - User preferred JSON import/export over `localStorage`.
   - Implemented full-tree JSON model: each node can carry `note`; export/import handles full structure + notes.
3. UX fixes after user validation
   - User reported note updates caused collapse of expanded tree.
   - Implemented expanded-state preservation via `details[data-node-path]` open-path restore.
   - User requested note edit/delete in addition to add.
   - Implemented explicit `备注` (add), `编辑`, `删除` actions per node.
4. Outcome
   - User confirmed feature works and is satisfactory.

## Prompt Engineering Discussion (LightRAG native QA)
1. User issue
   - Model output mostly correct, but "工频耐受电压试验" count was wrong (used `×2×15` and got 270).
2. Root cause identified
   - Existing prompt had hard constraints applying lightning-style formula to both power-frequency and lightning tests.
3. Fix direction provided
   - Split counting formulas by project:
     - 工频: direct row counts (合闸+分闸), no `×2×15`.
     - 雷电: keep `×2×15`.
4. Additional user constraints incorporated
   - "控制和辅助回路的绝缘试验" is mandatory only when insulation-domain task is in scope.
   - For that project, Party-A wording enforced: test on motor control circuit, test count = 1.
   - "辅助和控制回路温升试验" is mandatory when temperature-rise domain is in scope.
5. Delivered content
   - Provided full revised user prompt text with:
     - domain-conditional mandatory projects,
     - corrected counting rules,
     - unchanged strict evidence/coverage/gating checks,
     - fixed output format sections A-F.

---

## Conversation Memory (2026-02-26, Updated)

## Context
- User exported annotated tree JSON: `electrical_test_tree_20260226_161638.json`.
- User requirement: keep LightRAG main pipeline unchanged, but improve controlled KG output quality using manual annotations:
  - remove unwanted test items/parameters,
  - keep required parameters,
  - carry parameter values as standard/user-input/formula/default per notes.
- User confirmed current priority is KG generation correctness first; QA prompt-level custom merge logic remains in existing user prompt workflow.

## User Clarification on Scope
1. Current device-parameter input list is available (dynamic values at runtime), including:
   - rated voltage/current/frequency,
   - PF/LI withstand values (incl. fracture),
   - short-circuit currents/duration,
   - line/cable charging switching current,
   - E2/C2 capability info.
2. Immediate target is not full rule-engine redesign; first milestone is improving extracted graph structure/value format.
3. Future custom logic (merge tests, special station defaults, etc.) can continue via user prompt for now.

## Implemented Code Changes (This Session)

### 1) Annotation-driven override loader (`lightrag/operate.py`)
- Added tree-override parsing helpers:
  - `_load_tree_override_rules(...)`
  - `_resolve_tree_override_path(...)`
  - `_infer_value_source(...)`
  - `_note_is_remove(...)`
  - `_normalize_text_key(...)`
- Added in-memory file mtime cache `_TREE_OVERRIDE_CACHE` to avoid repeated parsing.

### 2) Controlled graph build hook (minimal-intrusion)
- In `_build_controlled_nodes_edges(...)`, applied override rules before node/edge creation:
  - skip test items marked as "这个不要/不需要",
  - replace extracted parameter list with reviewed template from override rules (when available),
  - attach `value_source` (`standard` / `user_input` / `formula` / `default`) to parameter nodes,
  - retain existing pipeline behavior when no override exists.
- No change to chunking, extraction scheduling, storage architecture, or query flow.

### 3) Fixed rule-file loading strategy (solidification)
- Previous implementation could read temporary exported tree JSON by path.
- Updated to support stable rules file format with root key `tests`.
- Default auto-load path added:
  - `lightrag/config/electrical_tree_override_rules.json` (if file exists).
- Optional explicit config path keys supported:
  - `tree_override_rules_path`
  - `tree_override_json_path` / `tree_override_path` (fallback compatibility).

### 4) Generated stable rule file from temporary export
- Created:
  - `lightrag/config/electrical_tree_override_rules.json`
- Content generated from `electrical_test_tree_20260226_161638.json` (35 test templates).
- This allows deleting/modifying the temporary export without breaking runtime behavior.

### 5) Container mount updated
- Updated `docker-compose.yml` to mount stable rules file into container:
  - `./lightrag/config/electrical_tree_override_rules.json:/app/lightrag/config/electrical_tree_override_rules.json`

## Validation
- Local syntax check passed:
  - `python3 -m py_compile lightrag/operate.py`
- No full integration test executed in this session.

## Rollback / Safety Notes
- Main pipeline remains unchanged; override is additive.
- Fast rollback options:
  1. remove/disable rules-file path config; or
  2. remove the mounted rules file entry; or
  3. revert `lightrag/operate.py` changes.
- User explicitly requested rollback capability; implementation preserved that requirement.

## Next Practical Steps for User (Tomorrow)
1. Recreate/restart container so new mount and code are active.
2. Re-upload one representative document and regenerate KG.
3. Check:
   - removed unwanted test items/parameters are actually absent,
   - parameter nodes now carry `value_source`,
   - key test-item parameter formatting is improved.
4. If output still has mismatches, continue iterative refinement by editing:
   - `lightrag/config/electrical_tree_override_rules.json`
   (instead of temporary tree export file).

---

## Conversation Memory (2026-02-27, Updated)

## Context
- User reviewed generated override file and multiple regenerated tree exports:
  - `electrical_test_tree_20260227_134351.json`
  - `electrical_test_tree_20260227_170738.json`
  - `electrical_test_tree_20260227_183640.json`
- User reported major remaining issues:
  - test items still mixed across categories (especially `开合性能型式试验` polluted by short-circuit items),
  - many parameters that should be removed still present,
  - corrected value notes partly applied, but noisy error text still leaks into output.

## Root Causes Identified
1. Rule-key collision from legacy name-only indexing:
   - same test name in different categories overwritten or ambiguously matched.
2. Runtime graph IDs were too coarse:
   - `test_id = test:{test_name}` and `param_id = param:{test_name}:{param_key}` caused cross-category merge pollution.
3. Deletion expectations vs upsert behavior:
   - pipeline upserts/merges new data but does not prune old nodes unless storage is rebuilt.
4. Value text sanitation was too weak:
   - marker text like `特征值不对` and mixed formula/table segments leaked into final `value_text`.

## Implemented Changes (This Session)

### 1) Override model redesign and compatibility (`lightrag/operate.py`)
- Added path-aware rule model:
  - `tests_by_path`, `tests_by_name`, `add_test_items`.
- Kept backward compatibility for old `tests` format.
- Added path parsing tolerance for both list and string paths (`a > b > c`, `a/b/c`).
- Added `remove_parameters` support in rule loading and generation.

### 2) Override application strategy changed to merge+remove
- Replaced destructive full parameter replacement with incremental merge by parameter identity.
- Added selective delete by `remove_parameters`.
- Supports explicit replace mode only when `parameters_mode = replace`; default is merge.

### 3) Note-driven correction heuristics expanded
- Added value source/value text correction helpers:
  - user-input extraction (`用户录入...`),
  - explicit correction extraction (`应当为/应为/改为/修改为...`),
  - parameter-name correction from notes (`特征值名称错误，应当为...`).
- Ensures correction markers themselves are not used as final values.

### 4) Stricter test-item gating for schema correctness
- Added out-of-schema drop:
  - if test item not in configured `ELECTRICAL_SCHEMA.test_items`, drop it.
- Added category mismatch drop:
  - if extracted category conflicts with matched override category, drop it.
- When override exists, unmatched items are dropped to reduce free-form drift.

### 5) Scope-aware stable IDs to stop cross-category pollution
- Updated IDs to include scope key (`report_type + category + test_name`):
  - `_stable_test_id(name, scope_key)`
  - `_stable_param_id(test_item, param_key, scope_key)`
  - `_stable_rule_id(test_item, rule_key, scope_key)`
- Prevents same-name test items in different categories from merging into one node.

### 6) Value text sanitation hardening
- Added `_sanitize_value_text(...)`:
  - remove marker fragments (`特征值不对`, `提取错误`, etc.),
  - prefer `默认为/默认/用户录入` segment when present,
  - for formula/table mixed strings, prefer table-reference segment,
  - normalize repeated units (`Hz Hz` -> `Hz`).

### 7) Rule file regenerated
- Regenerated `lightrag/config/electrical_tree_override_rules.json` with:
  - `tests_by_path` (65),
  - `tests_by_name` (48),
  - `parameters_mode: "merge"` defaults,
  - `remove_parameters` populated from tree notes.

## Operational Findings
- User hit `docker compose up --build` network failure on Docker Hub BuildKit frontend (`docker/dockerfile:1` token fetch reset).
- Practical workaround: run with existing image/bind-mount code path (`docker compose up -d` / `--no-build`) after clean restart.

## Validation
- Repeated local syntax validation passed:
  - `python3 -m py_compile lightrag/operate.py`
- Integration/e2e tests were not executed in this session.

## Runbook Clarification for User
1. Stop container: `docker compose down`
2. Rebuild storage: `rm -rf data/rag_storage && mkdir -p data/rag_storage`
3. Start service: `docker compose up -d` (or `--no-build` when build network is unstable)
4. Verify startup logs include override load message before re-uploading docs.

---

## Conversation Memory (2026-02-28)

## User-Reported Symptoms
- Tree structure and naming drifted (e.g. `test:... > ... > ...`) and some child nodes missing.
- Short-circuit family (`短路开断试验`) items/params missing or polluted.
- Parameter extraction had strong noise (wrong values, reference-only text, redundant params).
- Merge stage appeared to be "stuck" for long periods on large standard files.

## Decisions / Clarifications
- User wants strict extraction with whitelist behavior: keep required items, but do not over-filter correct ones.
- User wants `/`-style values to prefer the corrected/right side, and avoid keeping known-wrong side in final tree values.
- For chunking, current code path does **not** consume heading-based env keys; effective controls are only chunk size/overlap.

## Config Changes Applied
- `.env` updated to stable chunk profile:
  - `CHUNK_SIZE=2200`
  - `CHUNK_OVERLAP_SIZE=200`
- Removed misleading non-effective env keys:
  - `CHUNK_BY_HEADING`
  - `CHUNK_HEADING_PATTERN`
- `config.ini` cleanup:
  - removed `client_requirements` (loaded but not used in downstream extraction/merge path).

## Merge Performance Root Cause (Controlled Mode)
- In `electrical_controlled` merge path, data was previously upserted in per-chunk serial loops:
  - repeated `get + upsert` for duplicated nodes/edges across chunks,
  - causing very long merge time at high edge counts.

## Code Fix Applied (2026-02-28)
- File: `lightrag/operate.py`
- In `merge_nodes_and_edges(...)` for `kg_schema_mode == "electrical_controlled"`:
  1. aggregate & deduplicate nodes/edges before writing,
  2. merge duplicates in-memory using existing merge helpers,
  3. async upsert with semaphore (`llm_model_max_async`),
  4. keep cancellation checks,
  5. add clear aggregate progress log:
     - `Controlled merge aggregated: nodes X->Y, edges A->B, async=N`
  6. completion log now reports final deduped counts.

## Immediate Outcome
- User log after fix showed healthy aggregation behavior:
  - `nodes 7534->695`
  - `edges 14858->7894`
- This indicates de-dup works; remaining long time is mainly due to still-large edge write volume, not deadlock.

## Follow-up (Next Session)
- Consider adding merge heartbeat logs (periodic `processed/total`) to avoid "looks frozen" ambiguity.
- If merge still too long, reduce low-value edge cardinality (relation filtering/whitelist by relation type).

---

## Conversation Memory (2026-03-02, Updated)

## Context
- User continued debugging electrical controlled KG + QA for type-test planning and counting.
- Primary focus shifted to two fronts:
  1) KG side: improve structured evidence for counting (table rows, conditions) and retrievable descriptions.
  2) QA prompt side: enforce robust counting behavior, especially fracture/断口 and range-based parameter calculations.

## User-Reported Issues
1. Generated feature values still noisy/over-complete; many `A / B` mixed values and placeholder-style outputs appeared.
2. QA counting regressed between runs:
   - once gave expected style counts,
   - then degraded to incomplete row extraction (`[1,2,3]` + `[4,5,6]`), missing expected rows,
   - fracture triggered by user input but often not independently counted.
3. In capacitive switching QA, model incorrectly output raw base current (e.g., 25A) instead of applying rule `10%-40%`.

## Code Changes Applied (This Session)

### 1) `lightrag/operate.py` — stronger controlled-data quality and retrievability
- Added HTML-table-aware pre-processing in `chunking_by_token_size(...)`:
  - normalize table rows/cells into line-structured text before token chunking,
  - improve downstream retrieval of tabular conditions/row evidence.
- Extended controlled build path:
  - `_build_controlled_nodes_edges(...)` now accepts `chunk_text`.
  - Added `_extract_counting_evidence_rules(...)`:
    - attempts to infer table counting evidence (`表10/表13`) from chunk text,
    - emits structured `rule_type=table` rules with expressions like `close_rows/open_rows/fracture_rows`, target `test_count`.
- Improved graph content richness:
  - added automatic node description composition for `TestParameter/TestRule/TestItem/StandardClause`,
  - added automatic edge description composition,
  - ensures VDB payloads no longer rely on empty/UNKNOWN-like descriptions.

### 2) Prompt template reinforcement
- `lightrag/prompt.py` electrical extraction prompt updated to explicitly encourage table-count rule extraction (`rule_type=table`, row counts, `target_param_key=test_count`).
- `config.ini` prompt section similarly strengthened for table-driven count evidence extraction.

### 3) Validation
- Syntax check passed:
  - `python3 -m py_compile lightrag/operate.py lightrag/prompt.py lightrag/lightrag.py`

## QA Prompt Engineering Iterations
- Multiple full prompt revisions were generated to stabilize QA behavior.
- Key additions across iterations:
  1. strict evidence-validity rules (reject directory/index-only references as counting evidence),
  2. fracture-trigger and count-inclusion constraints,
  3. range-expansion checks for row ids,
  4. formula/value derivation hierarchy from user inputs,
  5. capacitive test current rule: enforce `10%-40%` range computation from base current.
- User requested final direct-paste prompt versions several times; final variant prioritized stability and deterministic counting over full generality.

## Current Findings / Risks
1. Retrieval quality remains a bottleneck for table counting even after chunk/table normalization; LLM may still latch onto weak references (e.g., table mentions without full row content).
2. Deterministic counting for insulation (e.g., expected stable totals) can be enforced by prompt policy, but this is prompt-side stabilization, not a full evidence-structural guarantee.
3. For capacitive projects, explicit prompt rules are required to avoid outputting base current directly instead of computed `10%-40%` interval.

## Practical Next Steps
1. Rebuild storage and re-ingest to activate new graph descriptions + inferred table rules.
2. Re-run QA with final prompt and inspect whether `trial current` values are interval outputs (not raw base values).
3. If row evidence is still unstable, add a dedicated structured extractor for table D.1/D.2 and table10/13 into KG (separate from free-text rule inference).
4. Keep fracture-count handling policy explicit in prompt until table-structured ingestion is fully reliable.

---

## Conversation Memory (2026-03-04, Updated)

## Context
- User focused on electrical controlled KG output alignment with business-expert annotations from multiple latest tree exports:
  - `electrical_test_tree_20260304_130331.json`
  - `electrical_test_tree_20260304_141835.json`
  - `electrical_test_tree_20260304_153418.json`
  - `electrical_test_tree_20260304_175313.json`
  - `electrical_test_tree_20260304_190746.json`
- Primary goal: make generated graph match expert-corrected tree semantics, especially:
  1) remove colloquial note fragments in values,
  2) expand shorthand references (e.g., `同工频` / `同雷电` / `同CC2`),
  3) fill missing test items/parameters from review notes,
  4) keep extraction constrained by project parameter templates.

## Main Issues Reported by User (Today)
1. Output still contained colloquial fragments like:
   - `1min / 一般我们用1min`
   - `正常或充气充油 / 我们一般用...`
2. Output still contained shorthand values:
   - `试验部位: 同工频`
   - `试验部位: 同雷电且＞252产品`
   - `操作顺序: 同CC2`, `试验相数: 同CC2`
3. Missing required items/parameters remained across runs:
   - `局部放电试验` absent under `绝缘性能型式试验`
   - `控制和辅助回路的绝缘试验` missing `试验电压=2kV`
   - `连续电流试验` missing `频率`
4. With long strict QA prompt, model occasionally degenerated into overlong/invalid evidence listing and non-whitelist parameter output.

## Code Changes Applied (This Session)

### 1) `lightrag/operate.py` — note/override parsing strengthened
- Extended remove-note markers in `_note_is_remove(...)` to include additional business deletion wording (`不涉及/无关/删除/错` etc.).
- Added `remove_rules` parsing + application:
  - parse rule deletions from `条件/规则` child notes,
  - drop matching rules before `TestRule` node creation.
- Enhanced value correction extraction:
  - support `改成/应改成` and reviewer phrasing patterns.
- Extended value-source detection:
  - recognize `客户录入/客户输入/用户输入` as `user_input`.
- Added colloquial cleanup in `_sanitize_value_text(...)`:
  - strips phrases like `我们一般用/一般我们用/...`.
- Added shorthand expansion helper `_expand_shorthand_param_value(...)`:
  - resolve `同工频/同雷电/同CC2/同T10` to concrete parameter values from matched override test rules.
- Improved fallback matching for shorthand expansion:
  - if strict same-report/category match misses, fallback to same-token candidates.
- Added parser for feature-level missing-parameter annotations:
  - `_extract_missing_feature_params(...)` parses notes like `缺特征值：参数，值`.
- Added parser for category-level missing-test annotations:
  - `_extract_missing_test_items(...)` + `_extract_test_item_detail_params(...)` parse notes like `缺少XX试验，特征值为...` and generate `add_test_items`.
- Added separator tolerance in `_split_name_and_value(...)` for more colon variants (`：﹕∶:`).
- Add-item injection guard relaxed:
  - report-scope additions (e.g. `型式试验`) now can pass when payload uses category-level report scopes.

### 2) `config.ini` updates
- Ensured `局部放电试验` exists in `test_items`.
- Added frequency key mappings to `param_map`:
  - `频率:test_frequency`
  - `额定频率:test_frequency`
  - `试验频率:test_frequency`
- Updated `test_item_param_requirements`:
  - `工频耐受电压试验` includes `介质性质`
  - `控制和辅助回路的绝缘试验` includes `试验电压`
  - `连续电流试验` includes `频率`
  - `局部放电试验:试验部位|试验电压|放电次数`

### 3) `lightrag/config/electrical_tree_override_rules.json` regeneration and patching
- Regenerated rules from latest annotated trees (progressive updates), preserving existing curated `add_test_items`.
- Confirmed `add_test_items` includes `局部放电试验` and later auto-generated `电寿命试验` from category-level note.
- Applied direct cleanups for key capacitive projects:
  - fill missing feature parameters from feature-note declarations,
  - map `LC2` shorthand (`同CC2`) to explicit values (`CO`, concrete phase expression).
- Current rule snapshot characteristics:
  - path/name entries reduced to reviewed scope (focused templates from latest annotation file),
  - `add_test_items` contains manual + generated additions.

## Validation Performed
- Repeated syntax checks passed:
  - `python3 -m py_compile lightrag/operate.py`
- Spot checks via `jq` confirmed:
  - `局部放电试验` exists in rules `add_test_items`
  - `控制和辅助回路的绝缘试验` has `试验电压 = 2 kV`
  - `连续电流试验` has `频率` template
  - `LC2` no longer stores `同CC2` for key fields in current rules file.

## Remaining Risk / Why 190746 Still Off
- `190746` was generated before latest fixes were verified end-to-end in runtime.
- Runtime/service may still be loading stale code or stale cached override payload.
- There may still be extraction-side drift for some projects if override match misses on report/category scope; fallback logic was improved but needs one full rerun confirmation.

## Next Steps For Tomorrow Rerun
1. Restart/redeploy runtime to ensure latest `operate.py` and updated rules are loaded.
2. Rebuild graph storage (clean ingest) and rerun the same source docs.
3. Export new tree and verify first:
   - `绝缘性能型式试验` contains `局部放电试验`
   - `LC2` no longer shows `同CC2` values
   - BC/CC/LC missing feature templates are concretely present
4. If still mismatched, continue by diffing latest export vs `175313` annotation list and patching only missed templates.

## 2026-03-04（会话补充：问答参数集合与图谱全量不一致）

### 1) 现象与根因确认
- 用户反馈：通过 API 查询的 HTML 树图可看到试验项目的全量特征值，但问答结果中同项目参数明显变少（例如 T10/T30/T60 仅剩少量参数）。
- 结论：不是缓存/离线 JSON 导入问题；是同一服务内两条链路语义不同导致。
  - 图谱端点（/graphs）：走图遍历（`get_knowledge_graph`），偏“结构全量展示”。
  - 问答端点（/query, /query/data）：走检索+截断+上下文拼装（`aquery_llm`/`aquery_data` -> `_build_query_context`），偏“相关子集”。
- 因此问答如果不显式注入参数全集，容易由检索子集隐式决定参数集合，出现漏参。

### 2) 本次代码改造（已完成）
#### A. 修复 merge 模式误删参数
- 文件：`lightrag/operate.py`
- 变更：`override_param_filter_to_template` 不再在 `parameters_mode=merge` 下默认裁剪抽取参数。
- 新规则：仅当 override 规则显式 `template_only=true` 时才按模板裁剪。
- 目的：避免 T10/T30/T60 等项目在 merge 时被误删为少量参数。

#### B. 在问答上下文注入 PROJECT_PARAM_MAP（参数全集骨架）
- 文件：`lightrag/operate.py`（`_build_context_str`）
- 新增：`_build_project_param_map()`
  - 当 `kg_schema_mode == electrical_controlled` 时，基于 `addon_params.electrical_schema.test_item_param_requirements` 构建 `PROJECT_PARAM_MAP`。
  - 优先按当前检索上下文命中的试验项目收敛；若未命中则回退配置全量，防止问答阶段漏掉项目参数集合。
- 注入：`project_param_map_str` 写入 `kg_query_context` 模板。
- 回传：`final_data.metadata.project_param_map = project_param_map`，便于 `/query/data` 直接核对问答实际拿到的参数白名单。

#### C. 扩展问答上下文模板
- 文件：`lightrag/prompt.py`
- 变更：`PROMPTS["kg_query_context"]` 增加
  - `Graph Parameter Whitelist (PROJECT_PARAM_MAP)` JSON 区块。
- 目的：让 LLM 在问答阶段有“参数全集约束”可用，而不是只看检索片段。

### 3) 已执行校验
- `python3 -m py_compile lightrag/operate.py lightrag/prompt.py` 通过。

### 4) 运行建议（给下次联调）
1. 重启服务使查询链路改动生效。
2. 可先不重建图谱（本次核心是问答上下文拼装改造）。
3. 用 `/query/data` 检查 `metadata.project_param_map` 是否为期望全量参数集合。
4. 再跑目标问题核对：T10/T30/T60 等项目不应再出现“2/2 伪完整”。

### 5) 待继续（如用户需要）
- 把“三类取值”做成代码级硬执行：
  - Type-A 标准抽取值
  - Type-B 业务规则判定值
  - Type-C 用户输入代入计算值
- 并在输出前做强校验：`OutputParams == ExpectedParams`（多出/缺失即报错）。

---

## Conversation Memory (2026-03-05, Updated)

## Context
- User requested review of `MEMORY.md` and latest implementation changes in workspace.
- Main goals: verify current implementation consistency, fix obvious integration issues, and continue improving electrical KG accuracy via annotation memory.

## Issues Found and Fixed

### 1) Container mount path drift
- Found mismatch:
  - code/config uses `lightrag/config/annotation_memory.json`
  - docker-compose still mounted old `electrical_tree_override_rules.json`
- Fixed `docker-compose.yml` mount to:
  - `./lightrag/config/annotation_memory.json:/app/lightrag/config/annotation_memory.json`

### 2) Electrical schema config not fully propagated
- Found `config.ini` options existed but were not loaded into `addon_params.electrical_schema`.
- Updated `lightrag/lightrag.py` parsing to include:
  - `annotation_memory_path`
  - `annotation_source_json_paths`
  - `annotation_auto_merge_to_memory`
  - `strict_tree_override_match`
  - `override_param_filter_to_template`
  - `annotation_guardrail_mode`
  - `annotation_guardrail_only_override`
  - retained explicit `enforce_param_whitelist`
- Impact: `operate.py` now receives intended runtime switches instead of falling back to defaults.

### 3) Query whitelist fallback hardening
- Found `_build_project_param_map()` could return empty map when retrieval missed test-item entities.
- Updated `lightrag/operate.py`:
  - fallback to `test_item_param_requirements` when no test candidates;
  - fallback per project when graph edges/nodes are missing;
  - final fallback if map remains empty.
- Impact: reduces QA-side missing parameter coverage caused by retrieval sparsity.

## Validation in this session
- Syntax checks passed:
  - `python3 -m py_compile lightrag/lightrag.py lightrag/operate.py`
  - `python3 -m py_compile lightrag/api/routers/document_routes.py lightrag/lightrag.py lightrag/llm/openai.py lightrag/operate.py lightrag/prompt.py`
- `ruff` unavailable in current shell.
- Full pytest/integration tests not run.

## Annotation Memory Updates (from user issue list)
- User provided structured mismatch list (without pasting full regenerated JSON).
- Patched `lightrag/config/annotation_memory.json` for targeted items:
  - 雷电冲击耐受电压试验：补 `电压极性=正极性和负极性`
  - 工频耐受电压试验：修 `试验时间=1 min`，修 `试验次数`条件文本
  - 控制和辅助回路的绝缘试验：修 `试验电压=2 kV`，移除错误参数名 `2kV`
  - 局部放电试验：补齐关键全量参数（额定电压、预加电压/时间、交流电压、测量时间、局部放电值、介质性质、试验相数、试验次数）
  - BC1/BC2/CC1/CC2/LC1/LC2：补齐/修正试验类别、能力级别、试验电流、试验电压、操作顺序、试验次数
- Rebuilt `tests_by_name` index from path rules to keep name/path consistency after edits.

## Prompt/Communication Outputs delivered
- Generated Mermaid end-to-end workflow (ingest -> chunk -> extract -> constrain -> remove -> dedup merge -> QA); then simplified Chinese-only version.
- Drafted leadership-facing concise status wording:
  - 方案三（MD递归）当前主要问题：树1-4层级难稳定固化。
  - LightRAG二开方案：结构化更强，但参数正确率/规则完整性仍需持续优化，当前更适合“初稿+复核”。
- Produced two prompt versions for QA generation:
  1. full strict prompt with added applicability rules,
  2. shortened “more stable” prompt.

## Working preference confirmed
- User asked whether textual issue summary or tree JSON annotations are better.
- Confirmed and aligned:
  - node-level JSON tree annotations provide higher correction precision and easier incremental rule merge.

---

## Conversation Memory (2026-03-06, Updated)

## Context
- User continued iterative correction of electrical controlled KG using multiple same-day tree exports:
  - `electrical_test_tree_20260306_095258.json`
  - `electrical_test_tree_20260306_110358.json`
  - `electrical_test_tree_20260306_125611.json`
  - `electrical_test_tree_20260306_144334.json`
  - `electrical_test_tree_20260306_145055.json`
  - `electrical_test_tree_20260306_162326.json`
  - `electrical_test_tree_20260306_171032.json`
- Primary goal: make annotation-driven memory truly converge, especially for:
  - missing test items under `开合性能型式试验`,
  - feature-level missing parameters under `特征值`,
  - stable parameter values for six capacitive switching test items:
    `LC1/LC2/CC1/CC2/BC1/BC2`,
  - repeated failure of `操作顺序` to follow expert-corrected values.

## Main Findings

### 1) Why repeated annotation did not appear to converge
- Root cause was not only annotation coverage.
- Three concrete causes were identified:
  1. graph storage upsert behavior kept historical noisy data unless storage was rebuilt;
  2. runtime config still allowed extracted values to leak through because guardrails were not strict enough;
  3. override matching could fail under `strict` mode when same test name had multiple candidate rules/paths, causing fallback to raw extracted values.

### 2) Typical failure pattern confirmed from latest trees
- `操作顺序` for capacitive switching tests was repeatedly truncated to partial values like `C1级单相`.
- `试验电流A` often mixed with broken table strings like `I1(3.6 ...`.
- `原文切块` nodes still appeared in exported tree, showing extraction noise remained present upstream.
- Therefore, annotation memory itself was partly correct, but runtime application was unstable.

## Config Changes Applied

### 1) Tightened annotation control in `config.ini`
- Updated:
  - `annotation_guardrail_only_override = true`
  - `strict_tree_override_match = true`
  - `override_param_filter_to_template = true`
  - `annotation_auto_merge_to_memory = false`
- Intent:
  - prefer curated override values over extracted values,
  - avoid auto-merging fresh noisy outputs back into memory,
  - make runtime behavior more deterministic.

## Code Changes Applied

### 1) Improved override candidate matching (`lightrag/operate.py`)
- In `_match_override_rule(...)`:
  - expanded report-scope matching to also consider category-level scope and root scope `型式试验`;
  - deduplicated effectively equivalent candidates produced by alias/path duplication;
  - when ambiguity remained, preferred the richer candidate (more parameters / more rules) instead of immediately dropping under `strict` mode.
- Purpose:
  - reduce fallback to raw extraction when annotation memory actually contains the correct rule.

### 2) Validation
- Syntax check passed:
  - `python3 -m py_compile lightrag/operate.py`

## Annotation Memory Updates Applied

### 1) Bulk merge and cleanup workflow
- `lightrag/config/annotation_memory.json` was repeatedly updated from latest tree notes.
- Work included:
  - merging note-driven missing parameters,
  - removing parameters marked `这个不要`,
  - normalizing noisy parameter names/values,
  - expanding shorthand references where possible,
  - rebuilding `tests_by_name` after each patch.

### 2) Added missing test items under `开合性能型式试验`
- Ensured `add_test_items` includes:
  - `试前及试后空载特性测量`
  - `T60(预备试验)`
  - `试后状态检查(工频耐受电压试验)`
- For `T60(预备试验)`, memory includes:
  - `额定短路开断电流`
  - `试验电流A = 额定短路开断电流的60%`
  - `试验电压 = 方便电压：中压12kV，高压21kV`
  - `操作顺序 = 3个O`

### 3) Six capacitive switching tests were explicitly differentiated
- User repeatedly clarified that `试验电流A` must not be shared across projects.
- Memory was updated to distinguish:
  - `LC1`: line charging current `I1`, use `10%~40%`
  - `LC2`: cable charging breaking current `Ic`
  - `CC1`: cable charging breaking current `Ic`, use `10%~40%`
  - `CC2`: cable charging breaking current `Ic`
  - `BC1`: back-to-back capacitor bank breaking current `Ibb`, use `10%~40%`
  - `BC2`: back-to-back capacitor bank breaking current `Ibb`
- Table-1 preferred values were encoded into memory summary text:
  - `I1`, `Ic`, `Isb`, `Ibb`, `Ibi`
  - including explicit values across `3.6kV` to `1100kV`.

### 4) Repeatedly corrected `操作顺序` target values
- User confirmed the intended final values:
  - `LC1 = O`
  - `BC1 = C1级或C级均为单相48次O;三相24次O;`
  - `LC2 = C1级单相/三相均24次CO;C2级单相24次O+24次CO,三相24次CO;`
  - `CC2 = C1级单相/三相均24次CO;C2级单相24次O+24次CO,三相24次CO;`
  - `BC2 = C1级或C级均为单相120次CO;三相80次CO;`
  - `CC1 = C1级单相/三相均24次O;C2级单相48次O,三相24次O;`
- `annotation_memory.json` was patched multiple times so these values are present in memory, but latest exported tree `171032` still showed partial extracted values (`C1级单相`, `单相48次O`, `单相120次CO`) at runtime.
- Conclusion:
  - memory content is now correct,
  - remaining issue is runtime application / override precedence, not annotation data absence.

### 5) Other notable parameter corrections added to memory
- `操作冲击耐受电压试验`:
  -补 `试验相数`
  -补 `介质性质`
  -修 `试验次数 = 1`
- `控制和辅助回路的绝缘试验`:
  -补 `试验电压 = 2 kV`
- `连续电流试验`:
  -修 `试验次数 = 1次`
  -修 `试验电流A = 用户录入额定参数中的额定电流`
- `工频耐受电压试验`:
  - `交流电压` was normalized to “按表1/表4额定绝缘水平选取”
- `雷电冲击耐受电压试验`:
  - `雷电冲击干耐受电压` was normalized to “按表1/表4额定雷电冲击耐受电压选取”

## Current State At End Of Session
- User judged latest result (`electrical_test_tree_20260306_162326.json`) as “差不多了”, except:
  - capacitive switching `操作顺序` still wrong in runtime result;
  - some `开合性能型式试验` feature-level parameters were still missing and were incrementally merged.
- Latest exported tree (`electrical_test_tree_20260306_171032.json`) confirmed:
  - `annotation_memory.json` already holds the full correct `操作顺序`;
  - runtime tree still shows truncated partial values for `CC1/LC2/CC2/BC1/BC2`.

## Most Important Pending Action For Next Session
1. Restart service with latest `config.ini` and code.
2. Rebuild / clear graph storage before rerun, otherwise old noisy data will continue to pollute output.
3. Verify whether `_match_override_rule(...)` improvement eliminates fallback for capacitive tests.
4. If `操作顺序` is still wrong after fresh rebuild:
   - do not further patch memory first;
   - instead inspect runtime precedence path in `lightrag/operate.py` for `operation_sequence` specifically, because memory is already correct.

## Key Conclusion To Preserve
- The remaining blocker is no longer “user annotations are insufficient”.
- The blocker is: correct annotation memory exists, but runtime override application is still not fully dominating extracted values for some high-noise parameters, especially `操作顺序` under capacitive switching tests.

---

# Session Memory Update (2026-03-09)

## Scope Of This Session
- Continued from the earlier controlled-tree / annotation-memory workflow.
- Focus shifted from capacitive switching cleanup to:
  - latest tree validation,
  - short-circuit type-test corrections,
  - missing `电寿命试验` injection,
  - query-prompt behavior and query-mode behavior.

## Files Updated In This Session
- `lightrag/operate.py`
- `config.ini`
- `lightrag/config/annotation_memory.json`

## Tree Review Outcomes

### 1) `electrical_test_tree_20260309_103737.json`
- User clarified latest annotation intent:
  - only `绝缘性能型式试验` / `温升性能型式试验` / `开合性能型式试验` and
    `容性电流开断试验(BC1) -> 特征值` contain actionable notes;
  - for `BC1 -> 特征值 -> 操作顺序: 单相48次O`, the note text is the correct override value.
- Extracted/confirmed issues:
  - insulation tests missing `电压极性` / `试验状态`;
  - LC/CC/BC items missing `试验次数` / `试验相数` / some `试验电流A`;
  - `LC2/CC2/CC1/BC1` operation sequence values needed hard override.

### 2) `electrical_test_tree_20260309_140332.json`
- Main new issue cluster was under `短路性能型式试验`.
- User asked for careful note review under every short-circuit test node.
- Latest notes were merged into memory for:
  - `短路开断试验(T100a/T10/T30/T60/T100s)`
  - `失步关合和开断试验(OP1/OP2)`
  - `近区故障试验(L75)`
  - `单相接地故障试验`
  - `异相接地故障试验`

### 3) `electrical_test_tree_20260309_160002.json`
- User found `短路性能型式试验` still lacked `电寿命试验`.
- Confirmed exported tree did not contain that test item even though memory already had an `add_test_items` entry.
- Root cause was not missing memory data, but add-item injection gating in runtime logic.

## Code Changes Applied

### 1) `lightrag/operate.py`
- Extended note parsing so feature notes like:
  - `缺试验次数：24次`
  - `缺试验相数：...`
  - `缺试验电流A：...`
  - `缺操作顺序标准循环`
  - `缺试验项数单相`
  are recognized as missing-parameter补充 instead of ignored free text.
- Short parameter notes like `O`, `CO`, `2 kV` now act as final override values.
- Added support for explicit feature-note assignments such as `直流分量：公式...`.
- Added required-name alignment so short-circuit parameters like `试验电流A` can align to schema names like `试验电流kA` and survive whitelist filtering.
- `_resolve_override_value_text(...)` was relaxed so long logic strings containing `用户输入/客户输入` are not incorrectly truncated.
- Added payload-category fallback for `add_test_items` injection:
  - add-item rules can now be injected when report-scope does not directly match payload report names but category-scope does match payload categories.
- This category fallback was the critical runtime fix for missing `短路性能型式试验 -> 电寿命试验`.

### 2) Validation
- `python3 -m py_compile lightrag/operate.py` passed after changes.

## Config Changes Applied

### 1) `config.ini`
- Expanded short-circuit project parameter requirements so whitelist filtering no longer drops expected parameters:
  - `T10/T30/T60/T100a/T100s/L75/单相接地故障试验/异相接地故障试验/OP2` include `额定频率`
  - `T100a/T100s` include `结构特征`
  - `OP2` include `试验电流kA`
- `电寿命试验` config presence was confirmed; no extra schema addition was needed beyond memory/runtime fixes.

## Annotation Memory Changes Applied

### 1) Capacitive / insulation updates
- Merged latest note-driven corrections so:
  - `LC1/LC2/CC1/CC2/BC1` have `试验次数=24次`
  - `LC2/CC2 -> 操作顺序=CO`
  - `CC1/BC1 -> 操作顺序=O`
  - `BC2` no longer loses `试验电流A`
  - `雷电冲击耐受电压试验` includes `电压极性=正极性及负极性`

### 2) Short-circuit memory rebuild
- Merged latest short-circuit notes into `tests_by_path`.
- Rebuilt `tests_by_name` from `tests_by_path` to avoid stale name-level fallback.
- Verified representative outcomes:
  - `T100a` includes `断路器等级` / `首开极系数kpp` / `额定频率` / `试验电流kA` / `结构特征` / `直流分量` / `操作顺序=O` / `试验次数=3`
  - `T100s` includes full long-form logic for `试验电压` / `试验次数` / `操作顺序`
  - `OP2` includes `试验项数=单相` / `试验电流kA=25%客户输入额定短路开断电流` / `操作顺序=CO-O-O`
  - `L75` includes `操作顺序=标准循环`

### 3) Added short-circuit `电寿命试验`
- Replaced a previously broken generic add-item entry with a short-circuit-specific one:
  - `report_type = 短路性能型式试验`
  - `category = 短路性能型式试验`
  - `required_reports = 短路性能型式试验`
- Parameters recorded in memory:
  - `操作顺序 = 用户提供特定的按照用户提供额定操作顺序展示`
  - `试验电压 = 40.5kV及以下值为额定电压`
  - `试验电流kA = 用户录入的额定短路开断电流`
  - `关合电流 = 用户录入的额定短路关合电流`
  - `试验相数 = 40.5kV及以下默认为3相`
  - `试验次数 = 默认20次，当用户提供特定的试验次数时按照用户提供的试验次数展示`

## Query Prompt / Answering Conclusions

### 1) Why `global` looked better than `mix`
- User observed LightRAG native querying was more accurate in `global` mode than in `mix`.
- Conclusion given:
  - `global` is cleaner for this workflow because it mainly uses KG relationship context;
  - `mix` combines local entities + global relations + vector chunks, which increases noise and evidence-priority conflicts;
  - in this business scenario, `global` is usually more stable unless a strong reranker and stricter context hierarchy are used.

### 2) Prompt weaknesses discovered from real answer
- In a short-circuit report query, the LLM still output both:
  - `单相接地故障试验`
  - `异相接地故障试验`
  even though user input did not provide `首开极系数`.
- Wrong reasoning used by the model:
  - defaulting `kpp=1.5`,
  - then using that default to trigger project applicability.
- This was identified as a prompt-logic issue:
  - applicability-gate parameters must never be defaulted to create project applicability;
  - default values may only be used after a project is already confirmed applicable.

### 3) Prompt changes required / drafted
- Added stronger applicability constraints:
  - gate parameters such as `首开极系数`, `S2级状态`, `失步额定值`, BC base currents, `机构是否带合分闸线圈` cannot be defaulted to trigger projects.
- Added explicit suppression rules:
  - if `首开极系数` cannot be uniquely determined, neither `单相接地故障试验` nor `异相接地故障试验` may appear in A/B/C/D;
  - if either is output, source must explicitly show unique `kpp=1.3` or `kpp=1.5` evidence.
- Removed the old restriction `T60(预备试验) only for 未进行过T60的C2级断路器`.
- New agreed prompt direction:
  - treat `T60(预备试验)` as a direct short-circuit test project without that earlier C2-only gate.

## Important User Clarifications To Preserve
- Current workflow:
  1. user uploads documents;
  2. system builds graph;
  3. graph stores feature values for each test item;
  4. feature values can be:
     - fixed values from standard,
     - rule texts that must be conditionally resolved by LLM at answer time,
     - formulas that must be calculated from user-rated inputs.
- Therefore the answering prompt must explicitly separate:
  - fixed values,
  - conditionally-resolved values,
  - formula-calculated values.

## Most Important Pending Action For Next Session
1. User will test the newly tightened prompt in real querying.
2. If `单相接地故障试验` / `异相接地故障试验` still leak into results without unique `kpp` evidence, the next investigation should move from prompt wording to actual query-context composition in LightRAG.
3. After next rerun / next query output, verify whether:
  - `短路性能型式试验` now includes `电寿命试验`,
  - `T60(预备试验)` appears according to the new prompt policy,
  - false-positive ground-fault tests are suppressed when `kpp` is missing.

---

# Session Memory Update (2026-03-10)

## Scope Of This Session
- User asked to review prior conversation memory, then focused on a query-vs-graph inconsistency:
  - graph/tree showed `工频耐受电压试验 -> 介质性质 = 正常或充气充油`
  - LLM answer rewrote it to `空气`
- Goal of this session:
  1. understand whether `global` mode still mixes graph + chunks,
  2. make query-time answers obey graph values more strictly,
  3. avoid over-freezing values that still require user input / condition resolution / calculation,
  4. persist the above reasoning into memory.

## Key Findings Confirmed

### 1) `global` mode is not “graph direct return”
- Confirmed from query context template that `global` mode still provides mixed context to the LLM:
  - `Knowledge Graph Data (Entity)`
  - `Knowledge Graph Data (Relationship)`
  - `PROJECT_PARAM_MAP`
  - `Document Chunks`
- Therefore, even in `global`, the LLM can still rewrite parameter values unless query context / prompt priority is tightened.

### 2) Root cause of graph-vs-answer inconsistency
- Before this session’s code change:
  - query side only injected parameter-name whitelist (`PROJECT_PARAM_MAP`),
  - did not inject a structured parameter-value map,
  - prompt did not force “if graph already has a unique final value, use it verbatim”.
- As a result, the LLM could treat graph values as just one evidence source and override them with chunk text or business-default reasoning.

## Code Changes Applied (This Session)

### 1) Query-time graph value injection (`lightrag/operate.py`)
- Added query-time build of `PROJECT_PARAM_VALUE_MAP` alongside existing `PROJECT_PARAM_MAP`.
- For each retrieved test item, query path now reads parameter nodes and injects:
  - `value_text`
  - `value_source`
  - `value_expr`
  - `unit`
  - `constraints`
  - `calc_rule`
  - `derive_from_rated`
- Added query metadata return:
  - `raw_data.metadata.project_param_value_map`
- Purpose:
  - allow `/query/data` inspection of actual graph values seen by the LLM at query time.

### 2) Query context template extended (`lightrag/prompt.py`)
- `PROMPTS["kg_query_context"]` now includes:
  - `Graph Parameter Values (PROJECT_PARAM_VALUE_MAP)`
- This makes graph parameter values explicit in the LLM context instead of leaving the model to infer them from entity strings only.

### 3) First prompt tightening: graph values as highest priority
- Initial change:
  - if `PROJECT_PARAM_VALUE_MAP` contains a unique non-empty value, the model must not override it with chunks/defaults.
- Immediate effect:
  - fixed graph drift cases like `介质性质` being rewritten away from graph/tree value.

### 4) Regression found after first tightening
- User reported a new failure mode:
  - values like `用户录入额定参数中的额定电流`
  - `40.5kV及以下默认三相...`
  were copied out as if they were already final values.
- Example:
  - user query clearly included `额定电流=1250A`,
  - but answer still returned rule text rather than resolved final value.
- Conclusion:
  - “graph value priority” was made too strict;
  - query side needed to distinguish final graph values from rule-like graph hints.

### 5) Resolution mode classification added (`lightrag/operate.py`)
- Added `_classify_query_value_resolution_mode(...)` in query pipeline.
- Each query-time parameter is now classified into one of:
  - `graph_final`
  - `needs_user_input`
  - `needs_formula`
  - `needs_condition`
  - `missing`
- Classification uses graph node fields:
  - `value_text`
  - `value_source`
  - `value_expr`
  - `constraints`
  - `calc_rule`
  - `derive_from_rated`
- Intended semantics:
  - `graph_final`: safe to output directly
  - `needs_user_input`: requires device inputs from user query
  - `needs_formula`: requires formula evaluation
  - `needs_condition`: requires branch/condition/table/default resolution
  - `missing`: no usable graph value

### 6) Prompt adjusted to obey `resolution_mode`
- Prompt was updated so that:
  - only `resolution_mode=graph_final` is hard-locked to the graph value,
  - `needs_user_input` must combine graph hint + user inputs,
  - `needs_formula` must calculate,
  - `needs_condition` must resolve conditionally or return `无法确定`.
- Goal:
  - keep graph-final parameters stable,
  - while still allowing dynamic resolution for user-input / formula / condition parameters.

### 7) Second regression found: some branch-like texts still misclassified
- User reported examples still being copied as fixed outputs, e.g.:
  - `S1或S2`
  - `额定50Hz做50Hz试验，额定60Hz做60Hz试验`
  - `单相试验影响试验电压`
- Conclusion:
  - classification still under-detected “branch/enum/conditional” texts.

### 8) Condition classification hardened (`lightrag/operate.py`)
- Expanded `needs_condition` detection to cover:
  - `则`
  - `或`
  - `和/或`
  - `影响`
  - `有关`
  - `分箱`
  - `共箱`
  - `单相试验`
  - `三相试验`
- Added regex-style detection for branch texts such as:
  - `S1或S2`
  - `额定50Hz做50Hz试验...`
  - voltage/phase branch patterns
  - user-choice wording
- Goal:
  - prevent branch/rule text from leaking as final values in answers.

## Validation
- Syntax checks passed after query-chain changes:
  - `python3 -m py_compile lightrag/operate.py lightrag/prompt.py`
  - later incremental check: `python3 -m py_compile lightrag/operate.py`
- No graph rebuild required for this session’s changes; these are query-side only.

## Operational Guidance Preserved
- After this session’s query-chain changes:
  1. restart service to load new query code,
  2. graph rebuild is generally unnecessary unless stored graph values themselves are stale/noisy,
  3. use `/query/data` to inspect:
     - `metadata.project_param_map`
     - `metadata.project_param_value_map`
- This is now the primary way to debug “graph says X, answer says Y” mismatches.

## Important Concept To Preserve
- Query-time behavior is now intended to work as:
  - graph final value -> answer must follow graph
  - graph rule needing user input -> resolve using user query inputs
  - graph formula -> calculate
  - graph branch/condition text -> resolve conditionally or say `无法确定`
- The remaining debugging lens for future sessions should be:
  - whether a parameter was misclassified into the wrong `resolution_mode`,
  - not just whether retrieval hit the right graph node.

---

# Session Memory Update
Timestamp: 2026-03-11 20:14:14 CST

## 1) Annotation memory / config cleanup status
- `lightrag/config/annotation_memory.json` was previously cleaned and now definitely contains these `add_test_items`:
  - `T60(预备试验)`
  - `电寿命试验`
  - `短时耐受电流试验`
  - `峰值耐受电流试验`
- `config.ini` was patched so schema whitelist now includes:
  - `T60(预备试验)`
  - `短时耐受电流试验`
  - `峰值耐受电流试验`
- `config.ini` was also patched to add missing parameter mappings:
  - `短时耐受电流`
  - `峰值电流kA`
  - `试验工位`
  - `短路持续时间`
- `config.ini` `test_item_param_requirements` was updated for:
  - `T60(预备试验)`
  - `短时耐受电流试验`
  - `峰值耐受电流试验`

## 2) Root cause of “new tree still unchanged” was not missing rules
- User reported newly exported trees still did not include:
  - `T60(预备试验)`
  - `电寿命试验`
  - `短时耐受电流试验`
  - `峰值耐受电流试验`
- Investigation found the running Docker container was not using this repo at first.
- `docker inspect lightrag` showed mounts pointed at:
  - `/Users/df/workspace/python/LightRAG/...`
  instead of:
  - `/Users/df/workspace/python/LightRAG-github/...`
- Therefore earlier runtime results came from another workspace’s old `annotation_memory.json`, not from the edited file in this repo.

## 3) Docker compose was corrected in current repo
- `docker-compose.yml` was rewritten to a single-service layout:
  - service name: `lightrag-2`
  - port mapping: `${PORT:-9622}:9621`
- Volume mounts now correctly point to current repo paths, including:
  - `./lightrag/config/annotation_memory.json:/app/lightrag/config/annotation_memory.json`
  - `./config.ini:/app/config.ini`
  - `./lightrag/operate.py:/app/lightrag/operate.py`
  - `./lightrag/lightrag.py:/app/lightrag/lightrag.py`
  - `./lightrag/prompt.py:/app/lightrag/prompt.py`
- This compose file is now the intended runtime entry for this repo.

## 4) Another missing-project cause was found in code path
- In `lightrag/operate.py`, `add_test_items` are injected first, but then all test items are filtered by `config.ini -> test_items`.
- This explained why:
  - `T60(预备试验)`
  - `短时耐受电流试验`
  - `峰值耐受电流试验`
  were still dropped before the `config.ini` patch.
- Separate issue:
  - `电寿命试验` was not just missing from schema filtering.
  - `annotation_memory.json` still had historical `skip=true` overrides under:
    - `开合性能型式试验 > 开合性能型式试验 > 电寿命试验`
    - `型式试验 > 开合性能型式试验 > 电寿命试验`
- These stale skip rules were deleted.
- `tests_by_name["电寿命试验"]` was reduced to `[]` to stop accidental same-name suppression.

## 5) Current graph storage observations (`data/rag_storage`)
- `data/rag_storage` currently contains:
  - `graph_chunk_entity_relation.graphml`
  - `kv_store_doc_status.json`
  - `kv_store_full_entities.json`
  - `kv_store_full_relations.json`
  - `kv_store_text_chunks.json`
  - `vdb_chunks.json`
  - `vdb_relationships.json`
- Observed timestamps were inconsistent:
  - `kv_store_full_entities.json` / `kv_store_full_relations.json` updated later
  - `graph_chunk_entity_relation.graphml` remained older
- This indicates storage may be in a mixed or partially rebuilt state.

## 6) IEC / DLT tree-view debugging conclusions
- User switched to IEC mode and turned these config flags off:
  - `annotation_guardrail_mode = false`
  - `annotation_guardrail_only_override = false`
  - `strict_tree_override_match = false`
  - `override_param_filter_to_template = false`
- This means graph labels now follow extracted report naming more closely and no longer align with GBT-only assumptions.
- Existing `electrical_test_tree.html` is tightly coupled to GBT:
  - hard-coded four Chinese root categories
  - fixed GBT template tree
- This is why it cannot serve IEC / DLT graph browsing directly.

## 7) New dynamic tree pages created during this session
- Created:
  - `electrical_test_tree_std.html`
  - `electrical_test_tree_dynamic.html`
- Goal:
  - preserve original visual style,
  - support IEC / DLT by discovering roots dynamically via `/graph/label/search`,
  - then loading graph data via `/graphs`.
- First versions failed because they treated any discovered `report:*` label as a tree root.
- User exports confirmed those pages were loading empty roots only.

## 8) Key graph-structure finding for IEC/DLT runtime
- Inspection of `data/rag_storage/graph_chunk_entity_relation.graphml` found:
  - nodes like `report:Short-circuit Performance`, `report:Switching Performance`, `report:Continuous Current Performance`
    exist,
    but they do **not** have `INCLUDES_TEST` outgoing edges.
- Instead, the actual test-bearing root is:
  - `report:型式试验`
- Confirmed edges:
  - `report:型式试验 -> test:短路开断试验(T10)` with `rel=INCLUDES_TEST`
  - similar for other short-circuit test items
- Example misleading nodes:
  - `report:Short-circuit Performance`
  - `report:Switching Performance`
  - `report:STL Type Test Certificate ...`
  are mostly report-semantic nodes and not tree roots.

## 9) Latest dynamic page patch status
- `electrical_test_tree_dynamic.html` was patched again to improve root discovery:
  - add fallback probe for `report:型式试验`
  - for each discovered `report:*` candidate, fetch one-hop graph and keep only candidates with:
    - `INCLUDES_TEST` edge
    - source == candidate root
    - target starts with `test:`
- Debug output added:
  - `root_check -> HAS_INCLUDES_TEST`
  - `root_check -> NO_INCLUDES_TEST`
- Intended effect:
  - filter out empty IEC report labels like `Short-circuit Performance`
  - keep only actual test-bearing report roots

## 10) Important uncertainty at end of session
- User pasted logs still showing old behavior:
  - loading empty roots such as `report:Short-circuit Performance`
  - no `root_check` lines visible
- Strong likelihood:
  - browser was still using an older cached HTML,
  - or user opened an older generated page instead of the latest patched `electrical_test_tree_dynamic.html`.
- This was not fully re-verified within this session.

## 11) Recommended first step for next session
- Re-open the latest file:
  - `electrical_test_tree_dynamic.html`
- Hard-refresh browser / use incognito
- Click:
  1. `发现根`
  2. confirm debug shows `root_check -> ...`
  3. then `加载树`
- If still failing:
  - inspect actual browser-loaded HTML version,
  - or add an explicit visible version banner such as `dynamic-v2` to remove cache ambiguity.

## 12) Practical takeaway to preserve
- For IEC / DLT graph browsing, do **not** assume:
  - discovered `report:*Performance` labels are tree roots.
- The graph may still organize test items under a single Chinese root:
  - `report:型式试验`
- Therefore any future tree UI must identify roots by structure:
  - “has `INCLUDES_TEST` outgoing edges”
  - not merely by report label text.
