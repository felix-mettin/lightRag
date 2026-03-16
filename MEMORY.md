# Conversation Memory (2026-03-16, Updated)

## Context
- Workspace is `/Users/df/Downloads/绝缘问答/LightRAG-github`.
- Current focus is electrical QA query-time rule layer, graph-backed parameter recall, and final answer stabilization.
- Current working principle remains:
  - graph stores candidate facts,
  - runtime rules decide allow/deny/split/merge/override,
  - final answer should follow runtime scope and runtime overrides.

## Stable Runtime State To Remember
- `CurrentReportScope` filtering is active in `lightrag/operate.py` and should prevent cross-domain test leakage.
- Query-time prompt context already includes:
  - `Domain Rule Decisions`
  - `Resolved Rule Overrides`
  - `Allowed Final Test Items`
  - `Removed Test Items`
- `pair_merge` target-scope bug had already been fixed earlier and should not be reverted casually.

## Changes Confirmed Today

### 1) Ground-fault applicability rules were added
- In `lightrag/config/domain_rules/insulation.gb.json`:
  - `首开极系数 = 1.3` => output `单相接地故障试验`
  - `首开极系数 = 1.5` => output `异相接地故障试验`
  - if user does not provide `首开极系数` and no stronger evidence exists => default `1.5` => output `异相接地故障试验`

### 2) Electrical-life split rules were expanded
- In `lightrag/config/domain_rules/insulation.gb.json`:
  - default 20-cycle split remains `8 / 6 / 1`
  - added 30-cycle split: `13 / 11 / 1`
  - added 50-cycle split: `23 / 21 / 1`
- Default 20-cycle rule was guarded so it does not fire when query explicitly contains `30次` or `50次`.

### 3) State-check T10 applicability was added
- In `lightrag/config/domain_rules/insulation.gb.json`:
  - output `作为状态检查的T10试验` only when query is for `短路性能型式试验` and breaker is:
    - gas-insulated / `充气` / `SF6`
    - vacuum interrupter / `真空断路器` / `真空灭弧室`
    - three-phase explicitly, or implicitly inferred by `额定电压 <= 40.5kV`
- In `lightrag/operate.py`:
  - added condition support for `less_or_equal_numeric`

### 4) BC1 / BC2 graph storage was completed
- In `data/rag_storage_gb/`:
  - added missing parameter entities and graph edges for:
    - `param:容性电流开断试验(BC1):test_current_a`
    - `param:容性电流开断试验(BC2):test_current_a`
- Stored descriptions now are:
  - BC1: `用户录入额定背对背电容器组开断电流(Ibb)的10%~40%；未录入按表1优选值Ibb的10%~40%`
  - BC2: `用户录入额定背对背电容器组开断电流(Ibb)；未录入按表1优选值Ibb`

### 5) BC runtime override bug was fixed
- Root cause: `lightrag/operate.py` had a runtime hard override that forced both `BC1` and `BC2` to use raw `Ibb`.
- Current intended behavior:
  - `BC1` => `10%~40% × Ibb`
  - `BC2` => `Ibb`
- This was fixed in runtime code, not only in graph storage.

### 6) T10 stored value and runtime ratio were both fixed
- Earlier storage/config value for `短路开断试验(T10) -> 试验电流kA` had been corrected to `10%`.
- Today confirmed a second root cause in `lightrag/operate.py`:
  - runtime short-circuit ratio map still had `T10: 0.3`
- Current intended runtime mapping is:
  - `T10 = 0.1`
  - `T30 = 0.3`
  - `T60 = 0.6`
  - `T100s = 1.0`

## Current Known Good Expectations
- For short-circuit queries:
  - `T10` test current must be `10% × 额定短路开断电流`
  - `T30` test current must be `30% × 额定短路开断电流`
  - `T60` test current must be `60% × 额定短路开断电流`
- For switching queries:
  - `BC1` must not directly echo full `Ibb`; it should be a `10%~40%` range
  - `BC2` may directly use `Ibb`
- For warm-rise queries:
  - short-circuit and electrical-life items should be filtered out by report scope

## Short Remaining Caution
- If page/API result still shows old values after config/code edits, first suspect stale process memory and restart the running service.
- The main area still worth rechecking on live queries is short-circuit answer wording, especially to ensure no stale explanation text or old cached overrides leak into final output.
