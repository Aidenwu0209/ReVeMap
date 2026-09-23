"""Bounded language -> validated query -> measured graph evidence.

No model-generated objects, executable expressions, or implicit reference choice.
Counts describe detected instances in the current map, never room completeness.
"""
import re

from .scene_graph import _label, _validate_graph, query_graph

RELATIONS = {"near", "above", "below", "inside", "contains", "supported_by", "nearest"}
SCHEMA = "revemap.scene_query.v1"
PLURALS = {"chairs": "chair", "tables": "table", "monitors": "monitor", "books": "book",
           "bottles": "bottle", "cups": "cup", "cabinets": "cabinet", "lamps": "lamp",
           "doors": "door", "windows": "window", "sofas": "sofa", "beds": "bed",
           "boxes": "box", "curtains": "curtain", "cameras": "camera", "tripods": "tripod"}
CN_RELATIONS = {"附近": "near", "旁边": "near", "上方": "above", "下方": "below",
                "包围盒内": "inside", "包围盒里面": "inside", "包围盒包含": "inside",
                "最近": "nearest"}


class UnsupportedQuery(ValueError):
    pass


def label_text(text):
    text = text.strip().lower()
    text = re.sub(r"^(?:the |all )", "", text)
    text = re.sub(r"^(?:所有的|所有|全部的|全部)", "", text)
    if text in {"", "物体", "对象", "东西", "什么", "哪些物体", "哪些对象", "objects", "object", "things"}:
        return None
    # Deliberately do not strip unknown adjectives, negation, or unsupported clauses.
    if len(text) > 80 or re.search(r"[，,。?？;；{}\[\]]|(?:且|或者|但是|除了|不要|不是)", text):
        raise UnsupportedQuery("无法解析对象条件；请使用类别名称或实例编号。")
    return _label(PLURALS.get(text, text))


def reference_text(text):
    text = re.sub(r"^(?:位于|在|离|距离)", "", text.strip())
    text = re.sub(r"的$", "", text).strip()
    match = re.fullmatch(r"(?:实例\s*#?\s*|#|instance\s*#?\s*)([1-9]\d*)", text, re.I)
    if match:
        return {"instance_id": int(match[1])}
    name = label_text(text)
    if not name:
        raise UnsupportedQuery("请明确参考对象的类别或实例编号。")
    return {"label": name}


def parse_question(question):
    if not isinstance(question, str) or not 0 < len(question.strip()) <= 512:
        raise UnsupportedQuery("问题需要为 1—512 字符。")
    text = question.strip().rstrip("？?。.!！").strip().lower()
    if re.search(r"(?:为什么|颜色|红色|蓝色|绿色|材质|能否|或者|除了|不要|不是|不在|没有|\bor\b|\bnot\b|\bred\b|\bblue\b)", text):
        raise UnsupportedQuery("当前支持对象查找、空间关系、且条件与计数；此问题包含不支持的条件。")
    operation = "count" if re.search(r"多少|几[个把张本台盏]?|数量|how many|^count\b", text) else "find"
    text = re.sub(r"^(?:请问|请|帮我)", "", text)
    text = re.sub(r"^(?:当前地图中|当前地图里|地图中|地图里|场景中|场景里)", "", text)
    text = re.sub(r"^(?:检测到了?|识别到了?|找出|查找|查询|列出|显示所有|看看)", "", text)
    text = re.sub(r"^(?:一共|总共)?(?:有)?(?:多少|几)[个把张本台盏]?", "", text)
    text = re.sub(r"(?:的)?数量(?:是多少)?$", "", text)
    conditions = []
    if re.search(r"[\u4e00-\u9fff]", text):
        parts = re.split(r"(?:并且|而且|且|同时)", text)
        target = None
        for index, part in enumerate(parts):
            part = part.strip()
            match = re.fullmatch(r"(.+?)(包围盒里面|包围盒包含|包围盒内|附近|旁边|上方|下方|最近)(.*)", part)
            if match:
                ref, relation, tail = match.groups()
                conditions.append({"relation": CN_RELATIONS[relation], "reference": reference_text(ref)})
                tail = re.sub(r"^(?:的|有|是|可以找到)*", "", tail)
                tail = re.sub(r"^(?:哪些|多少|几)[个把张本台盏]?", "", tail)
                tail = re.sub(r"^(?:的|有)", "", tail)
                if tail:
                    if index != len(parts) - 1:
                        raise UnsupportedQuery("请将目标对象写在最后一个关系条件之后。")
                    target = label_text(tail)
            elif len(parts) == 1:
                text = re.sub(r"^(?:有哪些|有什么|有)", "", part)
                target = label_text(text)
            else:
                raise UnsupportedQuery("不能完整解析组合条件；请使用“桌子附近且显示器下方的物体”。")
    else:
        text = re.sub(r"^(?:how many |count |find |list |show )", "", text)
        text = re.sub(r"\s+(?:are there|were detected|are detected)(?: in (?:the )?(?:current )?map)?$", "", text)
        text = re.sub(r"\s+(?:that are|which are|are|is)\s+", " ", text)
        text = re.sub(r"^what(?: objects)? (?:are|is) ", "objects ", text)
        parts = re.split(r"\s+and\s+", text)
        target = None
        for index, part in enumerate(parts):
            match = re.fullmatch(r"(.*?)\b(near|above|below|inside|nearest to)\s+(.+)", part)
            if match:
                prefix, relation, ref = match.groups()
                if prefix.strip():
                    if index:
                        raise UnsupportedQuery("A target is only allowed before the first relation.")
                    target = label_text(prefix)
                conditions.append({"relation": "nearest" if relation == "nearest to" else relation,
                                   "reference": reference_text(ref)})
            elif len(parts) == 1:
                target = label_text(part)
            else:
                raise UnsupportedQuery("Cannot parse every condition; use 'chairs near the table and below the monitor'.")
    return validate_plan({"schema": SCHEMA, "operation": operation, "label": target, "conditions": conditions})


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) - {"schema", "operation", "label", "conditions"}:
        raise ValueError("Invalid query fields")
    if plan.get("schema", SCHEMA) != SCHEMA or plan.get("operation", "find") not in {"find", "count"}:
        raise ValueError("Unsupported query operation")
    label = plan.get("label")
    if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > 80):
        raise ValueError("Invalid target label")
    conditions = plan.get("conditions", [])
    if not isinstance(conditions, list) or len(conditions) > 8:
        raise ValueError("At most eight AND conditions are supported")
    for condition in conditions:
        if not isinstance(condition, dict) or set(condition) != {"relation", "reference"} or condition['relation'] not in RELATIONS:
            raise ValueError("Invalid relation condition")
        ref = condition["reference"]
        if not isinstance(ref, dict) or set(ref) not in ({"label"}, {"instance_id"}):
            raise ValueError("A reference requires exactly one label or instance ID")
        if "instance_id" in ref:
            if type(ref['instance_id']) is not int or ref['instance_id'] <= 0:
                raise ValueError("Invalid reference instance")
        elif not isinstance(ref['label'], str) or not ref['label'].strip() or len(ref['label']) > 80:
            raise ValueError("Invalid reference label")
    if sum(c['relation'] == 'nearest' for c in conditions) > 1:
        raise ValueError("Only one nearest ranking is supported")
    return {"schema": SCHEMA, "operation": plan.get("operation", "find"),
            "label": _label(label) if label is not None else None, "conditions": conditions}


def execute_query(graph, *, question=None, plan=None):
    if (question is None) == (plan is None):
        raise ValueError("Supply exactly one question or query plan")
    nodes = _validate_graph(graph)
    base = {"instance_ids": [], "objects": [], "evidence": [], "count": None,
            "count_scope": "detected_instances_in_current_map", "provenance": graph.get("provenance", {"bound": False}),
            "relation_scope": graph["relation_scope"]}
    try:
        plan = parse_question(question) if question is not None else validate_plan(plan)
    except UnsupportedQuery as error:
        return {**base, "status": "unsupported", "reason": "unsupported_question", "answer": str(error)}
    base['query'] = plan
    selected = set(query_graph(graph, label=plan['label'])['instance_ids'])
    evidence, resolved = [], []
    for condition in plan['conditions']:
        reference = condition['reference']
        matches = ([nodes[reference['instance_id']]] if reference.get('instance_id') in nodes else []) if 'instance_id' in reference else query_graph(graph, label=reference['label'])['objects']
        if not matches:
            return {**base, "status": "unknown", "reason": "reference_not_found", "answer": "当前地图未找到参考对象，无法判断。", "reference": reference}
        if len(matches) > 1:
            return {**base, "status": "ambiguous", "reason": "ambiguous_reference", "answer": "存在多个参考对象，请用实例编号明确选择。",
                    "reference": reference, "candidates": [{k: n[k] for k in ('instance_id', 'label', 'name', 'centroid_m')} for n in matches]}
        resolved.append((condition['relation'], matches[0]['instance_id']))
    # Filter all logical conditions first; nearest ranks the resulting candidates.
    for relation, oid in sorted(resolved, key=lambda item: item[0] == 'nearest'):
        if relation == 'nearest':
            import numpy as np
            selected.discard(oid)
            distances = {i: float(np.linalg.norm(np.asarray(nodes[i]['centroid_m']) - nodes[oid]['centroid_m'])) for i in selected}
            if distances:
                best = min(distances.values())
                selected = {i for i, d in distances.items() if np.isclose(d, best, rtol=0, atol=1e-6)}
                evidence.extend({'instance_id': i, 'reference_id': oid, 'centroid_distance_m': distances[i]} for i in sorted(selected))
            continue
        result = query_graph(graph, relation=relation, reference_id=oid)
        if result.get('reason') == 'world_up_required':
            return {**base, 'status': 'unknown', 'reason': 'world_up_required', 'answer': '未确定地图的向上方向，不能判断上方或下方关系。'}
        selected.intersection_update(result['instance_ids'])
        evidence.extend(result['evidence'])
    ids = sorted(selected)
    evidence = [e for e in evidence if e.get('instance_id') in selected or e.get('source') in selected or e.get('target') in selected]
    answer = f"当前地图检测到符合条件的对象 {len(ids)} 个。" if plan['operation'] == 'count' else ("匹配实例：" + "、".join(f"#{i}" for i in ids) if ids else "当前地图未找到符合条件的对象。")
    return {**base, 'status': 'matched' if ids or plan['operation'] == 'count' else 'unknown',
            'reason': None if ids else 'no_matching_object', 'instance_ids': ids,
            'objects': [nodes[i] for i in ids], 'evidence': evidence, 'count': len(ids),
            'answer': answer, 'resolved_references': [{'relation': r, 'instance_id': i} for r, i in resolved]}
