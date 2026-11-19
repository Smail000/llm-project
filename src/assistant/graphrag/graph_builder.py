"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import Entity, ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    node_sources: dict[str, set[str]] = {}
    node_entities: dict[str, Entity] = {}
    # (начало, конец) -> тип связи -> чанки. В DiGraph между парой узлов одно ребро,
    # поэтому если для пары найдено несколько типов, остаётся самый подтверждённый.
    edge_sources: dict[tuple[str, str], dict[str, set[str]]] = {}

    def add_node(entity: Entity, chunk_id: str) -> None:
        node_entities.setdefault(entity.node_id, entity)
        node_sources.setdefault(entity.node_id, set()).add(chunk_id)

    for result in results:
        for entity in result.entities:
            add_node(entity, result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue
            add_node(relation.subject, result.chunk_id)
            add_node(relation.object, result.chunk_id)
            key = (relation.subject.node_id, relation.object.node_id)
            edge_sources.setdefault(key, {}).setdefault(relation.type, set()).add(result.chunk_id)

    graph = nx.DiGraph(conflicts=0)
    for node_id, entity in sorted(node_entities.items()):
        graph.add_node(node_id, type=entity.type, name=entity.name, sources=sorted(node_sources[node_id]))
    for (source, target), by_type in sorted(edge_sources.items()):
        kind, chunks = max(by_type.items(), key=lambda item: (len(item[1]), item[0]))
        graph.add_edge(source, target, type=kind, weight=len(chunks), sources=sorted(chunks))

    graph.graph["conflicts"] = _resolve_opposite_replacements(graph)
    return graph


def _resolve_opposite_replacements(graph: nx.DiGraph) -> int:
    """Убирает встречные замены деталей и возвращает число таких пар."""
    conflicts = 0
    pairs = [
        (a, b)
        for a, b, data in graph.edges(data=True)
        if data["type"] == "REPLACED_BY" and a < b and graph.edges.get((b, a), {}).get("type") == "REPLACED_BY"
    ]
    for a, b in pairs:
        conflicts += 1
        forward, backward = graph.edges[a, b]["weight"], graph.edges[b, a]["weight"]
        if forward >= backward:
            graph.remove_edge(b, a)
        if backward >= forward:
            graph.remove_edge(a, b)
    return conflicts
