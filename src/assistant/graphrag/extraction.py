"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

SYSTEM = """Ты извлекаешь граф знаний из документов сервисной службы холодильного оборудования.
Верни JSON с двумя списками: entities и relations. Извлекай только то, что прямо сказано в тексте.

Типы сущностей и запись имени:
- ProductLine – линейка: Арктика, Бриз, Гольфстрим
- Model – код модели: AR-210, BZ-110, GS-410
- Component – вид узла строчными буквами в именительном падеже: пусковое реле, компрессор
- Part – номер детали: P-1042, P-1042B
- ErrorCode – код ошибки: E-17
- Procedure – номер регламента: SP-104
- Supplier – название поставщика без кавычек и «ООО»: ХолодКомплект

Типы связей. Направление строго от первого типа ко второму:
- Model BELONGS_TO ProductLine – модель входит в линейку
- Model HAS_PART Part – в модели установлена деталь
- Part IS_A Component – деталь является узлом этого вида
- ErrorCode INDICATES Component – ошибка указывает на отказ узла
- ErrorCode FIXED_BY Procedure – ошибка устраняется по регламенту
- Part REPLACED_BY Part – старая деталь снята с производства и заменена новой (от старой к новой)
- Part SUPPLIED_BY Supplier – деталь поставляет поставщик
- Model FAILED Component – в отчёте о выезде зафиксирован отказ узла этой модели

Номера бюллетеней и отчётов (SB-2025-03, R-2025-017) – это имена документов, не сущности.
Если в тексте нет сущностей схемы, верни пустые списки."""

EXAMPLE_TEXT = """# Сервисный бюллетень SB-2025-99

Деталь P-2054 (компрессор) снята с производства. Вместо неё устанавливается деталь P-2054B.
Поставщик детали P-2054B – «КомпрессорМаш»."""

# Раздел «Состав узлов» руководства даёт около 25 связей, в 1024 токена такой JSON не помещается.
# Вместе с запросом (около 1200 токенов) ответ должен уложиться в контекст Ollama по умолчанию, 4096.
MAX_TOKENS = 2800
RETRY_HINT = "Прошлый ответ не разобрался как JSON по схеме. Верни только корректный JSON, без пояснений."

_RELATION_FIELDS = ("subject_type", "subject", "type", "object_type", "object")


def _example_relation(*values: str) -> dict:
    return dict(zip(_RELATION_FIELDS, values, strict=True))


EXAMPLE_REPLY = {
    "entities": [
        {"type": "Part", "name": "P-2054"},
        {"type": "Part", "name": "P-2054B"},
        {"type": "Component", "name": "компрессор"},
        {"type": "Supplier", "name": "КомпрессорМаш"},
    ],
    "relations": [
        _example_relation("Part", "P-2054", "REPLACED_BY", "Part", "P-2054B"),
        _example_relation("Part", "P-2054", "IS_A", "Component", "компрессор"),
        _example_relation("Part", "P-2054B", "IS_A", "Component", "компрессор"),
        _example_relation("Part", "P-2054B", "SUPPLIED_BY", "Supplier", "КомпрессорМаш"),
    ],
}


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_TEXT},
        {"role": "assistant", "content": json.dumps(EXAMPLE_REPLY, ensure_ascii=False)},
        {"role": "user", "content": chunk.text},
    ]
    for attempt in range(2):
        if attempt:
            # Повтор должен отличаться от первого запроса: иначе кеш вернёт тот же сломанный ответ.
            messages = [*messages[:-1], {"role": "user", "content": f"{RETRY_HINT}\n\n{chunk.text}"}]
        reply = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=MAX_TOKENS)
        parsed = _parse(reply.content)
        if parsed is not None:
            return _ground(chunk, *parsed)
    return ExtractionResult(chunk.chunk_id, chunk.doc_id)


def _parse(content: str) -> tuple[list[dict], list[dict]] | None:
    """Списки сущностей и связей, если ответ соответствует схеме, иначе None."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    entities, relations = data.get("entities"), data.get("relations")
    if not isinstance(entities, list) or not isinstance(relations, list):
        return None
    if not all(_has_strings(item, ("type", "name")) for item in entities):
        return None
    if not all(_has_strings(item, _RELATION_FIELDS) for item in relations):
        return None
    return entities, relations


def _has_strings(item, keys: tuple[str, ...]) -> bool:
    return isinstance(item, dict) and all(isinstance(item.get(key), str) for key in keys)


def _ground(chunk: Chunk, raw_entities: list[dict], raw_relations: list[dict]) -> ExtractionResult:
    """Оставляет только то, что подтверждается текстом чанка и словарём."""
    codes = set(find_codes(chunk.text))
    vocab = vocabulary()

    def resolve(entity_type: str, raw: str) -> Entity | None:
        name = canonical_name(entity_type, raw, vocab)
        if name is None or (entity_type in CODE_TYPES and name not in codes):
            return None
        return Entity(entity_type, name)

    entities: dict[str, Entity] = {}
    relations: dict[tuple, Relation] = {}
    dropped = 0
    for item in raw_entities:
        entity = resolve(item["type"], item["name"])
        if entity is None:
            dropped += 1
        else:
            entities.setdefault(entity.node_id, entity)
    for item in raw_relations:
        subject = resolve(item["subject_type"], item["subject"])
        obj = resolve(item["object_type"], item["object"])
        relation = Relation(subject, item["type"], obj) if subject and obj else None
        if relation is None or not relation.matches_schema() or subject == obj:
            dropped += 1
            continue
        relations.setdefault((subject.node_id, relation.type, obj.node_id), relation)
        for end in (subject, obj):
            entities.setdefault(end.node_id, end)
    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), list(relations.values()), dropped)
