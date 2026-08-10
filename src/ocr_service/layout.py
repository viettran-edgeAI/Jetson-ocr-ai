from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass(slots=True)
class _Block:
    bbox: list[int]
    items: list[dict[str, Any]]
    column: int = 0


def _bbox(item: dict[str, Any]) -> list[int]:
    value = item.get("bbox") or [0, 0, 0, 0]
    return [int(value[0]), int(value[1]), int(value[2]), int(value[3])]


def _union_bbox(items: list[dict[str, Any]]) -> list[int]:
    boxes = [_bbox(item) for item in items]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _vertical_overlap(first: list[int], second: list[int]) -> float:
    overlap = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    shortest = max(1, min(first[3] - first[1], second[3] - second[1]))
    return overlap / shortest


def _cluster_anchors(values: list[int], threshold: int) -> list[list[int]]:
    clusters: list[list[int]] = []
    for value in sorted(values):
        if not clusters or value - clusters[-1][-1] > threshold:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return clusters


def _region_blocks(
    items: list[dict[str, Any]],
    regions: list[dict[str, Any]],
) -> tuple[list[_Block], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    unassigned: list[dict[str, Any]] = []
    for item in items:
        region_id = item.get("region_id")
        if isinstance(region_id, int) and 0 <= region_id < len(regions):
            grouped.setdefault(region_id, []).append(item)
        else:
            unassigned.append(item)

    blocks = [
        _Block(bbox=_bbox(regions[region_id]), items=region_items)
        for region_id, region_items in grouped.items()
        if region_items
    ]
    return blocks, unassigned


def _infer_blocks(items: list[dict[str, Any]]) -> list[_Block]:
    if not items:
        return []
    if len(items) == 1:
        return [_Block(bbox=_bbox(items[0]), items=list(items))]

    heights = [max(1, box[3] - box[1]) for box in map(_bbox, items)]
    anchor_threshold = max(48, int(median(heights) * 3.0))
    column_anchors = _cluster_anchors([_bbox(item)[0] for item in items], anchor_threshold)

    # An indented paragraph should not become a second column. A real column
    # needs repeated content and vertical coexistence with its neighbour.
    if len(column_anchors) > 1:
        anchors = [int(median(cluster)) for cluster in column_anchors]
        column_items: list[list[dict[str, Any]]] = [[] for _ in anchors]
        for item in items:
            x1 = _bbox(item)[0]
            index = min(range(len(anchors)), key=lambda candidate: abs(x1 - anchors[candidate]))
            column_items[index].append(item)
        repeated = all(len(group) >= 2 for group in column_items)
        coexists = any(
            _vertical_overlap(_bbox(left), _bbox(right)) > 0.25
            for left in column_items[0]
            for right in column_items[1]
        )
        if not (repeated and coexists):
            column_items = [list(items)]
    else:
        column_items = [list(items)]

    blocks: list[_Block] = []
    for group in column_items:
        ordered = sorted(group, key=lambda item: ((_bbox(item)[1] + _bbox(item)[3]) / 2.0, _bbox(item)[0]))
        if len(ordered) < 2:
            blocks.append(_Block(bbox=_union_bbox(ordered), items=ordered))
            continue
        centers = [(_bbox(item)[1] + _bbox(item)[3]) / 2.0 for item in ordered]
        gaps = [later - earlier for earlier, later in zip(centers, centers[1:]) if later > earlier]
        normal_gap = median(gaps) if gaps else median(heights)
        current: list[dict[str, Any]] = [ordered[0]]
        for index, item in enumerate(ordered[1:], start=1):
            if centers[index] - centers[index - 1] > max(48.0, normal_gap * 2.5):
                blocks.append(_Block(bbox=_union_bbox(current), items=current))
                current = []
            current.append(item)
        blocks.append(_Block(bbox=_union_bbox(current), items=current))
    return blocks


def _block_rows(blocks: list[_Block]) -> list[list[_Block]]:
    rows: list[list[_Block]] = []
    for block in sorted(blocks, key=lambda candidate: (candidate.bbox[1], candidate.bbox[0])):
        matching_row: list[_Block] | None = None
        best_overlap = 0.0
        for row in rows:
            overlap = max(_vertical_overlap(block.bbox, candidate.bbox) for candidate in row)
            if overlap >= 0.25 and overlap > best_overlap:
                matching_row = row
                best_overlap = overlap
        if matching_row is None:
            rows.append([block])
        else:
            matching_row.append(block)
    for row in rows:
        row.sort(key=lambda candidate: candidate.bbox[0])
    return sorted(rows, key=lambda row: min(block.bbox[1] for block in row))


def _assign_columns(blocks: list[_Block]) -> int:
    if not blocks:
        return 0
    widths = [max(1, block.bbox[2] - block.bbox[0]) for block in blocks]
    threshold = max(48, int(median(widths) * 0.35))
    clusters = _cluster_anchors([block.bbox[0] for block in blocks], threshold)
    anchors = [int(median(cluster)) for cluster in clusters]
    for block in blocks:
        block.column = min(range(len(anchors)), key=lambda index: abs(block.bbox[0] - anchors[index]))
    return len(anchors)


def _block_lines(block: _Block) -> list[str]:
    ordered = sorted(block.items, key=lambda item: ((_bbox(item)[1] + _bbox(item)[3]) / 2.0, _bbox(item)[0]))
    rows: list[list[dict[str, Any]]] = []
    for item in ordered:
        item_box = _bbox(item)
        matching: list[dict[str, Any]] | None = None
        for row in rows:
            if _vertical_overlap(item_box, _union_bbox(row)) >= 0.35:
                matching = row
                break
        if matching is None:
            rows.append([item])
        else:
            matching.append(item)

    lines: list[str] = []
    for row in rows:
        sorted_row = sorted(row, key=lambda item: _bbox(item)[0])
        parts = [str(item.get("text") or "").strip() for item in sorted_row]
        line = " ".join(part for part in parts if part)
        if line:
            lines.append(line)
    return lines


def reconstruct_layout(
    items: list[dict[str, Any]],
    regions: list[dict[str, Any]] | None = None,
) -> str:
    """Render OCR items as aligned plain text while preserving document blocks."""
    if not items:
        return ""

    blocks, unassigned = _region_blocks(items, regions or [])
    blocks.extend(_infer_blocks(unassigned))
    if not blocks:
        blocks = _infer_blocks(items)

    rows = _block_rows(blocks)
    column_count = _assign_columns(blocks)
    block_lines = {id(block): _block_lines(block) for block in blocks}
    column_widths = [0] * column_count
    for block in blocks:
        for line in block_lines[id(block)]:
            column_widths[block.column] = max(column_widths[block.column], len(line))

    rendered_rows: list[str] = []
    for row in rows:
        by_column = {block.column: block for block in row}
        line_count = max((len(block_lines[id(block)]) for block in row), default=0)
        rendered: list[str] = []
        for line_index in range(line_count):
            output = ""
            last_populated_column = max(by_column)
            for column in range(last_populated_column + 1):
                block = by_column.get(column)
                lines = block_lines[id(block)] if block is not None else []
                text = lines[line_index] if line_index < len(lines) else ""
                if column < last_populated_column:
                    output += text.ljust(column_widths[column] + 2)
                else:
                    output += text
            rendered.append(output.rstrip())
        if rendered:
            rendered_rows.append("\n".join(rendered))
    return "\n\n\n".join(rendered_rows)
