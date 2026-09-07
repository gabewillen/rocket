"""Immutable Qwen3.8 QSA compressed top-k expansion contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

QSA_COMPRESS_RATIO = 4
QSA_TOKEN_TOPK = 2048
QSA_BLOCK_TOPK = QSA_TOKEN_TOPK // QSA_COMPRESS_RATIO
QSA_EXPANDED_WIDTH = QSA_TOKEN_TOPK + QSA_COMPRESS_RATIO - 1
QSA_HEADS = 4
QSA_HEAD_DIM = 128
QSA_PAGE_SIZE = 64
QSA_MAX_BLOCKS = 65536
QSA_INDEXER_SCHEMA = "qwen3.8-flash-next:qsa-indexer-score-select-expand:c16:v2"


class QsaIndexerError(RuntimeError):
    """Compressed-selection or causal metadata contract failure."""


@dataclass(frozen=True)
class QsaIndexerContract:
    """Fixed post-top-k ABI shared by K0 and later lazy-depth graphs."""

    schema: str = QSA_INDEXER_SCHEMA
    max_rows: int = 16
    block_topk: int = QSA_BLOCK_TOPK
    compress_ratio: int = QSA_COMPRESS_RATIO
    token_topk: int = QSA_TOKEN_TOPK
    output_width: int = QSA_EXPANDED_WIDTH
    heads: int = QSA_HEADS
    head_dim: int = QSA_HEAD_DIM
    page_size: int = QSA_PAGE_SIZE
    max_blocks: int = QSA_MAX_BLOCKS


def reference_select_qsa_blocks(
    query: tuple[tuple[tuple[float, ...], ...], ...],
    keys: tuple[tuple[float, ...], ...],
    visible_blocks: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Reference Qwen score and deterministic top-k with invalid -1 padding."""

    if (
        not 1 <= len(query) <= 16
        or len(visible_blocks) != len(query)
        or any(len(heads) != QSA_HEADS for heads in query)
        or any(len(head) != QSA_HEAD_DIM for heads in query for head in heads)
        or any(len(key) != QSA_HEAD_DIM for key in keys)
        or any(not 0 <= visible <= len(keys) for visible in visible_blocks)
    ):
        raise QsaIndexerError("QSA score/select input ABI is invalid")
    result = []
    divisor = math.sqrt(QSA_HEAD_DIM)
    for heads, visible in zip(query, visible_blocks):
        scores = []
        for index, key in enumerate(keys[:visible]):
            score = sum(
                max(sum(q * k for q, k in zip(head, key)), 0.0)
                for head in heads
            ) / divisor
            scores.append((score, -index, index))
        scores.sort(reverse=True)
        selected = [item[2] for item in scores[:QSA_BLOCK_TOPK]]
        selected.extend([-1] * (QSA_BLOCK_TOPK - len(selected)))
        result.append(tuple(selected))
    return tuple(result)


def reference_expand_qsa_topk(
    block_indices: tuple[tuple[int, ...], ...],
    logical_positions: tuple[int, ...],
    sequence_lengths: tuple[int, ...],
    token_to_request: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Expand fixed compressed selections and append the open causal tail."""

    rows = len(logical_positions)
    if (
        not 1 <= rows <= 16
        or len(block_indices) != rows
        or len(token_to_request) != rows
        or not sequence_lengths
        or any(len(row) != QSA_BLOCK_TOPK for row in block_indices)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in logical_positions
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in sequence_lengths
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in token_to_request
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for row in block_indices
            for value in row
        )
    ):
        raise QsaIndexerError("QSA top-k expansion input ABI is invalid")
    result = []
    for row in range(rows):
        request = token_to_request[row]
        output = [-1] * QSA_EXPANDED_WIDTH
        if not 0 <= request < len(sequence_lengths):
            result.append(tuple(output))
            continue
        query_end = logical_positions[row] + 1
        sequence_length = sequence_lengths[request]
        complete_blocks = min(
            query_end // QSA_COMPRESS_RATIO,
            sequence_length // QSA_COMPRESS_RATIO,
            QSA_BLOCK_TOPK,
        )
        expanded_count = complete_blocks * QSA_COMPRESS_RATIO
        for column in range(expanded_count):
            block = block_indices[row][column // QSA_COMPRESS_RATIO]
            token = block * QSA_COMPRESS_RATIO + column % QSA_COMPRESS_RATIO
            if 0 <= token < sequence_length:
                output[column] = token
        tail_start = query_end // QSA_COMPRESS_RATIO * QSA_COMPRESS_RATIO
        tail_count = query_end - tail_start
        for offset in range(min(tail_count, QSA_COMPRESS_RATIO - 1)):
            column = expanded_count + offset
            token = tail_start + offset
            if token < sequence_length:
                output[column] = token
        result.append(tuple(output))
    return tuple(result)
