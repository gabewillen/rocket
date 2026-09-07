"""Immutable Qwen3.8 QSA compressed top-k expansion contract."""

from __future__ import annotations

from dataclasses import dataclass

QSA_COMPRESS_RATIO = 4
QSA_TOKEN_TOPK = 2048
QSA_BLOCK_TOPK = QSA_TOKEN_TOPK // QSA_COMPRESS_RATIO
QSA_EXPANDED_WIDTH = QSA_TOKEN_TOPK + QSA_COMPRESS_RATIO - 1
QSA_INDEXER_SCHEMA = "qwen3.8-flash-next:qsa-indexer-expand:c16:v1"


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
