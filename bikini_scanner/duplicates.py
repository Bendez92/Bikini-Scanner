"""Near-duplicate grouping.

`FolderStore.duplicate_groups` finds files that are byte-for-byte identical, which is
the easy half. The half that costs a reviewer their afternoon is the burst: eight
frames of the same pose, a second apart, no two files identical and every one of them
needing the same verdict.

Those frames sit almost on top of each other in embedding space, so grouping them is a
question about angles between vectors. Doing it exactly means comparing every image
with every other, which at 80 000 photos is 3.2 billion pairs and not an option. This
uses random-projection LSH instead: each vector is reduced to a handful of short
signatures, only vectors sharing a signature are ever compared properly, and the
survivors are merged with a union-find.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

# Cosine similarity at which two photos are "the same shot". Deliberately strict: a
# false group applies one verdict to a photo nobody looked at, which is worse than
# leaving a burst ungrouped.
DEFAULT_THRESHOLD = 0.97
# Signatures per vector, and bits per signature. Two vectors 14 degrees apart agree on
# any one random hyperplane with probability ~0.92, so a 12-bit signature matches with
# probability 0.92**12 = 0.37 and six independent signatures find them 94% of the time.
# More bands costs linear time and finds slightly more; these are the knee of that curve.
BANDS = 6
BITS = 12
# No bucket is compared beyond this many members. A degenerate bucket — a few thousand
# flat grey frames, say — would otherwise reintroduce the quadratic blow-up this whole
# approach exists to avoid.
MAX_BUCKET = 512
# Hard ceiling on candidate pairs held at once. MAX_BUCKET bounds one bucket, not the
# total, and the total grows with the square of the bucket size: a folder of 80 000
# photos of one subject — the case this exists for — buckets far more unevenly than
# random vectors, and buckets averaging ~150 members would hold tens of millions of
# pairs. Past this the grouping is merged from what it already has, so it degrades by
# finding fewer groups rather than by exhausting memory.
MAX_PAIRS = 4_000_000


class _Union:
    """Disjoint sets over vector rows, by index."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, item: int) -> int:
        parent = self._parent
        root = item
        while parent[root] != root:
            root = parent[root]
        # Path compression, iterative: a recursive find blows the stack on a long chain.
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root


def _unit_rows(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised copy, so a dot product is the cosine."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("embeddings must be a 2-D array")
    norms = np.linalg.norm(matrix, axis=1)
    # A zero vector has no direction, so it can never be near-identical to anything.
    safe = np.where(norms > 0, norms, 1.0)
    return (matrix / safe[:, None]).astype(np.float32), norms > 0


def _candidate_pairs(
    unit: np.ndarray,
    usable: np.ndarray,
    bands: int,
    bits: int,
    seed: int,
    max_bucket: int,
    max_pairs: int = MAX_PAIRS,
) -> set[int]:
    """Row pairs worth comparing exactly, from the LSH signatures.

    Pairs are packed into one integer each (`left * rows + right`) rather than kept
    as tuples: a set of ints costs roughly a third of a set of two-element tuples,
    and this set is the peak memory of the whole pass.
    """
    rows, dim = int(unit.shape[0]), int(unit.shape[1])
    rng = np.random.default_rng(seed)
    pairs: set[int] = set()
    capped = False
    weights = 1 << np.arange(bits, dtype=np.int64)
    for band in range(bands):
        planes = rng.normal(size=(dim, bits)).astype(np.float32)
        # One matmul per band, then pack the sign bits into a single integer key.
        signature = ((unit @ planes) > 0).astype(np.int64) @ weights
        order = np.argsort(signature, kind="stable")
        sorted_keys = signature[order]
        # Runs of equal signature are the buckets; boundaries are where the key changes.
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], len(sorted_keys)] if len(starts) else np.empty(0, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True):
            members = [int(index) for index in order[start:end] if usable[index]]
            if len(members) < 2:
                continue
            if len(members) > max_bucket:
                LOGGER.debug("Band %d bucket of %d capped to %d", band, len(members), max_bucket)
                members = members[:max_bucket]
            for position, left in enumerate(members):
                for right in members[position + 1 :]:
                    low, high_index = (left, right) if left < right else (right, left)
                    pairs.add(low * rows + high_index)
            if len(pairs) >= max_pairs:
                capped = True
                break
        if capped:
            break
    if capped:
        LOGGER.warning(
            "Near-duplicate search hit its %d candidate-pair ceiling; grouping from what "
            "was collected so far, so some bursts may be missed.",
            max_pairs,
        )
    return pairs


def near_duplicate_groups(
    paths: Sequence[str],
    embeddings: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    bands: int = BANDS,
    bits: int = BITS,
    seed: int = 0,
    max_bucket: int = MAX_BUCKET,
    max_pairs: int = MAX_PAIRS,
) -> list[list[str]]:
    """Group photos that are the same shot, near enough to share one verdict.

    Returns groups of two or more paths, largest first, in the order given within each
    group. Photos in no group are not returned at all.
    """
    paths = [str(path) for path in paths]
    matrix = np.asarray(embeddings, dtype=np.float32)
    if not paths or matrix.size == 0 or matrix.shape[0] != len(paths):
        return []
    unit, usable = _unit_rows(matrix)
    if not usable.any():
        return []
    pairs = _candidate_pairs(
        unit, usable, bands=bands, bits=bits, seed=seed, max_bucket=max_bucket, max_pairs=max_pairs
    )
    if not pairs:
        return []
    # One vectorised pass over the candidates rather than a dot product per pair.
    packed: np.ndarray = np.fromiter(pairs, dtype=np.int64, count=len(pairs))
    left: np.ndarray = packed // len(paths)
    right: np.ndarray = packed % len(paths)
    similarity = np.einsum("ij,ij->i", unit[left], unit[right])
    union = _Union(len(paths))
    for index in np.flatnonzero(similarity >= float(threshold)):
        union.union(int(left[index]), int(right[index]))
    grouped: dict[int, list[str]] = {}
    for row, path in enumerate(paths):
        if usable[row]:
            grouped.setdefault(union.find(row), []).append(path)
    groups = [members for members in grouped.values() if len(members) > 1]
    groups.sort(key=len, reverse=True)
    return groups
