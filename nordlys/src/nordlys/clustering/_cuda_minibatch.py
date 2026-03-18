"""Shared CUDA MiniBatchKMeans core matching sklearn semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cupy as cp


def greedy_kmeanspp_init(
    data: "cp.ndarray",
    n_clusters: int,
    random_state: np.random.RandomState,
    x_squared_norms: "cp.ndarray",
) -> "cp.ndarray":
    """Greedy k-means++ initialization matching sklearn behavior.

    sklearn uses n_init=1 by default (n_init='auto' resolves to 1 for
    MiniBatchKMeans when input is not a precomputed init), draws an
    init_size subset for initialization, and uses 2 + int(log(n_clusters))
    candidate trials per center selection.
    """
    import cupy as cp

    n_samples = data.shape[0]
    centers = []

    first_idx = random_state.randint(0, n_samples)
    centers.append(cp.asarray(data[first_idx]))

    n_trials = max(2, int(np.log(n_clusters))) if n_clusters > 1 else 1

    for _ in range(n_clusters - 1):
        distances = cp.full(n_samples, cp.inf, dtype=data.dtype)
        for center in centers:
            dist = cp.sum((data - center) ** 2, axis=1)
            distances = cp.minimum(distances, dist)

        best_dist = 0.0
        best_idx = 0
        for _ in range(n_trials):
            r = random_state.random()
            cumsum = cp.cumsum(distances / distances.sum())
            idx = int(cp.searchsorted(cumsum, cp.asarray([r]))[0])
            idx = min(idx, n_samples - 1)
            d = float(cp.sum((data[idx] - data[first_idx]) ** 2))
            if d > best_dist:
                best_dist = d
                best_idx = idx

        centers.append(cp.asarray(data[best_idx]))

    return cp.stack(centers)


def _fit_minibatch_core(
    data: "cp.ndarray",
    init_centers: "cp.ndarray",
    batch_size: int,
    max_iter: int,
    max_no_improvement: int,
    reassignment_ratio: float,
    random_state: np.random.RandomState,
) -> tuple["cp.ndarray", "cp.ndarray", float, int]:
    """Core mini-batch loop matching sklearn MiniBatchKMeans.

    Returns (centers, labels, inertia, n_steps).
    """
    import cupy as cp

    n_samples, n_features = data.shape
    n_clusters = init_centers.shape[0]

    centers = init_centers.copy()
    centers_new = cp.empty_like(centers)
    counts = cp.zeros(n_clusters, dtype=data.dtype)

    _batch_size = min(batch_size, n_samples)
    n_steps = (max_iter * n_samples) // max(_batch_size, 1)
    if n_steps == 0:
        n_steps = 1

    ewa_inertia = None
    ewa_inertia_min = None
    no_improvement = 0

    for step in range(n_steps):
        mb_indices = random_state.randint(0, n_samples, _batch_size)
        mb_data = data[mb_indices]

        sq_dists = cp.sum(
            (mb_data[:, cp.newaxis, :] - centers[cp.newaxis, :, :]) ** 2,
            axis=2,
        )
        batch_labels = cp.argmin(sq_dists, axis=1)
        batch_inertia = float(sq_dists[cp.arange(_batch_size), batch_labels].sum())

        _mini_batch_step(
            centers=centers,
            centers_new=centers_new,
            counts=counts,
            mb_data=mb_data,
            batch_labels=batch_labels,
            n_clusters=n_clusters,
            reassignment_ratio=reassignment_ratio,
            random_state=random_state,
        )

        centers, centers_new = centers_new, centers

        if step == 0:
            continue

        batch_inertia /= _batch_size

        alpha = _batch_size * 2.0 / (n_samples + 1)
        alpha = min(alpha, 1.0)
        if ewa_inertia is None:
            ewa_inertia = batch_inertia
        else:
            ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

        if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
            no_improvement = 0
            ewa_inertia_min = ewa_inertia
        else:
            no_improvement += 1

        if max_no_improvement is not None and no_improvement >= max_no_improvement:
            break

    final_sq_dists = cp.sum(
        (data[:, cp.newaxis, :] - centers[cp.newaxis, :, :]) ** 2,
        axis=2,
    )
    final_labels = cp.argmin(final_sq_dists, axis=1)
    inertia = float(final_sq_dists[cp.arange(n_samples), final_labels].sum())

    return centers, final_labels, inertia, step + 1


def _mini_batch_step(
    centers: "cp.ndarray",
    centers_new: "cp.ndarray",
    counts: "cp.ndarray",
    mb_data: "cp.ndarray",
    batch_labels: "cp.ndarray",
    n_clusters: int,
    reassignment_ratio: float,
    random_state: np.random.RandomState,
) -> None:
    """Per-batch update step matching sklearn."""
    import cupy as cp

    n_samples_batch = mb_data.shape[0]

    for k in range(n_clusters):
        mask = batch_labels == k
        batch_count = int(cp.count_nonzero(mask))
        if batch_count > 0:
            batch_sum = cp.sum(mb_data[mask], axis=0)
            new_count = counts[k] + batch_count
            centers_new[k] = (centers[k] * counts[k] + batch_sum) / new_count
            counts[k] = new_count
        else:
            centers_new[k] = centers[k]

    n_since_last = n_samples_batch
    do_reassign = bool(
        cp.count_nonzero(counts == 0) > 0 or n_since_last >= 10 * n_clusters
    )

    if do_reassign:
        for k in range(n_clusters):
            if counts[k] == 0 or (
                reassignment_ratio > 0
                and float(counts[k]) < reassignment_ratio * float(cp.max(counts))
            ):
                idx = random_state.randint(0, mb_data.shape[0])
                centers_new[k] = mb_data[idx]
                counts[k] = (
                    cp.min(counts[counts > 0])
                    if int(cp.count_nonzero(counts > 0)) > 0
                    else 1
                )
