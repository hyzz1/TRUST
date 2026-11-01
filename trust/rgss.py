"""RGSS: inference-only reliability and exact-budget source retention."""
import numpy as np

def collapse_queries(query_logits, query_idx, classes):
    logits = np.asarray(query_logits, dtype=np.float32)
    groups = np.asarray(query_idx, dtype=np.int64)
    if logits.ndim != 2 or groups.shape != (logits.shape[1],) or classes < 2:
        raise ValueError('Expected N x Q logits and Q query-to-class indices, C >= 2')
    if not np.isfinite(logits).all() or set(groups.tolist()) != set(range(classes)):
        raise ValueError('Nonfinite logits or incomplete class mapping')
    # Same alias merge as the original model, before softmax/background rejection.
    return np.stack([logits[:, groups == c].max(1) for c in range(classes)], axis=1)

def graph_arrays(indices, weights, n):
    indices = np.asarray(indices)
    weights = np.asarray(weights, dtype=np.float32)
    if indices.dtype.kind not in 'iu' or indices.ndim != 2 or weights.shape != indices.shape:
        raise ValueError('Expected integer N x K indices and matching weights')
    if indices.shape[0] != n or indices.shape[1] < 1 or n < 1:
        raise ValueError('Invalid graph shape')
    if indices.min() < 0 or indices.max() >= n or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Invalid graph edge or weight')
    own = indices == np.arange(n)[:, None]
    if not own[:, 0].all() or not (own.sum(1) == 1).all():
        raise ValueError('Expected exactly one self edge in column zero')
    if not np.allclose(weights.sum(1), 1, rtol=1e-5, atol=1e-5):
        raise ValueError('Expected row-normalized graph weights')
    return indices.astype(np.int64, copy=False), weights, own

def source_scores(class_logits, indices, weights):
    values = np.asarray(class_logits, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError('Expected finite N x C class logits')
    idx, w, own = graph_arrays(indices, weights, len(values))
    predicted = values.argmax(1)
    best_two = np.partition(values, -2, axis=1)[:, -2:]
    margin = best_two[:, 1] - best_two[:, 0]
    # One-hop agreement uses raw predictions/margins only, never refined outputs.
    evidence = w.astype(np.float64) * (~own) * margin[idx]
    total = evidence.sum(1)
    agreeing = (evidence * (predicted[idx] == predicted[:, None])).sum(1)
    # With no informative neighbour, keep the own margin rather than force it to zero.
    agreement = np.divide(agreeing, total, out=np.ones(len(values)), where=total > 0)
    return dict(predicted=predicted, margin=margin,
                agreement=agreement.astype(np.float32),
                margin_agreement=(margin * agreement).astype(np.float32))

def select_sources(indices, weights, reliability, keep_fraction, seed):
    score = np.asarray(reliability, dtype=np.float32)
    if score.ndim != 1 or not np.isfinite(score).all() or not 0 < keep_fraction <= 1:
        raise ValueError('Expected finite source scores and 0 < keep_fraction <= 1')
    idx, w, own = graph_arrays(indices, weights, len(score))
    eligible = (~own) & (w > 0)
    count = np.ceil(eligible.sum(1) * keep_fraction).astype(np.int64)
    w64 = w.astype(np.float64)
    target_mass = w64.sum(1)
    if keep_fraction == 1:
        retained = eligible
        updated = w.copy()
        updated64 = w64
    else:
        # Seeded tie-breaking for equal source reliability.
        noise = np.random.default_rng(seed).random(w.shape)
        priority = np.where(eligible, score[idx], -np.inf)
        order = np.lexsort((noise, -priority), axis=1)
        retained = np.zeros_like(eligible)
        np.put_along_axis(retained, order, np.arange(w.shape[1])[None, :] < count[:, None], axis=1)
        retained &= eligible
        donor = w64 * retained
        remaining = donor.sum(1, keepdims=True)
        original = (w64 * eligible).sum(1, keepdims=True)
        factor = np.divide(original, remaining, out=np.zeros_like(original), where=remaining > 0)
        updated64 = donor * factor + w64 * own
        updated = updated64.astype(np.float32)
        updated[count == 0] = w[count == 0]
    if not np.array_equal(retained.sum(1), count):
        raise AssertionError('Retention count mismatch')
    if not np.array_equal(updated[own], w[own]):
        raise AssertionError('Self weight changed')
    # Check conservation before and after the required float32 cast separately.
    # Casting each edge back to float32 can move a 64-way row sum by several
    # ulps, so comparing float32 reductions with a 2e-7 tolerance spuriously
    # aborts otherwise valid graphs on some images.
    if keep_fraction != 1 and not np.allclose(
            updated64.sum(1), target_mass, atol=1e-12, rtol=1e-12):
        raise AssertionError('Total propagation mass changed')
    observed_mass = updated.astype(np.float64).sum(1)
    cast_bound = (np.finfo(np.float32).eps * 0.5 *
                  np.abs(updated64).sum(1) + 4 * np.finfo(np.float64).eps)
    cast_error = np.abs(observed_mass - target_mass)
    if (cast_error > cast_bound).any():
        raise AssertionError('Float32 conversion changed propagation mass beyond rounding error')
    details = dict(retained_nonself_edges=int(retained.sum()),
                   original_nonself_edges=int(eligible.sum()),
                   isolated_queries=int((count == 0).sum()),
                   row_mass_max_error=float(cast_error.max()),
                   keep_fraction=float(keep_fraction))
    return updated, retained, details
