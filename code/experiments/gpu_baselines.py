"""GPU versions of the two baseline pieces that dominate the cost, with a validation gate.

WHY THIS EXISTS. ESCROW's engine cannot use a GPU. It is sequential bookkeeping over
Krichevsky-Trofimov counts held in dictionaries, branchy and integer-heavy, with no linear algebra
anywhere in it; a record costs about fifty microseconds and the repair pass is what dominates. The
BASELINES are the opposite. The matched representation is a dense matrix, one column per observed
(key, value) pair, which reaches 4,500 by 32,253 on the largest stream here, and the suite spends
nearly all of its time in two places on it: the two pairwise distance matrices, and a k-means fitted
once per candidate k for the silhouette selection. Both are matrix multiplication.

THE HONESTY PROBLEM, AND THE GATE. A GPU k-means is not the same program as scikit-learn's. It
differs in initialisation draws and in tie-breaking, so it can return different labels on the same
input, and quoting its numbers beside CPU numbers from earlier experiments would be comparing two
implementations while claiming to compare two methods. So nothing here is used until it agrees with
scikit-learn on a stream where both can be run:

    validate() runs both implementations on the same matrix and reports the agreement between their
    partitions and the difference in the score each earns against the labels. The caller uses the GPU
    path only if that difference is within TOL, and records the comparison either way.

The distance matrices carry no such risk. They are the same arithmetic in a different place, and
they are checked here to floating point tolerance rather than to a judgement call.
"""
from __future__ import annotations

import numpy as np

TOL = 0.01          # largest ARI difference against scikit-learn that still counts as agreement


def _torch():
    import torch
    return torch


def to_device(X, dtype=None):
    torch = _torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype or torch.float32
    return torch.as_tensor(np.ascontiguousarray(X), dtype=dtype, device=dev)


def pairwise_cosine(X):
    """Cosine distance matrix, the same quantity sklearn.metrics.pairwise_distances computes."""
    torch = _torch()
    T = to_device(X)
    Tn = T / T.norm(dim=1, keepdim=True).clamp_min(1e-12)
    D = 1.0 - (Tn @ Tn.T)
    D.fill_diagonal_(0.0)
    return D.clamp_min_(0.0)


def pairwise_euclidean(X):
    torch = _torch()
    T = to_device(X)
    D = torch.cdist(T, T, p=2)
    D.fill_diagonal_(0.0)
    return D.clamp_min_(0.0)


def kmeans(X, k, n_init=10, seed=0, iters=300, tol=1e-6):
    """k-means++ initialisation and Lloyd iterations, the same algorithm scikit-learn runs.

    Returns the labels of the run with the lowest inertia over n_init restarts, which is what
    scikit-learn's n_init does.
    """
    torch = _torch()
    T = to_device(X)
    n = T.shape[0]
    g = torch.Generator(device=T.device).manual_seed(int(seed))
    best_lab, best_inertia = None, None
    sq = (T * T).sum(1)

    for start in range(n_init):
        # k-means++ seeding
        idx = torch.randint(n, (1,), generator=g, device=T.device)
        C = T[idx]
        d2 = ((T - C[0]) ** 2).sum(1)
        for _ in range(1, k):
            probs = d2.clamp_min(0)
            s = probs.sum()
            if s <= 0:
                nxt = torch.randint(n, (1,), generator=g, device=T.device)
            else:
                nxt = torch.multinomial(probs / s, 1, generator=g)
            C = torch.cat([C, T[nxt]], 0)
            d2 = torch.minimum(d2, ((T - T[nxt][0]) ** 2).sum(1))

        prev = None
        for _ in range(iters):
            D = sq[:, None] - 2.0 * (T @ C.T) + (C * C).sum(1)[None, :]
            lab = D.argmin(1)
            newC = torch.zeros_like(C)
            cnt = torch.zeros(k, device=T.device, dtype=T.dtype)
            newC.index_add_(0, lab, T)
            cnt.index_add_(0, lab, torch.ones(n, device=T.device, dtype=T.dtype))
            empty = cnt == 0
            if empty.any():                       # restart an empty cluster on the worst-fit point
                far = D.gather(1, lab[:, None]).squeeze(1).argsort(descending=True)
                for j, e in enumerate(torch.nonzero(empty).flatten().tolist()):
                    newC[e] = T[far[j]]
                    cnt[e] = 1.0
            C = newC / cnt[:, None]
            inertia = D.gather(1, lab[:, None]).sum().item()
            if prev is not None and abs(prev - inertia) <= tol * max(1.0, abs(prev)):
                break
            prev = inertia
        if best_inertia is None or inertia < best_inertia:
            best_inertia, best_lab = inertia, lab.detach().cpu().numpy().copy()
    return best_lab, float(best_inertia)


def silhouette_from_D(D, labels):
    """Silhouette from a precomputed distance matrix, the same definition sklearn uses."""
    torch = _torch()
    lab = torch.as_tensor(np.asarray(labels), device=D.device)
    uniq = torch.unique(lab)
    if uniq.numel() < 2:
        return 0.0
    onehot = (lab[:, None] == uniq[None, :]).to(D.dtype)
    sums = D @ onehot                                    # distance sums to each cluster
    counts = onehot.sum(0)
    own = onehot.argmax(1)
    a_den = (counts[own] - 1.0).clamp_min(1.0)
    a = sums.gather(1, own[:, None]).squeeze(1) / a_den
    other = sums / counts[None, :].clamp_min(1.0)
    other.scatter_(1, own[:, None], float("inf"))
    b = other.min(1).values
    s = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
    s[counts[own] <= 1] = 0.0
    return float(s.mean().item())


def validate(X, truth, ks, ari_fn):
    """Run both implementations and report whether the GPU path may be used.

    Nothing about the paper's numbers depends on this returning True. If it returns False the caller
    runs the scikit-learn path and says so.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    rows = []
    Dc_gpu = pairwise_cosine(X).cpu().numpy()
    Dc_cpu = pairwise_distances(X, metric="cosine")
    dist_err = float(np.abs(Dc_gpu - Dc_cpu).max())

    for k in ks:
        cpu = KMeans(k, n_init=4, random_state=0).fit_predict(X)
        gpu, _ = kmeans(X, k, n_init=4, seed=0)
        rows.append({"k": int(k),
                     "ARI_cpu_vs_truth": round(ari_fn(truth, list(cpu)), 4),
                     "ARI_gpu_vs_truth": round(ari_fn(truth, list(gpu)), 4),
                     "agreement_between_the_two": round(ari_fn(list(cpu), list(gpu)), 4)})
    worst = max(abs(r["ARI_cpu_vs_truth"] - r["ARI_gpu_vs_truth"]) for r in rows)
    return {"max_abs_distance_error": dist_err,
            "per_k": rows,
            "worst_ARI_difference": round(worst, 4),
            "tolerance": TOL,
            "gpu_usable": bool(worst <= TOL and dist_err < 1e-4)}
