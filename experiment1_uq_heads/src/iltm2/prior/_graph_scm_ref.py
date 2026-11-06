"""Unmodified graph_scm tasks, exactly as TabICLv2 was pretrained on: the reference prior the observed cut is
compared against (scripts/00f_difficulty_match.py, scripts/01b_check_val_bank.py; design Training §1.4).

No replicates exist on these tasks (CLAUDE.md, "the abduction rule"): they are used only for comparisons of the
backbone's forecast and of task difficulty, never for targets.
"""
import torch

from tabicl.prior._dataset import Prior, should_filter
from tabicl.prior._graph_scm import GraphSCM

from iltm2.prior._cut_graph_scm import v2_prior_config

PRETRAIN = v2_prior_config(ensure_iid=False)   # unmodified graph_scm, exactly as TabICLv2 was trained


def graph_scm_task(shape, log):
    """tabicl's `GraphPrior.generate_dataset` loop for one regression dataset, with rejections counted.

    Same steps: GraphSCM -> delete_unique_features -> should_filter, retry on failure. tabicl pads X to
    max_features; here max_features = d (no padding), as for the cut tasks, so both see the same columns.
    GraphSCM's NaN guard replaces a dataset with a NaN in X or y by X = 0, y = -100 (tabicl then rejects it as
    constant); it is detected here and logged as "non_finite", so the two priors' non-finite rates are comparable.
    `shape` needs N, n_ctx and d. Draws from the global RNGs: call it under `task_rng`.
    """
    params = dict(regression=True, seq_len=shape.N, train_size=shape.n_ctx, num_features=shape.d,
                  max_features=shape.d, config=PRETRAIN, device="cpu")
    while True:
        X, y = GraphSCM(**params)()
        if bool(torch.all(y == -100.0)):                    # GraphSCM's NaN guard fired
            log.reject("non_finite")
            continue
        X, d = Prior.delete_unique_features(X[None], torch.tensor([shape.d]))
        if not (d > 0).all():
            log.reject("constant_columns")
            continue
        if should_filter(X[0], y, PRETRAIN, is_classif=False):
            log.reject("unpredictable")
            continue
        log.accepted += 1
        return X[0, :, :int(d[0])], y
