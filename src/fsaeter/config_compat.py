"""Compatibility helpers between older internal configs and fSAEter configs."""

from __future__ import annotations

from copy import deepcopy


def _merged(base: dict, extra: dict) -> dict:
    result = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merged(result[key], value)
        else:
            result[key] = value
    return result


def normalize_extract_config(config: dict) -> dict:
    cfg = deepcopy(config)
    run = dict(cfg.get("run") or {})
    data = dict(cfg.get("data") or {})
    encoder = dict(cfg.get("encoder") or {})
    tokens = dict(cfg.get("tokens") or {})
    extraction = dict(cfg.get("extraction") or {})
    storage = dict(cfg.get("storage") or {})

    tokens = _merged(
        tokens,
        {
            k: v
            for k, v in extraction.items()
            if k in {"device", "precision", "batch_size", "num_workers"}
        },
    )
    if "save_dtype" not in tokens and "save_dtype" in storage:
        tokens["save_dtype"] = storage["save_dtype"]

    return {
        "run": run,
        "data": data,
        "encoder": encoder,
        "tokens": tokens,
    }


def normalize_train_config(config: dict) -> dict:
    cfg = deepcopy(config)
    run = dict(cfg.get("run") or {})
    tokens = dict(cfg.get("tokens") or {})
    sae = dict(cfg.get("sae") or {})
    train = dict(cfg.get("train") or {})

    legacy_training = dict(cfg.get("training") or {})
    legacy_matry = dict(cfg.get("matryoshka") or {})
    if legacy_training:
        for key in ("variant", "d_model", "d_sae", "target_k", "decoder_row_norm"):
            if key in legacy_training and key not in sae:
                sae[key] = legacy_training[key]
        for key in (
            "device",
            "precision",
            "batch_size",
            "epochs",
            "lr",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip_norm",
            "val_fraction",
            "split_seed",
            "num_workers",
            "log_every",
            "backend",
            "max_train_rows",
            "max_val_rows",
            "compile",
        ):
            if key in legacy_training and key not in train:
                train[key] = legacy_training[key]

    if "prefixes" in legacy_matry and "matryoshka_prefixes" not in sae:
        sae["matryoshka_prefixes"] = legacy_matry["prefixes"]
    if "weights" in legacy_matry and "matryoshka_weights" not in sae:
        sae["matryoshka_weights"] = legacy_matry["weights"]

    return {
        "run": run,
        "tokens": tokens,
        "sae": sae,
        "train": train,
    }


def normalize_build_h_config(config: dict) -> dict:
    cfg = deepcopy(config)
    run = dict(cfg.get("run") or {})
    tokens = dict(cfg.get("tokens") or {})
    sae = dict(cfg.get("sae") or {})
    build_h = dict(cfg.get("build_h") or {})
    inspect = dict(cfg.get("inspect") or {})

    inference = dict(cfg.get("inference") or {})
    pooling = dict(cfg.get("pooling") or {})
    qc = dict(cfg.get("qc") or {})

    for key in ("device", "precision", "image_batch_size", "token_batch_size", "max_images"):
        if key in inference and key not in build_h:
            build_h[key] = inference[key]
    for key in ("image_top_k", "active_threshold", "save_max", "save_dtype"):
        if key in pooling and key not in build_h:
            build_h[key] = pooling[key]
    for key, value in qc.items():
        if key not in inspect:
            inspect[key] = value

    return {
        "run": run,
        "tokens": tokens,
        "sae": sae,
        "build_h": build_h,
        "inspect": inspect,
    }
