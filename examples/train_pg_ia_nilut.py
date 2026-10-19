"""Train PG-IA-NILUT on a folder of (input, retouched target) pairs, then export a LUT.

    pip install "lutlab[torch,eval,clip]"
    python examples/train_pg_ia_nilut.py /data/fivek_expertC --out pg_ia_nilut.pt

Layout: ROOT/input/<name>.png and ROOT/target/<name>.png (FiveK / PPR10K exports).
Learning rates, step counts, batch size, lambda_j and lambda are PLACEHOLDERS (the paper does not
state them): see `TrainConfig`. Needs CLIP weights (downloaded by open_clip on first use).
"""
from __future__ import annotations

import argparse

import lutlab
from lutlab.evaluation import PairFolder, evaluate
from lutlab.methods import pg_ia_nilut as pg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="pg_ia_nilut.pt")
    ap.add_argument("--dataset", default="fivek", choices=["fivek", "ppr10k"])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    pairs = PairFolder(a.root)
    cfg = pg.TrainConfig(dataset=a.dataset,
                         predictor="cnn" if a.dataset == "fivek" else "resnet18")
    res = pg.train_pg_ia_nilut(pairs, cfg=cfg, device=a.device)      # builds a CLIP scorer
    res.model.save(a.out)

    t = res.model.transform()                                          # image-adaptive
    print(evaluate(t, pairs, metrics=("psnr", "ssim", "de76"), max_pairs=50,
                   timing={"warmup": 1, "repeat": 5}))
    # the interpretable part: freeze the filters at chosen controls and ship a plain .cube
    res.model.filters.lut(res.model.filters.weights(exposure=0.5), N=33).save("exposure_plus.cube")


if __name__ == "__main__":
    main()
