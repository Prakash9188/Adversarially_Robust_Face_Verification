"""LFW (funneled) loading and a subject-disjoint, class-balanced pair protocol.

Why our own pairs? The official 10-fold pairs.txt mixes identities across folds, so
a 70/15/15 *identity-disjoint* split would leave only a few dozen impostor pairs in
the test set. Instead we split IDENTITIES 70/15/15 and then sample balanced
genuine/impostor pairs inside each split (fixed seed -> reproducible).
"""
import json
import random
import tarfile
import urllib.request
from pathlib import Path

import torch
from PIL import Image

# Original host: http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (did not resolve on our
# machine). Same archive, figshare mirror (also used by scikit-learn):
LFW_URL = "https://ndownloader.figshare.com/files/5976015"
SPLITS = ("train", "val", "test")


def download_lfw(root="data"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "lfw_funneled").exists():
        return root
    tgz = root / "lfw-funneled.tgz"
    if not tgz.exists():
        print(f"Downloading {LFW_URL} ...")
        urllib.request.urlretrieve(LFW_URL, tgz)
    with tarfile.open(tgz) as t:
        t.extractall(root, filter="data")
    return root


def scan_lfw(root):
    base = Path(root) / "lfw_funneled"
    people = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        files = sorted(str(f.relative_to(root)).replace("\\", "/") for f in d.glob("*.jpg"))
        if files:
            people[d.name] = files
    return people


def make_pairs(root, fractions, counts, seed):
    """Return {split: [(path1, path2, label, name1, name2), ...]} (shuffled)."""
    rng = random.Random(seed)
    people = scan_lfw(root)
    names = sorted(people)
    rng.shuffle(names)
    n = len(names)
    c1, c2 = int(fractions[0] * n), int((fractions[0] + fractions[1]) * n)
    groups = {"train": names[:c1], "val": names[c1:c2], "test": names[c2:]}
    out = {}
    for split, ids in groups.items():
        multi = [i for i in ids if len(people[i]) >= 2]
        k = counts[split] // 2
        gen, imp, seen = [], [], set()
        tries = 0
        while len(gen) < k and tries < 100 * k:
            tries += 1
            nm = rng.choice(multi)
            a, b = rng.sample(people[nm], 2)
            if (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            gen.append((a, b, 1, nm, nm))
        while len(imp) < k:
            n1, n2 = rng.sample(ids, 2)
            a, b = rng.choice(people[n1]), rng.choice(people[n2])
            if (a, b) in seen:
                continue
            seen.add((a, b))
            imp.append((a, b, 0, n1, n2))
        pairs = gen + imp
        rng.shuffle(pairs)
        out[split] = pairs
    # sanity: identities never cross splits
    sets = {s: {p[3] for p in v} | {p[4] for p in v} for s, v in out.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])
    return out


def load_images(root, paths, crop, size):
    """Center-crop `crop` px of the 250x250 image, resize to `size`, uint8 (N,3,S,S)."""
    imgs = []
    for p in paths:
        im = Image.open(Path(root) / p).convert("RGB")
        w, h = im.size
        l, t = (w - crop) // 2, (h - crop) // 2
        im = im.crop((l, t, l + crop, t + crop)).resize((size, size), Image.BILINEAR)
        imgs.append(torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1))
    return torch.stack(imgs)


def get_split(cfg, split):
    """-> (x1, x2, labels): uint8 tensors (N,3,S,S), (N,3,S,S), long (N,). Cached."""
    d = cfg["data"]
    root = Path(d["root"])
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"s{cfg['seed']}_c{d['crop']}_z{d['size']}_n{d['pairs'][split]}"
    f = cache / f"{split}_{tag}.pt"
    if f.exists():
        return torch.load(f)
    pairs_f = cache / f"pairs_s{cfg['seed']}_{'-'.join(str(d['pairs'][s]) for s in SPLITS)}.json"
    if pairs_f.exists():
        allp = json.loads(pairs_f.read_text())
    else:
        allp = make_pairs(root, d["split"], d["pairs"], cfg["seed"])
        pairs_f.write_text(json.dumps(allp))
    p = allp[split]
    x1 = load_images(root, [q[0] for q in p], d["crop"], d["size"])
    x2 = load_images(root, [q[1] for q in p], d["crop"], d["size"])
    y = torch.tensor([q[2] for q in p])
    torch.save((x1, x2, y), f)
    return x1, x2, y


def to_float(x):
    return x.float() / 255.0
