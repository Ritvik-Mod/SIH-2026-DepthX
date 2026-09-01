"""GAMUS dataset.

Layout (as published at earthflow/GAMUS on HuggingFace):
    images/{split}/{CITY}_{r}_{c}_RGB.h5    (1024,1024,3) uint8
    heights/{split}/{CITY}_{r}_{c}_AGL.h5   (1024,1024)   float32  metres above ground
    classes/{split}/{CITY}_{r}_{c}_CLS.h5   (1024,1024)   float32  class ids

Each file holds a single dataset under the key 'image'.

Two split modes:
  official     — the published train/val/test.  NYC appears in both train and test,
                 so this measures within-city performance.  Use for comparability.
  city_holdout — train on every city except holdout_cities, evaluate on those.
                 This is the number that predicts performance on unseen landscapes.
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from . import transforms as T

# The release is not internally consistent: DC and PHL name their images *_RGB.h5
# while NYC uses *_IMG.h5.  Heights (*_AGL) and classes (*_CLS) agree everywhere.
# Globbing only _RGB silently drops all 1,167 NYC tiles -- and with them the entire
# held-out city, which is the split the generalisation number depends on.
IMAGE_KINDS = ("RGB", "IMG")
TILE_RE = re.compile(r"^([A-Za-z]+)_(.+?)_(RGB|IMG|AGL|CLS)\.h5$")
H5_KEY = "image"


def parse_tile(fname: str):
    m = TILE_RE.match(Path(fname).name)
    if not m:
        return None
    city, rest, _kind = m.groups()
    return city, f"{city}_{rest}"


def index_tiles(root: str | Path, split: str) -> list[tuple[str, str]]:
    """-> [(city, tile_id)] present in all three directories for this split."""
    root = Path(root)
    got = {}
    for key, sub, kinds in (("img", "images", IMAGE_KINDS),
                            ("agl", "heights", ("AGL",)),
                            ("cls", "classes", ("CLS",))):
        d = root / sub / split
        ids = set()
        if d.is_dir():
            for kind in kinds:
                for f in d.glob(f"*_{kind}.h5"):
                    p = parse_tile(f.name)
                    if p:
                        ids.add(p)
        got[key] = ids
    common = got["img"] & got["agl"] & got["cls"]
    return sorted(common, key=lambda t: t[1])


class GamusDataset(Dataset):
    def __init__(self, cfg, split: str, train: bool, tiles=None):
        self.cfg, self.split, self.train = cfg, split, train
        self.root = Path(cfg.data.root)
        self.tiles = list(tiles) if tiles is not None else index_tiles(self.root, split)
        if not self.tiles:
            raise FileNotFoundError(
                f"no GAMUS tiles under {self.root}/{{images,heights,classes}}/{split}. "
                f"Run scripts/download_gamus.py, or scripts/make_synthetic.py for a smoke test."
            )
        self.sun = _load_sun_cache(cfg.data.sun_cache)
        self._h = None  # h5py handles are opened lazily per worker: see __getstate__

    def __getstate__(self):
        """h5py handles cannot be pickled and do not survive fork.  DataLoader workers
        are created by pickling (spawn) or forking (linux) this object, so the handle
        cache must never travel with it -- each worker reopens its own."""
        st = self.__dict__.copy()
        st["_h"] = None
        return st

    def __len__(self):
        return len(self.tiles)

    def _open(self, sub: str, tile_id: str, kinds):
        """kinds may hold several candidate suffixes (see IMAGE_KINDS)."""
        import h5py
        if self._h is None:
            self._h = {}
        key = (sub, tile_id)
        if key not in self._h:
            base = self.root / sub / self.split
            for kind in ((kinds,) if isinstance(kinds, str) else kinds):
                f = base / f"{tile_id}_{kind}.h5"
                if f.exists():
                    self._h[key] = h5py.File(f, "r")
                    break
            else:
                raise FileNotFoundError(
                    f"{tile_id}: none of {kinds} found under {base}")
        return self._h[key]

    @staticmethod
    def _array(h):
        """Read the tile, tolerating a dataset key other than 'image'."""
        import numpy as _np
        if H5_KEY in h:
            return _np.asarray(h[H5_KEY][()])
        keys = list(h.keys())
        if len(keys) != 1:
            raise KeyError(f"expected one dataset, found {keys}")
        return _np.asarray(h[keys[0]][()])

    def __getitem__(self, i: int):
        city, tid = self.tiles[i]
        rgb = self._array(self._open("images", tid, IMAGE_KINDS))
        agl = self._array(self._open("heights", tid, "AGL")).astype(np.float32)
        cls = self._array(self._open("classes", tid, "CLS"))

        mask = np.isfinite(agl) & (agl > -100.0)
        agl = np.clip(np.nan_to_num(agl, nan=0.0, posinf=0.0, neginf=0.0),
                      0.0, float(self.cfg.data.h_max))

        # NYC labels carry a 255 nodata value that DC and PHL do not, and 255 is far
        # outside the 0..6 range the semantic head predicts.  cross_entropy would raise
        # a device-side assert and kill the job -- but only on the batches that happen
        # to contain such a tile, so it survives a short smoke test.  Fold anything out
        # of range into the ignore index.
        cls = cls.astype(np.int64)
        n_cls = int(self.cfg.model.n_classes)
        ign = int(self.cfg.model.ignore_index)
        bad = (cls < 0) | (cls >= n_cls)
        if bad.any():
            cls[bad] = ign if ign >= 0 else 0

        s = T.Sample(rgb, agl, cls, mask, float(self.cfg.data.gsd_base))
        rng = np.random.default_rng(None if self.train else (hash(tid) & 0xFFFFFFFF))

        if self.train:
            s = T.scale_and_crop(s, int(self.cfg.data.crop),
                                 float(self.cfg.data.scale_min), float(self.cfg.data.scale_max), rng)
            if self.cfg.aug.geometric:
                s = T.geometric(s, rng)
            img = T.photometric(s.image, rng, self.cfg.aug) if self.cfg.aug.photometric else s.image
        else:
            s = T.center_crop(s, int(self.cfg.data.crop))
            img = s.image

        x = T.normalise(img).transpose(2, 0, 1)
        elev, azim = self.sun.get(tid, (float("nan"), float("nan")))
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "height": torch.from_numpy(s.height).unsqueeze(0),
            "cls": torch.from_numpy(s.cls),
            "mask": torch.from_numpy(s.mask).unsqueeze(0),
            "gsd": torch.tensor(s.gsd, dtype=torch.float32),
            "sun": torch.tensor([elev, azim], dtype=torch.float32),
            "tile_id": tid,
            "city": city,
        }


def _load_sun_cache(path):
    if not path:
        return {}
    import json
    with open(path) as f:
        return {k: (float(v[0]), float(v[1])) for k, v in json.load(f).items()}


def build_splits(cfg):
    """-> (train_tiles_by_split, val_tiles_by_split) as {split: [(city,tile_id)]}."""
    official = {s: index_tiles(cfg.data.root, s) for s in ("train", "val", "test")}
    if cfg.data.split_mode == "official":
        return {"train": official["train"]}, {"val": official["val"]}, {"test": official["test"]}

    hold = {c.upper() for c in cfg.data.holdout_cities}
    tr, ev = {}, {}
    for split, tiles in official.items():
        keep = [t for t in tiles if t[0].upper() not in hold]
        drop = [t for t in tiles if t[0].upper() in hold]
        if keep:
            tr[split] = keep
        if drop:
            ev[split] = drop
    if not tr:
        raise ValueError(f"city_holdout removed every tile; holdout={hold}")
    if not ev:
        raise ValueError(f"no tiles for holdout cities {hold}")
    return tr, {"holdout_val": [t for s in ("val", "train") for t in ev.get(s, [])][:400]}, \
        {"holdout_test": [t for t in ev.get("test", [])] or [t for s in ev for t in ev[s]]}


class ConcatSplits(torch.utils.data.ConcatDataset):
    """Wraps several GamusDataset (one per underlying split directory)."""

    def __init__(self, cfg, tiles_by_split: dict, train: bool):
        parts = [GamusDataset(cfg, split, train, tiles) for split, tiles in tiles_by_split.items() if tiles]
        super().__init__(parts)
