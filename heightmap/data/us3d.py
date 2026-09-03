"""US3D / DFC2019 as a drop-in source alongside GAMUS.

Why this dataset is here at all: it is the only public source that pairs GAMUS's exact
triple (RGB + semantic CLS + AGL) with a genuine SENSOR change and clean LiDAR truth.
GAMUS's three cities are one aerial orthophoto programme, so DC/PHL/NYC differ in urban
fabric but not in optics; India differs in both, and US3D's WorldView-3 imagery is the
only place we can rehearse the optical half with ground truth good enough to check
ourselves against.

Differences from GAMUS that this module absorbs, so the rest of the pipeline sees one
uniform sample format:

  format      GeoTIFF, not HDF5
  naming      JAX_163_010_RGB.tif  (site _ tile _ view), flat directories, no split dirs
  GSD         ~0.30 m against GAMUS's 0.33 m -- close enough to share the FiLM band
  classes     LAS codes, NOT GAMUS ids.  This is the trap; see CLS_LAS_TO_GAMUS.
  DSM         Track 3 only, and on the Track 1 tile ids, so DTM = DSM - AGL is derivable

The output dict is byte-for-byte the same shape GamusDataset returns, so MixedDomains
and the training loop need no special case.
"""
from __future__ import annotations
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import transforms as T

#: US3D ships semantic labels as LAS classification codes; GAMUS uses its own 0-6 ids
#: (heightmap/data/classes.py is the authority).  Feeding LAS codes straight into a head
#: trained on GAMUS ids would silently score "building" against class 6 -- which in
#: GAMUS is TREE -- and the semantic loss would fight the height loss for the whole run
#: without ever raising anything.  Bug #6 in this project was exactly this class of
#: error (building was 2, should have been 3) and it went unnoticed through a full
#: ablation.
#:
#: LAS standard:  2 ground, 3 low veg, 4 med veg, 5 high veg, 6 building, 9 water,
#:                17 bridge deck.  DFC2019 uses the subset {2, 5, 6, 9, 17}.
#: GAMUS ids:     0 unlabelled, 1 ground, 2 low-veg, 3 building, 4 water, 5 road, 6 tree
#:
#: VERIFY THIS AGAINST THE REAL FILES before trusting any semantic number: run
#: scripts/inspect_us3d.py, which prints the actual unique CLS values found on disk and
#: refuses to guess.  Bridge deck -> road is the one judgement call here; DFC2019 has no
#: separate road class, and a bridge deck is the closest GAMUS analogue.
CLS_LAS_TO_GAMUS = {2: 1, 3: 2, 4: 2, 5: 6, 6: 3, 9: 4, 17: 5}

US3D_GSD = 0.30                      # metres/pixel, WorldView-3 pan
TILE_RE = re.compile(r"^(?P<site>[A-Z]{3})_(?P<tile>\d+)_(?P<view>\d+)_(?P<kind>[A-Z]+)\.tif$",
                     re.IGNORECASE)


def index_us3d(root: str | Path, require=("RGB", "AGL", "CLS")) -> list[tuple[str, str]]:
    """-> [(site, stem)] for every tile that has ALL the required products.

    A tile missing any one product is skipped rather than half-loaded: US3D ships its
    RGB, reference and DSM as separate archives, so a partial download is the normal
    failure mode here, and a dataset that silently shrinks is how you end up reporting a
    number computed on a third of the data you thought you had.
    """
    root = Path(root)
    found: dict[str, set[str]] = {}
    sites: dict[str, str] = {}
    for p in root.rglob("*.tif"):
        m = TILE_RE.match(p.name)
        if not m:
            continue
        stem = f"{m['site']}_{m['tile']}_{m['view']}"
        found.setdefault(stem, set()).add(m["kind"].upper())
        sites[stem] = m["site"].upper()
    out = [(sites[s], s) for s, kinds in found.items()
           if all(k in kinds for k in require)]
    return sorted(out)


def _find(root: Path, stem: str, kind: str) -> Path | None:
    hits = list(root.rglob(f"{stem}_{kind}.tif"))
    return hits[0] if hits else None


def read_tif(path: Path) -> np.ndarray:
    """Read a GeoTIFF with rasterio, falling back to PIL.

    rasterio first because US3D's AGL and DSM are float32 with a nodata tag that PIL
    does not surface, and losing the nodata tag turns a sentinel into a height.
    """
    try:
        import rasterio
        with rasterio.open(path) as ds:
            a = ds.read()
            nod = ds.nodata
        a = a[0] if a.shape[0] == 1 else np.transpose(a, (1, 2, 0))
        if nod is not None and np.issubdtype(a.dtype, np.floating):
            a = np.where(a == nod, np.nan, a)
        return a
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path))


class US3DDataset(Dataset):
    """One US3D tile -> the same sample dict GamusDataset returns.

    `split` is accepted and ignored: US3D has no split directories, and the argument
    exists only so this class is interchangeable with GamusDataset inside MixedDomains.
    """

    MAX_OPEN = 0      # GeoTIFFs are opened and closed per read; no handle cache needed

    def __init__(self, cfg, split: str, train: bool, tiles=None, root: str | None = None):
        self.cfg, self.split, self.train = cfg, split, train
        self.root = Path(root or getattr(cfg.data, "us3d_root", "data/US3D"))
        self.tiles = list(tiles) if tiles is not None else index_us3d(self.root)
        if not self.tiles:
            raise FileNotFoundError(
                f"no complete US3D tiles under {self.root} (need RGB+AGL+CLS per tile). "
                f"Track 1 'RGB images' and Track 1 'Reference' are both required.")
        self._cache: OrderedDict = OrderedDict()

    def __len__(self):
        return len(self.tiles)

    def _remap_cls(self, cls: np.ndarray) -> np.ndarray:
        """LAS codes -> GAMUS ids.  Anything unrecognised becomes the ignore index.

        Deliberately a lookup rather than arithmetic: an unknown code must land on
        'ignore', never on a valid class it happens to be numerically near.
        """
        out = np.zeros(cls.shape, np.int64)
        for las, gam in CLS_LAS_TO_GAMUS.items():
            out[cls == las] = gam
        ign = int(self.cfg.model.ignore_index)
        n_cls = int(self.cfg.model.n_classes)
        bad = (out < 0) | (out >= n_cls)
        if bad.any():
            out[bad] = ign if ign >= 0 else 0
        return out

    def __getitem__(self, i: int):
        site, stem = self.tiles[i]
        rgb = read_tif(_find(self.root, stem, "RGB"))
        agl = read_tif(_find(self.root, stem, "AGL")).astype(np.float32)
        cls = read_tif(_find(self.root, stem, "CLS"))

        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, -1)
        rgb = rgb[..., :3].astype(np.uint8)

        mask = np.isfinite(agl) & (agl > -100.0)
        agl = np.clip(np.nan_to_num(agl, nan=0.0, posinf=0.0, neginf=0.0),
                      0.0, float(self.cfg.data.h_max))
        cls = self._remap_cls(np.asarray(cls).astype(np.int64))

        # US3D's real GSD, not GAMUS's.  The FiLM head is conditioned on log(GSD), so
        # handing it 0.33 for a 0.30 m tile is a small lie that the conditioning is
        # specifically built to notice -- and the point of this dataset is the sensor
        # change, so the one number describing the sensor must be right.
        s = T.Sample(rgb, agl, cls, mask, float(getattr(self.cfg.data, "us3d_gsd", US3D_GSD)))
        rng = np.random.default_rng(None if self.train else (hash(stem) & 0xFFFFFFFF))

        if self.train:
            s = T.scale_and_crop(s, int(self.cfg.data.crop), float(self.cfg.data.scale_min),
                                 float(self.cfg.data.scale_max), rng)
            if self.cfg.aug.geometric:
                s = T.geometric(s, rng)
            img = T.photometric(s.image, rng, self.cfg.aug) if self.cfg.aug.photometric else s.image
        else:
            s = T.center_crop(s, int(self.cfg.data.crop))
            img = s.image

        x = T.normalise(img).transpose(2, 0, 1)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x)),
            "height": torch.from_numpy(s.height).unsqueeze(0),
            "cls": torch.from_numpy(s.cls),
            "mask": torch.from_numpy(s.mask).unsqueeze(0),
            "gsd": torch.tensor(s.gsd, dtype=torch.float32),
            # No sun-angle estimates for US3D yet.  NaN rather than a plausible default:
            # the shadow loss skips non-finite entries, and a fabricated 40 deg would be
            # silently consumed as if it were measured.
            "sun": torch.tensor([float("nan"), float("nan")], dtype=torch.float32),
            "tile_id": stem,
            "city": site,
        }
