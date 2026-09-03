"""Fixed-ratio batch mixing across domains.

The problem this exists to solve.  Fine-tuning on a new domain -- a new city, a new
sensor, India -- means training on a target set that is far SMALLER than GAMUS.  Left to
a normal shuffled DataLoader, the ratio between domains is whatever the relative dataset
sizes happen to be, so a few hundred Indian tiles against 5,004 GAMUS tiles would appear
in roughly 1 batch in 10 and contribute almost nothing.  Oversample instead and the
model forgets GAMUS: the held-out-NYC run already showed a -2.02 m offset appearing on a
city in the SAME country with the SAME sensor, and that is the mild version.

So the mixing ratio has to be an explicit, swept hyperparameter, not an accident of
directory sizes.  `MixedDomainBatchSampler` makes every batch contain an exact,
specified number of samples from each domain.

Exact rather than in-expectation (a WeightedRandomSampler would give the latter) because
at batch_size 8 a 25% target is 2 samples, and Poisson noise around 2 means some batches
carry 0 -- and a batch with no target-domain samples contributes a pure GAMUS gradient
step.  Over a short fine-tune those add up to a measurably different run, which would
show up as seed noise and be indistinguishable from it.

Epoch length is driven by a nominated ANCHOR domain (default: the largest), so "one
epoch" keeps meaning "one pass over the big dataset" and the small domain is repeated
within it.  That keeps epoch counts comparable with every run in the A-ladder.
"""
from __future__ import annotations
import math
from typing import Iterator

import torch
from torch.utils.data import ConcatDataset, Sampler

from .gamus import GamusDataset


class MixedDomains(ConcatDataset):
    """Several domains, each itself several splits, concatenated with domain bookkeeping.

    tiles_by_domain: {"gamus": {"train": [(city, id), ...]}, "us3d": {...}, ...}

    Keys of the inner dict MUST be real split directory names -- GamusDataset._open
    reads root/images/<split>/, so a synthetic key resolves to a directory that does not
    exist.  That was bug #18 and it cost a full training run before it surfaced.
    """

    def __init__(self, cfg, tiles_by_domain: dict, train: bool, dataset_cls=GamusDataset):
        parts, self.domain_ranges, self.domains = [], {}, []
        start = 0
        for dom, by_split in tiles_by_domain.items():
            n_dom = 0
            for split, tiles in by_split.items():
                if not tiles:
                    continue
                parts.append(dataset_cls(cfg, split, train, tiles))
                n_dom += len(tiles)
            if n_dom == 0:
                raise ValueError(f"domain '{dom}' contributed no tiles")
            self.domain_ranges[dom] = (start, start + n_dom)
            self.domains.append(dom)
            start += n_dom
        if not parts:
            raise ValueError("no domains with tiles")
        super().__init__(parts)

    def indices_of(self, domain: str) -> list[int]:
        a, b = self.domain_ranges[domain]
        return list(range(a, b))

    def sizes(self) -> dict:
        return {d: b - a for d, (a, b) in self.domain_ranges.items()}


class MixedDomainBatchSampler(Sampler[list[int]]):
    """Yield batches holding an exact per-domain count.

    ratios: {"gamus": 0.75, "india": 0.25}.  Normalised, then turned into integer
    per-batch counts by largest-remainder, so the counts sum to batch_size exactly and
    no domain that was asked for silently disappears.

    A domain whose share rounds to 0 is an ERROR rather than a silent drop: asking for
    10% of a batch of 4 and receiving nothing is exactly the kind of quiet divergence
    between intent and behaviour that this project keeps finding the hard way.  Raise
    the batch size or accum_steps instead.
    """

    def __init__(self, dataset: MixedDomains, batch_size: int, ratios: dict,
                 anchor: str | None = None, seed: int = 0, drop_last: bool = True):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.ds, self.batch_size, self.seed, self.drop_last = dataset, batch_size, seed, drop_last

        missing = set(ratios) - set(dataset.domain_ranges)
        if missing:
            raise ValueError(f"ratios name unknown domains: {sorted(missing)}; "
                             f"dataset has {sorted(dataset.domain_ranges)}")
        unlisted = set(dataset.domain_ranges) - set(ratios)
        if unlisted:
            raise ValueError(f"domains present but given no ratio: {sorted(unlisted)}. "
                             f"Pass 0.0 explicitly to exclude one.")

        total = float(sum(ratios.values()))
        if total <= 0:
            raise ValueError("ratios sum to zero")
        want = {d: batch_size * v / total for d, v in ratios.items()}

        # Largest remainder: floor everything, then hand out the leftover slots to the
        # biggest fractional parts.  Guarantees sum(counts) == batch_size.
        counts = {d: int(math.floor(v)) for d, v in want.items()}
        left = batch_size - sum(counts.values())
        for d, _ in sorted(want.items(), key=lambda kv: -(kv[1] - math.floor(kv[1])))[:left]:
            counts[d] += 1

        for d, v in ratios.items():
            if v > 0 and counts[d] == 0:
                raise ValueError(
                    f"domain '{d}' asked for {v/total:.1%} of a batch of {batch_size} "
                    f"and rounds to 0 samples. Raise train.batch_size, or use "
                    f"accum_steps and a larger effective batch. Refusing to silently "
                    f"drop a domain.")
        self.counts = {d: c for d, c in counts.items() if c > 0}

        self.anchor = anchor or max(self.counts, key=lambda d: dataset.sizes()[d])
        if self.anchor not in self.counts:
            raise ValueError(f"anchor '{self.anchor}' has a zero share")
        n_anchor = dataset.sizes()[self.anchor]
        self.n_batches = n_anchor // self.counts[self.anchor]
        if self.n_batches == 0:
            raise ValueError(f"anchor domain '{self.anchor}' has {n_anchor} samples, "
                             f"fewer than the {self.counts[self.anchor]} needed per batch")
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Reshuffle differently each epoch.  Without this every epoch would draw the
        same small-domain repeats in the same order, which turns oversampling into
        memorising one fixed sequence."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator().manual_seed(self.seed + 100003 * self.epoch)
        pools = {}
        for d, c in self.counts.items():
            idx = self.ds.indices_of(d)
            need = self.n_batches * c
            # Repeat-then-shuffle for a domain smaller than one epoch's demand, so every
            # sample is seen an equal number of times rather than sampled with
            # replacement, which would leave some tiles unused for a whole epoch.
            reps = math.ceil(need / len(idx))
            buf = []
            for _ in range(reps):
                perm = torch.randperm(len(idx), generator=g).tolist()
                buf.extend(idx[i] for i in perm)
            pools[d] = buf[:need]
        for b in range(self.n_batches):
            batch = []
            for d, c in self.counts.items():
                batch.extend(pools[d][b * c:(b + 1) * c])
            # Shuffle WITHIN the batch: some layers (and any future BatchNorm) are
            # order-sensitive, and a batch laid out [gamus...][india...] would put a
            # systematic structure into the batch axis for no reason.
            perm = torch.randperm(len(batch), generator=g).tolist()
            yield [batch[i] for i in perm]

    def describe(self) -> str:
        sz = self.ds.sizes()
        parts = ", ".join(f"{d}:{c}/{self.batch_size} of {sz[d]} tiles"
                          for d, c in self.counts.items())
        return (f"mixed batches [{parts}]  anchor={self.anchor}  "
                f"{self.n_batches} batches/epoch")
