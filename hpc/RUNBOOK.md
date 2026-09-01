# DepthWizard on the BIT Mesra HPCF

PBS Pro cluster. Scheduler commands are `qsub` / `qstat` / `qdel` — **not** Slurm.

| | |
|---|---|
| Username | `btech10848.24` |
| Internal IP (on campus) | `172.16.220.100` |
| External IP (off campus) | `115.240.90.140` |
| Queues | `gpu`, `workq` |
| Shared conda | `/apps/anaconda3` (env `deeplearning`) |
| Contact | Rajan Kumar, DST-PURSE, CSE dept |

> Change the default password on first login (`passwd`). It was mailed in plain text and
> is the same for every new account.

## 0 · Connect

```bash
# on campus / VPN
ssh btech10848.24@172.16.220.100
# off campus
ssh btech10848.24@115.240.90.140
```

Set up a key so jobs and rsync stop asking for a password:

```bash
ssh-keygen -t ed25519 -C sih2026            # if you have no key yet
ssh-copy-id btech10848.24@172.16.220.100
```

Add to `~/.ssh/config` on the Mac so everything below can just say `hpcf`:

```
Host hpcf
    HostName 172.16.220.100
    User btech10848.24
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

## 1 · Push the code

```bash
cd "/Users/ritvikmod/SIH 2026"
rsync -rtv --delete \
  --exclude '.venv' --exclude 'data' --exclude 'outputs' --exclude '__pycache__' \
  ./ hpcf:~/SIH2026/
```

Code only — the 80 GB dataset is downloaded **on the cluster**, never uploaded from the Mac.

## 2 · Discover before you commit anything

```bash
ssh hpcf
cd ~/SIH2026 && mkdir -p logs
qsub hpc/00_discover.pbs
qstat -u $USER
cat discover.out
```

This answers the five things I cannot know from here, and every later decision depends
on them:

1. **GPU model and VRAM** → batch size, and whether ViT-L fits
2. **CUDA/driver version** → which torch wheel to install in step 3
3. **Walltime limits per queue** → how the run is chunked
4. **Disk quota and scratch paths** → where the 80 GB goes
5. **Whether compute nodes reach the internet** → whether assets must be pre-staged

**Send me `discover.out` before running step 3.** The wheel tag and batch sizes are
chosen from it; guessing here wastes a queue slot.

> The GPU template you were given has `ngpus=0`, which silently gives you a GPU-queue
> job with no GPU. My scripts set `ngpus=1`. If `nvidia-smi` fails in `discover.out`,
> that is the first thing to check.

## 3 · Build the environment (login node — it needs internet)

```bash
bash hpc/01_setup_env.sh cu121        # tag comes from discover.out
```

Own env rather than the shared `deeplearning` one, so a cluster-wide update cannot
change your results halfway through the ablation ladder.

## 4 · Stage data and weights (login node, inside tmux)

```bash
tmux new -s fetch
export DATA_ROOT=$HOME/SIH2026/data      # or the scratch path from discover.out
bash hpc/02_fetch_assets.sh $DATA_ROOT
# detach: Ctrl-b then d
```

Downloads the three DAv2 backbones plus GAMUS (~80 GB). Compute nodes usually have no
internet, so this must happen here; the job scripts then run fully offline
(`HF_HUB_OFFLINE=1`).

```bash
qsub hpc/10_repack.pbs                   # 80 GB -> ~50 GB
```

Classes `float32→uint8` is exactly lossless (ids 0–6). Heights `float32→float16` costs
~0.015 m, two orders of magnitude below the ~1.5 m MAE target. The dataloader already
casts on read, so no code changes.

## 5 · Prepare and verify on real data

```bash
qsub hpc/20_prepare.pbs
```

Runs, in order: dataset statistics → all 16 sanity checks **on real GAMUS** → the A0
zero-shot baseline → sun-angle estimation for the shadow loss → a one-epoch timing probe.

**Read `logs/prepare.out` before going further.** Three things there change the configs:

- `outputs/gamus_stats.json` → confirms or corrects `h_max` (currently 250) and whether
  class 0 really is unlabelled (`ignore_index`)
- `outputs/a0_zeroshot.json` → the baseline number the whole project argues against
- the timing probe → **the real seconds/epoch.** Every duration below is an estimate
  until this number exists; recalibrate from it.

## 6 · Run the ladder

```bash
bash hpc/31_ladder.sh                     # chains A1 → A7 → holdout
qstat -u $USER -a
```

A dependency chain, not parallel jobs: student allocations usually cap concurrent runs,
and each stage tells you whether the next is worth the queue time.

Single run:

```bash
qsub -v CFG=configs/ablations/a2_bins.yaml hpc/30_train.pbs
```

**Walltime is handled.** `train.resume=auto` continues from `out_dir/last.pt`, and the
checkpoint is written atomically, so a job killed at the limit loses at most one epoch.
Re-`qsub` the identical command to continue; it no-ops once the epoch count is reached.
If the queue's max walltime is short, lower `#PBS -l walltime` and just submit the same
job several times.

## 7 · Collect and pull back

```bash
python scripts/collect_results.py --runs outputs --out outputs/results.md
```

```bash
# from the Mac
rsync -rtv --exclude '*.pt' hpcf:~/SIH2026/outputs/ ./outputs_hpc/
rsync -rtv hpcf:~/SIH2026/outputs/a6_shadow/best.pt ./checkpoints/
```

Exclude `*.pt` on the bulk pull — checkpoints are ~200 MB (ViT-S) to ~2.7 GB (ViT-L).
Fetch only the ones you need.

## Cheat sheet

| Task | Command |
|---|---|
| Submit | `qsub hpc/30_train.pbs` |
| Submit with a config | `qsub -v CFG=configs/ablations/a2_bins.yaml hpc/30_train.pbs` |
| My jobs | `qstat -u $USER -a` |
| Why is it queued | `qstat -f <jobid> \| grep -i comment` |
| Kill | `qdel <jobid>` |
| Hold / release | `qhold <jobid>` / `qrls <jobid>` |
| Live log | `tail -f logs/train.out` |
| Queue limits | `qstat -Qf` |

## Traps specific to this cluster

- **`ngpus=0` in the supplied GPU template.** Set `ngpus=1` or you get no GPU.
- **No internet on compute nodes** (verify in `discover.out`). Everything HuggingFace
  must be pre-staged from the login node, and jobs run with `HF_HUB_OFFLINE=1`.
- **`cd $PBS_O_WORKDIR`** — PBS starts you in `$HOME`, not where you submitted from.
- **Quota.** 80 GB before repacking. If home is small, put `DATA_ROOT` on scratch and
  check whether scratch is purged on a timer.
- **`#PBS -m abe`** mails on abort/begin/end. Useful for overnight runs; noisy for a chain
  of eight, so drop the `b` if you get spammed.
- **Never run training on the login node.** It is shared, and admins kill long processes.
