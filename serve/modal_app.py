"""DepthX inference on Modal: an L4 GPU that exists only while someone is using it.

    modal deploy serve/modal_app.py          # publish (prints the permanent URL)
    modal run serve/modal_app.py::bench      # time a cold start and warm runs

Replaces Ritvik's Mac + Cloudflare quick tunnel. Same pipeline (scripts/run_pipeline.py),
same bundle out, same HTTP contract as serve/app.py -- so the website needs no new
upload path, only a new address.

TWO FUNCTIONS, ON PURPOSE.

  web        a small CPU container: uploads, job status, file downloads, the admin
             log. Cheap enough (~$0.02/hr while up) that a page load, a health check
             or a status poll never has to wake a GPU.
  Inference  the L4. Started by an upload or by the site's "Warm up GPU" button,
             kept alive only by heartbeats from a page someone is actually looking at,
             and shut down the moment the last such page closes.

COLD START. What costs time when a GPU container starts, and what was done about it:

  - the 1.3 GB Depth Anything V2 backbone: HeightNet calls from_pretrained() on every
    build. Baked into the image at build time, loaded offline.
  - Python imports (torch, transformers, rasterio) and building + loading the 1.25 GB
    checkpoint: done once, then captured in a MEMORY SNAPSHOT (enter(snap=True)).
    Later cold starts restore that memory image instead of redoing the work.
  - moving the model to the GPU and the first CUDA kernels: unavoidable per container,
    done in enter(snap=False) with one warm-up forward pass, so the first real request
    is not also paying for kernel selection.

SHUTTING DOWN. A GPU left idle costs $0.80/hr for nothing, so:

  - scaledown_window is short (90 s): with nothing arriving, the container exits;
  - open pages send a heartbeat every ~25 s, ONLY while the tab is visible and someone
    has touched it in the last ten minutes, so a tab left open overnight still lets the
    GPU sleep;
  - closing the page sends a release beacon; when no other page is active the GPU
    container stops taking work and exits immediately (stop_fetching_inputs).

COST GUARDS. max_containers=1 caps the worst case at one L4 (~$1.12/hr all-in, CPU and
RAM included), and uploads are rate-limited per IP and per day.

ADMIN LOG. Every upload is recorded -- time, IP, browser, file, outcome, timings -- with
the image itself, readable at /admin on the site with the password held in the Modal
secret `depthx-admin`. The password is never in this repository or the website bundle.
"""

import os
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
APP = "depthx"
BACKBONE = "depth-anything/Depth-Anything-V2-Large-hf"
CKPT = "/weights/a7_vitl_v0.1.0.pt"
GPU = "L4"

SCALEDOWN_S = 90          # idle seconds before the GPU container exits on its own
HEARTBEAT_EVERY_S = 20    # the web tier forwards at most one heartbeat per this many seconds
CLIENT_TTL_S = 75         # a page silent for this long no longer counts as open
RATE_PER_IP_HOUR = 20
RATE_PER_DAY = 400
MAX_UPLOAD_MB = 40

# Testing aid ONLY: without the checkpoint, build the same network with untrained heads.
# Identical speed, meaningless heights. Set it for a timing deploy; never for the real one.
ALLOW_UNTRAINED = os.environ.get("DEPTHX_ALLOW_UNTRAINED", "0")

app = modal.App(APP)
weights = modal.Volume.from_name("depthx-weights", create_if_missing=True)
data = modal.Volume.from_name("depthx-data", create_if_missing=True)
state = modal.Dict.from_name("depthx-state", create_if_missing=True)
admin_secret = modal.Secret.from_name("depthx-admin")


def _bake_backbone():
    from huggingface_hub import snapshot_download
    snapshot_download(BACKBONE)


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        "transformers>=4.44,<4.50", "huggingface_hub", "safetensors",
        "numpy<2.3", "scipy", "rasterio", "opencv-python-headless", "pillow",
        "imageio", "matplotlib", "omegaconf", "tqdm", "h5py",
    )
    .env({"HF_HOME": "/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .run_function(_bake_backbone)
    .env({"HF_HUB_OFFLINE": "1", "DEPTHX_ALLOW_UNTRAINED": ALLOW_UNTRAINED,
          "PYTHONPATH": "/root:/root/scripts"})
    .add_local_dir(ROOT / "heightmap", "/root/heightmap")
    .add_local_dir(ROOT / "configs", "/root/configs")
    .add_local_dir(ROOT / "scripts", "/root/scripts")
)

web_image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "fastapi[standard]", "python-multipart", "pillow",
)


# ============================================================================ helpers

def _now() -> float:
    return time.time()


def _job(job_id: str) -> dict:
    return state.get(f"job:{job_id}", {}) or {}


def _set_job(job_id: str, **patch) -> dict:
    j = _job(job_id)
    j.update(patch)
    state[f"job:{job_id}"] = j
    return j


def _gpu_state() -> dict:
    g = state.get("gpu", None) or {"state": "asleep", "since": 0}
    # A start that never reported in (container failed to boot) must not read as
    # "starting" forever.
    if g.get("state") == "starting" and _now() - g.get("since", 0) > 240:
        g = {"state": "asleep", "since": g.get("since", 0), "note": "start timed out"}
    return g


def _busy(delta: int) -> int:
    """Jobs queued or running. A release never stops a GPU that still has one."""
    n = max(0, (state.get("busy", 0) or 0) + delta)
    state["busy"] = n
    return n


def _write_meta(run_id: str, patch: dict, name: str = "meta.json"):
    """Merge fields into /data/runs/<id>/<name> -- the admin log's record.

    Two containers write to each run: the web tier writes meta.json (who, what, when)
    and the GPU writes result.json (outcome, timings). One file each, never shared:
    a Volume commit replaces whole files, and the GPU container's view of the volume
    can predate the web tier's write, so a shared file lost the IP, filename and
    browser whenever the GPU's copy landed second.
    """
    import json
    p = Path("/data/runs") / run_id / name
    meta = json.loads(p.read_text()) if p.exists() else {}
    meta.update(patch)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(meta, indent=2, default=str))


def _repack_heights(src: Path, dst: Path):
    """Copy a float GeoTIFF with deflate + floating-point predictor (lossless)."""
    import rasterio
    with rasterio.open(src) as ds:
        prof = ds.profile.copy()
        arr = ds.read()
        tags = ds.tags()
        band_tags = [ds.tags(i + 1) for i in range(ds.count)]
        desc = ds.descriptions
    prof.update(compress="deflate", predictor=3, zlevel=6, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(dst, "w", **prof) as out:
        out.write(arr)
        out.update_tags(**tags)
        for i, t in enumerate(band_tags):
            out.update_tags(i + 1, **t)
            if desc and desc[i]:
                out.set_band_description(i + 1, desc[i])


# ============================================================================ GPU tier

@app.cls(
    image=gpu_image,
    gpu=GPU,
    cpu=4,
    memory=16384,
    volumes={"/weights": weights, "/data": data},
    enable_memory_snapshot=True,
    scaledown_window=SCALEDOWN_S,
    max_containers=1,
    timeout=600,
)
class Inference:

    @modal.enter(snap=True)
    def load(self):
        """Everything here is captured in the memory snapshot: paid once, not per start."""
        import torch
        import run_pipeline                      # noqa: F401  (heavy imports into the snapshot)
        from heightmap.predict import load_model

        self.error = None
        self.untrained = False
        self.model = self.cfg = None
        if Path(CKPT).exists():
            # Force the HF id for the backbone ARCHITECTURE: the checkpoint's saved cfg
            # may name a cluster path. Its own weights overwrite the backbone's anyway.
            self.model, self.cfg = load_model(CKPT, torch.device("cpu"),
                                              override_cfg=[f"model.checkpoint={BACKBONE}"])
        elif os.environ.get("DEPTHX_ALLOW_UNTRAINED") == "1":
            from heightmap import config
            from heightmap.models.net import HeightNet
            self.cfg = config.load(overrides=[f"model.checkpoint={BACKBONE}"])
            self.model = HeightNet(self.cfg)
            self.untrained = True
        else:
            self.error = ("The model weights are not uploaded yet: a7_vitl_v0.1.0.pt is "
                          "missing from the depthx-weights volume.")
        if self.model is not None:
            self.model.eval()

    @modal.enter(snap=False)
    def to_gpu(self):
        import torch
        t0 = _now()
        # TF32 tensor cores for fp32 matmuls/convs: ~2x on an L4. Inputs keep a
        # 10-bit mantissa and accumulate in fp32, far below the model's own error.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # No per-shape autotuning: with it, every new batch size (the last batch of a
        # scene is usually a partial one) paid a one-off ~3 s tuning pass inside a
        # request someone was timing. Measured: 4.2 s first run vs 1.0 s after.
        torch.backends.cudnn.benchmark = False
        self.dev = torch.device("cuda")
        if self.model is not None:
            self.model.to(self.dev)
            with torch.inference_mode():
                # A 1024 px frame is 9 tiles: one full batch of 8 and a partial batch,
                # so both code paths, the allocator and the kernels are warm before
                # the first real request instead of during it.
                from heightmap.infer import predict_scene
                import numpy as np
                predict_scene(self.model, np.zeros((1024, 1024, 3), np.uint8), 0.33,
                              tile=int(self.cfg.infer.tile), overlap=float(self.cfg.infer.overlap),
                              batch=8, tta=False, level=True, device=self.dev, progress=False)
            torch.cuda.synchronize()
        self.ready_at = _now()
        self.fresh = True                     # the first request on this container was cold
        state["gpu"] = {"state": "warm", "since": self.ready_at,
                        "gpu_load_s": round(self.ready_at - t0, 2),
                        "error": self.error, "untrained": self.untrained}

    @modal.exit()
    def bye(self):
        state["gpu"] = {"state": "asleep", "since": _now()}

    @modal.method()
    def warm(self) -> dict:
        # The warm-up paid the start-up; the next request on this container is warm.
        self.fresh = False
        return {"ready_at": self.ready_at, "error": self.error}

    @modal.method()
    def ping(self) -> bool:
        return True

    @modal.method()
    def release(self) -> bool:
        """Finish anything in flight, take no more work, exit."""
        modal.experimental.stop_fetching_inputs()
        return True

    @modal.method()
    def run(self, job_id: str, image_name: str, image_bytes: bytes, opts: dict) -> dict:
        import argparse
        import shutil
        import traceback

        started = _now()
        try:
            data.reload()          # see the web tier's commit (preview, record) for this run
        except Exception:
            pass
        cold = self.fresh
        self.fresh = False
        j = _set_job(job_id, status="running", started_at=started, cold=cold,
                     untrained=self.untrained)
        wait_s = round(started - j.get("submitted_at", started), 2)
        _set_job(job_id, wait_s=wait_s)

        if self.error:
            _busy(-1)
            _set_job(job_id, status="error", error=self.error, finished_at=_now())
            _write_meta(job_id, {"status": "error", "error": self.error, "cold": cold}, "result.json")
            data.commit()
            return {"error": self.error}

        work = Path("/tmp") / job_id
        work.mkdir(parents=True, exist_ok=True)
        src = work / image_name
        src.write_bytes(image_bytes)
        try:
            import run_pipeline
            a = argparse.Namespace(
                ckpt=CKPT, image=str(src), out=str(work), gsd=opts.get("gsd"),
                assume_gsd=None, dem=None, auto_dem=bool(opts.get("auto_dem")),
                terrain_mode="dem", max_structure=120.0, base_elevation=0.0,
                render_quantity=opts.get("render_quantity", "agl"),
                batch=8, tta=False, no_level=False, device="cuda", progress=False)
            manifest = run_pipeline.run(a, preloaded=(self.model, self.cfg, self.dev))

            out = Path("/data/runs") / job_id / "bundle"
            out.mkdir(parents=True, exist_ok=True)
            for name in ("texture.png", "metadata.json", "heightmap_preview.png"):
                p = work / "bundle" / name
                if p.exists():
                    shutil.copy(p, out / name)
            # Smaller downloads, same data. Measured on this service, the bundle
            # download (not the model) was the slowest step: 61 s for a 3088 px scene
            # over an ordinary connection. The heights are re-saved with the TIFF
            # floating-point predictor -- lossless, every value bit-identical -- and
            # the texture goes out as JPEG q90, as the sample scenes already do.
            _repack_heights(work / "bundle" / "heightmap.tif", out / "heightmap.tif")
            if (out / "texture.png").exists():
                from PIL import Image
                Image.open(out / "texture.png").convert("RGB").save(
                    out / "texture.jpg", "JPEG", quality=90, optimize=True)
            # admin thumbnail, from the texture the model actually saw
            prev = Path("/data/runs") / job_id / "preview.jpg"
            if not prev.exists() and (out / "texture.png").exists():
                from PIL import Image
                im = Image.open(out / "texture.png").convert("RGB")
                im.thumbnail((640, 640))
                im.save(prev, "JPEG", quality=82)

            finished = _now()
            timing = {
                "wait_s": wait_s,                                       # queue + GPU start-up
                "inference_s": manifest["timing_s"]["inference"],      # the model itself
                "pipeline_s": round(finished - started, 2),            # model + terrain + files
            }
            meta = {
                "status": "done", "cold": cold, "timing": timing,
                "quantity": manifest.get("quantity"),
                "georeferenced": manifest.get("georeferenced"),
                "untrained": self.untrained,
            }
            _write_meta(job_id, meta, "result.json")
            data.commit()
            # The finished files travel back as this call's RETURN VALUE, which Modal
            # delivers reliably. They used to be handed over through the shared volume,
            # and the web tier's first read raced its own reload of that volume: the
            # three parallel downloads of a first run came back 404/500/200.
            names = [n for n in ("heightmap.tif", "texture.jpg", "metadata.json") if (out / n).exists()]
            payload = {n: (out / n).read_bytes() for n in names}
            _set_job(job_id, status="done", finished_at=finished, timing=timing,
                     manifest=manifest,
                     files={n: f"/api/jobs/{job_id}/bundle/{n}" for n in names})
            return {"timing": timing, "files": payload}
        except Exception as e:                     # the client gets the reason, not a 500
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            _set_job(job_id, status="error", error=msg, finished_at=_now())
            _write_meta(job_id, {"status": "error", "error": msg, "cold": cold}, "result.json")
            data.commit()
            return {"error": msg}
        finally:
            _busy(-1)
            shutil.rmtree(work, ignore_errors=True)


# ============================================================================ web tier

@app.function(
    image=web_image,
    volumes={"/data": data},
    secrets=[admin_secret],
    cpu=0.5,
    memory=2048,
    scaledown_window=300,
    max_containers=2,
)
@modal.concurrent(max_inputs=64)
@modal.asgi_app(label="depthx-api")
def web():
    import hmac
    import io
    import json
    import re
    import uuid

    from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response

    import threading
    results: dict = {}                 # job_id -> {name: bytes}, most recent last
    results_lock = threading.Lock()
    reload_lock = threading.Lock()

    api = FastAPI(title="DepthX inference (Modal)")
    api.add_middleware(
        CORSMiddleware,
        # Any Vercel deployment (production and every branch preview) and any local
        # dev port: preview hostnames cannot be listed in advance.
        allow_origin_regex=r"^(https://[a-z0-9.-]+\.vercel\.app|http://localhost(:\d+)?"
                           r"|http://127\.0\.0\.1(:\d+)?)$",
        allow_methods=["*"], allow_headers=["*"],
    )
    ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    def client_ip(req: Request) -> str:
        fwd = req.headers.get("x-forwarded-for", "")
        return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")

    def check_admin(pw: str | None):
        want = os.environ.get("DEPTHX_ADMIN_PASSWORD", "")
        if not want or not pw or not hmac.compare_digest(pw.encode(), want.encode()):
            raise HTTPException(401, "wrong password")

    def rate_limit(ip: str):
        now = _now()
        hits = [t for t in (state.get(f"rate:{ip}", []) or []) if now - t < 3600]
        if len(hits) >= RATE_PER_IP_HOUR:
            raise HTTPException(429, f"Limit of {RATE_PER_IP_HOUR} reconstructions per hour reached. "
                                     "The sample scenes still work.")
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        n = state.get(f"day:{day}", 0) or 0
        if n >= RATE_PER_DAY:
            raise HTTPException(429, "Today's reconstruction budget is used up. The sample scenes still work.")
        state[f"rate:{ip}"] = hits + [now]
        state[f"day:{day}"] = n + 1

    def active_clients() -> dict:
        now = _now()
        c = state.get("clients", {}) or {}
        return {k: v for k, v in c.items() if now - v < CLIENT_TTL_S}

    def wake_gpu():
        g = _gpu_state()
        if g["state"] in ("warm", "starting"):
            return g
        g = {"state": "starting", "since": _now()}
        state["gpu"] = g
        Inference().warm.spawn()
        return g

    # ---------------------------------------------------------------- public API
    @api.get("/api/health")
    def health():
        return {"ok": True, "backend": "modal", "gpu_type": GPU, "gpu": _gpu_state(),
                "model_warm": _gpu_state()["state"] == "warm",
                "scaledown_s": SCALEDOWN_S, "max_upload_mb": MAX_UPLOAD_MB}

    @api.post("/api/warmup")
    def warmup():
        return {"gpu": wake_gpu()}

    @api.post("/api/keepalive")
    async def keepalive(req: Request):
        try:
            cid = (await req.json()).get("client", "")[:64]
        except Exception:
            cid = ""
        if cid:
            c = active_clients()
            c[cid] = _now()
            state["clients"] = c
        g = _gpu_state()
        # Only a GPU that is already up is kept up -- a heartbeat never starts one.
        if g["state"] == "warm" and _now() - (state.get("last_ping", 0) or 0) > HEARTBEAT_EVERY_S:
            state["last_ping"] = _now()
            Inference().ping.spawn()
        return {"gpu": g}

    @api.post("/api/release")
    async def release(req: Request):
        # Sent with navigator.sendBeacon as text/plain when a page closes.
        try:
            cid = json.loads((await req.body()) or b"{}").get("client", "")
        except Exception:
            cid = ""
        c = active_clients()
        c.pop(cid, None)
        state["clients"] = c
        busy = state.get("busy", 0) or 0
        g = _gpu_state()
        if not c and not busy and g["state"] == "warm":
            Inference().release.spawn()
            return {"released": True}
        return {"released": False, "active_pages": len(c)}

    @api.post("/api/predict")
    async def predict(req: Request, image: UploadFile = File(...),
                      gsd: float | None = Form(None), auto_dem: bool = Form(False),
                      render_quantity: str = Form("agl")):
        ext = Path(image.filename or "").suffix.lower()
        if ext not in ALLOWED:
            raise HTTPException(415, f"{ext or 'file'} not accepted; use {sorted(ALLOWED)}")
        body = await image.read()
        if len(body) > MAX_UPLOAD_MB * 1e6:
            raise HTTPException(413, f"{len(body)/1e6:.1f} MB exceeds the {MAX_UPLOAD_MB} MB limit")
        ip = client_ip(req)
        rate_limit(ip)

        job_id = uuid.uuid4().hex[:12]
        now = _now()
        run_dir = Path("/data/runs") / job_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"input{ext}").write_bytes(body)
        size = None
        try:                                       # thumbnail now, if PIL can read it
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(io.BytesIO(body))
            size = list(im.size)
            im = im.convert("RGB")
            im.thumbnail((640, 640))
            im.save(run_dir / "preview.jpg", "JPEG", quality=82)
        except Exception:
            pass                                   # the GPU writes one from the texture instead
        gpu_before = _gpu_state()["state"]
        _write_meta(job_id, {
            "id": job_id, "submitted_at": now,
            "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
            "ip": ip, "user_agent": req.headers.get("user-agent", ""),
            "origin": req.headers.get("origin", ""), "filename": image.filename,
            "bytes": len(body), "input_size": size, "input": f"input{ext}",
            "gsd": gsd, "auto_dem": auto_dem, "render_quantity": render_quantity,
            "gpu_before": gpu_before, "status": "queued",
        })
        data.commit()

        _set_job(job_id, id=job_id, status="queued", submitted_at=now, gpu_before=gpu_before)
        _busy(+1)
        if gpu_before != "warm":
            state["gpu"] = {"state": "starting", "since": now}
        call = Inference().run.spawn(job_id, f"input{ext}", body,
                                     {"gsd": gsd, "auto_dem": auto_dem, "render_quantity": render_quantity})
        _set_job(job_id, call_id=call.object_id)
        return {"job_id": job_id, "status": "queued", "poll": f"/api/jobs/{job_id}",
                "gpu": gpu_before}

    @api.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        j = _job(job_id)
        if not j:
            raise HTTPException(404, "unknown job")
        out = {k: v for k, v in j.items() if k != "manifest"}
        out["manifest"] = j.get("manifest")
        out["now"] = _now()
        out["gpu"] = _gpu_state()
        return out

    MEDIA = {".tif": "image/tiff", ".jpg": "image/jpeg", ".png": "image/png", ".json": "application/json"}

    def result_files(job_id: str):
        """The finished bundle, straight from the GPU call's return value."""
        with results_lock:
            if job_id in results:
                return results[job_id]
        cid = _job(job_id).get("call_id")
        if not cid:
            return None
        try:
            res = modal.FunctionCall.from_id(cid).get(timeout=60)
        except Exception:
            return None
        files = (res or {}).get("files")
        if not files:
            return None
        with results_lock:
            results[job_id] = files
            while len(results) > 8:                # the last few jobs, per container
                results.pop(next(iter(results)))
        return files

    @api.get("/api/jobs/{job_id}/bundle/{name}")
    def bundle_file(job_id: str, name: str):
        if "/" in name or ".." in name or not re.fullmatch(r"[a-f0-9]{12}", job_id):
            raise HTTPException(400, "bad name")
        media = MEDIA.get(Path(name).suffix, "application/octet-stream")
        files = result_files(job_id)
        if files and name in files:
            return Response(files[name], media_type=media,
                            headers={"Content-Disposition": f'attachment; filename="{name}"'})
        # Older jobs: read from the archive. Bytes are read whole rather than streamed,
        # and reloads are serialised, so no request can have the volume swapped out
        # from under a response it is still sending.
        p = Path("/data/runs") / job_id / "bundle" / name
        if not p.exists():
            with reload_lock:
                try:
                    data.reload()
                except Exception:
                    pass
        if not p.exists():
            raise HTTPException(404, name)
        return Response(p.read_bytes(), media_type=media)

    # ---------------------------------------------------------------- admin
    def locate(ips: list) -> dict:
        """IP -> "City, Region, Country". Cached per IP; one batch request for misses.

        ip-api.com's batch endpoint: free, no key, up to 100 addresses per request.
        Any failure degrades to "Unknown location" -- the log must never fail to load
        because a geolocation service is down.
        """
        import ipaddress
        import urllib.request
        out, misses = {}, []
        for ip in set(ips):
            try:
                a = ipaddress.ip_address(ip)
                if a.is_private or a.is_loopback or a.is_link_local:
                    out[ip] = "Local network"
                    continue
            except ValueError:
                out[ip] = "Unknown location"
                continue
            hit = state.get(f"geo:{ip}", None)
            if hit:
                out[ip] = hit
            else:
                misses.append(ip)
        for i in range(0, len(misses), 100):
            chunk = misses[i:i + 100]
            try:
                req = urllib.request.Request(
                    "http://ip-api.com/batch?fields=status,country,regionName,city,query",
                    data=json.dumps(chunk).encode(), headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=4) as r:
                    for row in json.loads(r.read()):
                        ip = row.get("query")
                        if row.get("status") == "success":
                            parts = []
                            for p in (row.get("city"), row.get("regionName"), row.get("country")):
                                if p and p not in parts:
                                    parts.append(p)
                            place = ", ".join(parts) or "Unknown location"
                            state[f"geo:{ip}"] = place
                        else:
                            place = "Unknown location"
                        out[ip] = place
            except Exception:
                pass
        return {ip: out.get(ip, "Unknown location") for ip in ips}

    def visitor_tag(ip: str) -> str:
        # A stable anonymous id, so "distinct visitors" can still be counted. Keyed
        # with the admin secret: a bare hash of an IPv4 address is reversible by
        # simply hashing all four billion of them.
        key = os.environ.get("DEPTHX_ADMIN_PASSWORD", "depthx").encode()
        return hmac.new(key, (ip or "").encode(), "sha256").hexdigest()[:8]

    @api.get("/api/admin/runs")
    def admin_runs(x_admin_password: str | None = Header(None)):
        check_admin(x_admin_password)
        data.reload()
        runs = []
        root = Path("/data/runs")
        if root.exists():
            for d in root.iterdir():
                meta = {}
                for f in ("meta.json", "result.json"):
                    try:
                        if (d / f).exists():
                            meta.update(json.loads((d / f).read_text()))
                    except Exception:
                        pass
                if meta:
                    meta.setdefault("id", d.name)
                    meta["has_preview"] = (d / "preview.jpg").exists()
                    runs.append(meta)
        runs.sort(key=lambda r: r.get("submitted_at", 0), reverse=True)
        # The IP stays on the server (rate limiting needs it); the log page gets only
        # a coarse location and an anonymous visitor tag, never the address itself.
        places = locate([r.get("ip", "") for r in runs])
        for r in runs:
            ip = r.pop("ip", "")
            r["location"] = places.get(ip, "Unknown location")
            r["visitor"] = visitor_tag(ip)
        return {"runs": runs, "gpu": _gpu_state(), "active_pages": len(active_clients())}

    @api.get("/api/admin/runs/{run_id}/{what}")
    def admin_file(run_id: str, what: str, x_admin_password: str | None = Header(None)):
        check_admin(x_admin_password)
        if not re.fullmatch(r"[a-f0-9]{12}", run_id):
            raise HTTPException(400, "bad id")
        d = Path("/data/runs") / run_id
        if what == "preview":
            p = d / "preview.jpg"
        elif what == "input":
            meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
            p = d / meta.get("input", "input")
        else:
            raise HTTPException(404, what)
        if not p.exists():
            data.reload()
        if not p.exists():
            raise HTTPException(404, what)
        return FileResponse(p, filename=p.name)

    return api


# ============================================================================ benchmark

@app.local_entrypoint()
def bench(image: str = str(ROOT / "final_test_files" / "07_nongeo_nyc_rowhouses.png"),
          repeat: int = 2, auto_dem: bool = False):
    """Time a cold start (container killed first) and warm runs, end to end on Modal."""
    import uuid
    body = Path(image).read_bytes()
    name = "input" + Path(image).suffix.lower()
    for i in range(repeat + 1):
        jid = uuid.uuid4().hex[:12]
        t0 = time.time()
        state[f"job:{jid}"] = {"id": jid, "status": "queued", "submitted_at": t0}
        _busy(+1)
        r = Inference().run.remote(jid, name, body,
                                   {"auto_dem": auto_dem, "render_quantity": "dsm" if auto_dem else "agl"})
        j = state.get(f"job:{jid}", {})
        print(f"run {i} ({'cold' if j.get('cold') else 'warm'}): "
              f"wall {time.time() - t0:5.1f} s   {r.get('timing', r)}")
