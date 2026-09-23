"""DepthX inference service: one uploaded image in, a Three.js bundle out.

    python -m uvicorn serve.app:app --host 0.0.0.0 --port 8000

    POST /api/predict        multipart image -> {"job_id": ...}
    GET  /api/jobs/{id}      status, then the manifest when it is done
    GET  /api/jobs/{id}/bundle/{name}   heightmap.tif | texture.png | metadata.json
    GET  /api/jobs/{id}/bundle.zip      all of it in one download
    GET  /api/health         device, checkpoint, whether the model is warm

WHY A JOB ID AND NOT A PLAIN SYNCHRONOUS RESPONSE.  A 2048x2048 scene takes ~30 s on an
M-series Mac.  Vercel's serverless functions time out well before that, and browsers and
proxies drop long-held connections, so a synchronous design fails exactly on the large
scenes worth demonstrating.  Upload returns immediately; the client polls.

THE MODEL IS LOADED ONCE, at startup, and inference is serialised behind a lock.  Two
concurrent 25-tile scenes on one GPU do not go twice as fast, they go twice as slowly
and risk running out of memory, so requests queue instead.

This is a private service by default: it binds where you tell it, checks a token when
DEPTHX_TOKEN is set, and only allows the origins in DEPTHX_ORIGINS. Set both before
putting it behind any tunnel.
"""
from __future__ import annotations
import os, sys, io, json, uuid, time, shutil, zipfile, threading, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

CKPT = os.environ.get("DEPTHX_CKPT", "outputs/a7_vitl/best.pt")
JOBS_DIR = Path(os.environ.get("DEPTHX_JOBS", "outputs/jobs"))
TOKEN = os.environ.get("DEPTHX_TOKEN")                       # unset = no auth
# Browsers send an Origin header on every cross-site fetch and refuse the
# RESPONSE unless it names that origin back. The default therefore has to cover
# the ways this service is actually reached during development, because the
# failure mode is invisible from the server side -- the request arrives, is
# served, logs 200, and the browser then throws the answer away. Someone
# debugging that from the front end sees only "Failed to fetch".
#
# "localhost" and "127.0.0.1" are DIFFERENT ORIGINS to a browser even though
# they are the same machine, so both are listed. Set DEPTHX_ORIGINS explicitly
# for anything public, and keep the local ones if a teammate is pointing
# `next dev` at this box with ?api=<tunnel>.
DEFAULT_ORIGINS = ",".join([
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:3001", "http://127.0.0.1:3001",
])
ORIGINS = [o.strip() for o in os.environ.get(
    "DEPTHX_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()]
MAX_MB = float(os.environ.get("DEPTHX_MAX_MB", "40"))
MAX_PIXELS = int(os.environ.get("DEPTHX_MAX_PIXELS", str(6000 * 6000)))
ALLOWED = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

app = FastAPI(title="DepthX inference", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])

JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()          # one inference at a time
_MODEL = None                     # (model, cfg, device), loaded lazily on first use


def _auth(token: str | None):
    if TOKEN and token != TOKEN:
        raise HTTPException(401, "bad or missing X-DepthX-Token")


def get_model():
    global _MODEL
    if _MODEL is None:
        from heightmap.predict import load_model
        from heightmap.utils.misc import pick_device
        dev = pick_device()
        t0 = time.time()
        model, cfg = load_model(CKPT, dev)
        print(f"loaded {CKPT} on {dev} in {time.time()-t0:.1f} s", flush=True)
        _MODEL = (model, cfg, dev)
    return _MODEL


def _run_job(job_id: str, image_path: Path, opts: dict):
    import run_pipeline
    job = JOBS[job_id]
    try:
        job.update(status="running", started_at=time.time())
        a = argparse.Namespace(
            ckpt=CKPT, image=str(image_path), out=str(image_path.parent),
            gsd=opts.get("gsd"), assume_gsd=None, dem=None,
            auto_dem=bool(opts.get("auto_dem")),
            terrain_mode=opts.get("terrain_mode", "dem"), max_structure=120.0,
            base_elevation=float(opts.get("base_elevation", 0.0)),
            render_quantity=opts.get("render_quantity", "agl"),
            batch=4, tta=False, no_level=False, device="auto", progress=False)
        with _LOCK:
            manifest = run_pipeline.run(a, preloaded=get_model())
        job.update(status="done", manifest=manifest, finished_at=time.time())
    except Exception as e:                       # the client gets the reason, not a 500
        import traceback; traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}",
                   finished_at=time.time())


@app.get("/api/health")
def health():
    return {"ok": True, "checkpoint": CKPT, "checkpoint_present": Path(CKPT).exists(),
            "model_warm": _MODEL is not None,
            "device": str(_MODEL[2]) if _MODEL else None,
            "auth_required": bool(TOKEN), "allowed_origins": ORIGINS,
            "max_upload_mb": MAX_MB, "jobs": len(JOBS)}


@app.post("/api/predict")
async def predict(background: BackgroundTasks, image: UploadFile = File(...),
                  gsd: float | None = Form(None), auto_dem: bool = Form(False),
                  render_quantity: str = Form("agl"),
                  x_depthx_token: str | None = Header(None)):
    _auth(x_depthx_token)
    ext = Path(image.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(415, f"{ext or 'file'} not accepted; use {sorted(ALLOWED)}")
    data = await image.read()
    if len(data) > MAX_MB * 1e6:
        raise HTTPException(413, f"{len(data)/1e6:.1f} MB exceeds the {MAX_MB} MB limit")

    job_id = uuid.uuid4().hex[:12]
    d = JOBS_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"input{ext}"
    path.write_bytes(data)

    # Reject an image too large to serve before it occupies the GPU for ten minutes.
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            w, h = im.size
        if w * h > MAX_PIXELS:
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(413, f"{w}x{h} exceeds {MAX_PIXELS:,} pixels")
    except HTTPException:
        raise
    except Exception:
        w = h = None                              # a GeoTIFF PIL cannot open is fine

    JOBS[job_id] = {"id": job_id, "status": "queued", "created_at": time.time(),
                    "filename": image.filename, "bytes": len(data),
                    "input_size": [w, h] if w else None, "dir": str(d)}
    background.add_task(_run_job, job_id, path,
                        {"gsd": gsd, "auto_dem": auto_dem,
                         "render_quantity": render_quantity})
    return {"job_id": job_id, "status": "queued",
            "poll": f"/api/jobs/{job_id}", "queued_ahead": _LOCK.locked()}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "unknown job")
    out = {k: v for k, v in j.items() if k != "dir"}
    if j["status"] == "done":
        out["files"] = {n: f"/api/jobs/{job_id}/bundle/{n}" for n in
                        ("heightmap.tif", "texture.png", "metadata.json",
                         "heightmap_preview.png")}
        out["zip"] = f"/api/jobs/{job_id}/bundle.zip"
    return out


@app.get("/api/jobs/{job_id}/bundle/{name}")
def bundle_file(job_id: str, name: str):
    j = JOBS.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404, "not ready")
    if "/" in name or ".." in name:               # never join a client string blindly
        raise HTTPException(400, "bad name")
    p = Path(j["dir"]) / "bundle" / name
    if not p.exists():
        raise HTTPException(404, name)
    return FileResponse(p, filename=name)


@app.get("/api/jobs/{job_id}/bundle.zip")
def bundle_zip(job_id: str):
    j = JOBS.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404, "not ready")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((Path(j["dir"]) / "bundle").iterdir()):
            z.write(p, p.name)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition":
                             f'attachment; filename="depthx_{job_id}.zip"'})


@app.delete("/api/jobs/{job_id}")
def drop(job_id: str, x_depthx_token: str | None = Header(None)):
    _auth(x_depthx_token)
    j = JOBS.pop(job_id, None)
    if not j:
        raise HTTPException(404, "unknown job")
    shutil.rmtree(j["dir"], ignore_errors=True)
    return {"deleted": job_id}
