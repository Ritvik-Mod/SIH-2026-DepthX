"""One-off: copy the v0.1.0 checkpoint from the private GitHub release into Modal.

    modal secret create depthx-github GITHUB_TOKEN=<token>     # read access to the repo
    modal run serve/fetch_weights_modal.py
    modal secret delete depthx-github                          # the token is needed once

Runs inside Modal, so the 1.25 GB goes GitHub -> Modal over datacentre links instead
of down to a laptop and back up again. Refuses to keep the file unless its size and
sha256 match the release record in PIPELINE.md.
"""
import modal

REPO = "Ritvik-Mod/SIH-2026-DepthX"
TAG = "v0.1.0"
ASSET = "a7_vitl_v0.1.0.pt"
SHA256 = "27c4098c4689ff7866fa09276637107c79cee9bf1ca7b1dd9d5d6cb9d4dbe1df"
SIZE = 1_341_893_322

app = modal.App("depthx-fetch-weights")
weights = modal.Volume.from_name("depthx-weights", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").uv_pip_install("requests")


@app.function(image=image, volumes={"/weights": weights},
              secrets=[modal.Secret.from_name("depthx-github")], timeout=1800)
def fetch() -> dict:
    import hashlib
    import os
    import time
    import requests

    tok = os.environ["GITHUB_TOKEN"].strip()
    h = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    r = requests.get(f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}", headers=h, timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"release lookup failed: HTTP {r.status_code} {r.text[:200]} "
                         "(does the token have read access to this repository?)")
    asset = next((a for a in r.json()["assets"] if a["name"] == ASSET), None)
    if not asset:
        raise SystemExit(f"{ASSET} not found in release {TAG}")

    tmp = f"/weights/.{ASSET}.part"
    dst = f"/weights/{ASSET}"
    sha = hashlib.sha256()
    got = 0
    t0 = time.time()
    with requests.get(asset["url"], headers={**h, "Accept": "application/octet-stream"},
                      stream=True, timeout=60) as dl:
        dl.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in dl.iter_content(8 << 20):
                f.write(chunk)
                sha.update(chunk)
                got += len(chunk)
    digest = sha.hexdigest()
    secs = time.time() - t0
    if got != SIZE or digest != SHA256:
        os.remove(tmp)
        weights.commit()
        raise SystemExit(f"REJECTED: {got} bytes, sha256 {digest} (want {SIZE}, {SHA256})")
    os.replace(tmp, dst)
    weights.commit()
    return {"bytes": got, "sha256": digest, "seconds": round(secs, 1),
            "MB_per_s": round(got / 1e6 / secs, 1)}


@app.local_entrypoint()
def main():
    print(fetch.remote())
