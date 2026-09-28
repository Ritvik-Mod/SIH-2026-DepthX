'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import Brand from '@/components/Brand';
import Icon from '@/components/Icons';
import { MODAL_BASE } from '@/lib/api';

/**
 * Run log for Team DepthX: every reconstruction anyone started, with the image.
 *
 * The password is checked by the model service (a Modal secret), not here, so it is
 * not in this page's code. It is kept in sessionStorage so a refresh does not ask
 * again, and forgotten when the tab closes.
 */

const KEY = 'depthx_admin';

async function authed(base, path, pw) {
  let r;
  try {
    r = await fetch(`${base}${path}`, { headers: { 'X-Admin-Password': pw } });
  } catch {
    throw new Error(`Could not reach the model service at ${base}.`);
  }
  if (r.status === 401) throw new Error('Wrong password.');
  if (!r.ok) throw new Error(`Service returned ${r.status}.`);
  return r;
}

function Thumb({ base, pw, run, onOpen }) {
  const [url, setUrl] = useState(null);
  useEffect(() => {
    if (!run.has_preview) return undefined;
    let made = null;
    let cancelled = false;
    authed(base, `/api/admin/runs/${run.id}/preview`, pw)
      .then((r) => r.blob())
      .then((b) => { if (!cancelled) { made = URL.createObjectURL(b); setUrl(made); } })
      .catch(() => {});
    return () => { cancelled = true; if (made) URL.revokeObjectURL(made); };
  }, [base, pw, run.id, run.has_preview]);
  return (
    <button type="button" className="aThumb" onClick={() => url && onOpen(url, run)} disabled={!url}>
      {url ? <img src={url} alt="" /> : <span>{run.has_preview ? '' : 'no preview'}</span>}
    </button>
  );
}

const browser = (ua = '') => {
  const os = /Windows/.test(ua) ? 'Windows' : /Mac OS X/.test(ua) ? 'macOS' : /Android/.test(ua) ? 'Android'
    : /iPhone|iPad/.test(ua) ? 'iOS' : /Linux/.test(ua) ? 'Linux' : '';
  const b = /Edg\//.test(ua) ? 'Edge' : /Chrome\//.test(ua) ? 'Chrome' : /Firefox\//.test(ua) ? 'Firefox'
    : /Safari\//.test(ua) ? 'Safari' : /node|curl|python/i.test(ua) ? 'script' : 'other';
  return [b, os].filter(Boolean).join(' · ');
};

const median = (xs) => {
  const a = xs.filter((x) => Number.isFinite(x)).sort((p, q) => p - q);
  return a.length ? a[Math.floor(a.length / 2)] : null;
};

export default function AdminPage() {
  const [pw, setPw] = useState('');
  const [draft, setDraft] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [big, setBig] = useState(null);
  const [base, setBase] = useState('');

  useEffect(() => {
    // The run log only exists on the Modal service, so this page always talks to it,
    // never to a tab-level override. ?api= still works for testing a copy.
    const q = new URLSearchParams(window.location.search).get('api');
    setBase((q || MODAL_BASE).trim().replace(/\/$/, ''));
    try { const saved = sessionStorage.getItem(KEY); if (saved) setPw(saved); } catch {}
  }, []);

  const load = useCallback(async (password) => {
    if (!password || !base) return;
    setLoading(true);
    setError('');
    try {
      const r = await authed(base, '/api/admin/runs', password);
      setData(await r.json());
      try { sessionStorage.setItem(KEY, password); } catch {}
    } catch (e) {
      setError(e.message);
      if (/password/i.test(e.message)) {
        setPw('');
        try { sessionStorage.removeItem(KEY); } catch {}
      }
    } finally {
      setLoading(false);
    }
  }, [base]);

  useEffect(() => { if (pw) load(pw); }, [pw, load]);
  useEffect(() => {
    if (!pw) return undefined;
    const id = setInterval(() => load(pw), 30000);
    return () => clearInterval(id);
  }, [pw, load]);

  const stats = useMemo(() => {
    const runs = data?.runs || [];
    const done = runs.filter((r) => r.status === 'done');
    return {
      total: runs.length,
      done: done.length,
      failed: runs.filter((r) => r.status === 'error').length,
      people: new Set(runs.map((r) => r.ip)).size,
      inference: median(done.map((r) => r.timing?.inference_s)),
      cold: median(done.filter((r) => r.cold).map((r) => r.timing?.wait_s)),
    };
  }, [data]);

  const download = async (run) => {
    try {
      const r = await authed(base, `/api/admin/runs/${run.id}/input`, pw);
      const url = URL.createObjectURL(await r.blob());
      const a = document.createElement('a');
      a.href = url;
      a.download = run.filename || `${run.id}`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    } catch (e) {
      setError(e.message);
    }
  };

  return (
    <div className="home admin">
      <header className="topbar">
        <Brand />
        <div className="topMeta">
          <span>Run log</span>
          {pw && (
            <button type="button" className="btnText" onClick={() => {
              setPw(''); setData(null);
              try { sessionStorage.removeItem(KEY); } catch {}
            }}>Sign out</button>
          )}
        </div>
      </header>

      <main className="homeMain">
        {!pw ? (
          <form className="card aLogin" onSubmit={(e) => { e.preventDefault(); setPw(draft); }}>
            <h2>Team DepthX</h2>
            <p>Enter the team password to see every reconstruction run on the site.</p>
            <div className="field">
              <span>Password</span>
              <input type="password" value={draft} autoFocus onChange={(e) => setDraft(e.target.value)} />
            </div>
            <button type="submit" className="btnPrimary" disabled={!draft}>Open run log</button>
            {error && <p className="errLine">{error}</p>}
          </form>
        ) : (
          <>
            <section className="aStats">
              <div><span>Runs</span><b>{stats.total}</b></div>
              <div><span>Succeeded</span><b>{stats.done}</b></div>
              <div><span>Failed</span><b>{stats.failed}</b></div>
              <div><span>Distinct IPs</span><b>{stats.people}</b></div>
              <div><span>Median inference</span><b>{stats.inference != null ? `${stats.inference.toFixed(1)} s` : 'n/a'}</b></div>
              <div><span>Median cold start</span><b>{stats.cold != null ? `${stats.cold.toFixed(1)} s` : 'n/a'}</b></div>
              <div>
                <span>GPU now</span>
                <b className={`aGpu ${data?.gpu?.state || ''}`}>{data?.gpu?.state || '...'}</b>
              </div>
              <div><span>Open pages</span><b>{data?.active_pages ?? '...'}</b></div>
            </section>

            <div className="aBar">
              <h2>Runs</h2>
              <button type="button" className="btnSecondary" onClick={() => load(pw)} disabled={loading}>
                {loading ? <span className="spinner" /> : <Icon name="replay" size={13} />} Refresh
              </button>
            </div>
            {error && <p className="errLine">{error}</p>}

            <div className="aList">
              {(data?.runs || []).map((r) => (
                <article key={r.id} className="aRow">
                  <Thumb base={base} pw={pw} run={r} onOpen={(url, run) => setBig({ url, run })} />
                  <div className="aMain">
                    <div className="aTitle">
                      <b title={r.filename}>{r.filename || r.id}</b>
                      <span className={`aStatus ${r.status}`}>{r.status}</span>
                      {r.untrained && <span className="aStatus error">test build</span>}
                    </div>
                    <div className="aMeta">
                      <span>{new Date((r.submitted_at || 0) * 1000).toLocaleString()}</span>
                      <span>{r.bytes ? `${(r.bytes / 1e6).toFixed(1)} MB` : ''}</span>
                      {r.input_size && <span>{r.input_size[0]} × {r.input_size[1]} px</span>}
                      {r.quantity && <span>{r.quantity}</span>}
                      <span>{r.georeferenced ? 'georeferenced' : 'not georeferenced'}</span>
                    </div>
                    <div className="aMeta">
                      <span>IP {r.ip}</span>
                      <span>{browser(r.user_agent)}</span>
                      {r.origin && <span>{r.origin.replace(/^https?:\/\//, '')}</span>}
                    </div>
                    {r.error && <p className="aErr">{r.error}</p>}
                  </div>
                  <div className="aTimes">
                    <div><span>GPU start</span><b>{r.status !== 'done' ? '' : r.cold ? `${r.timing?.wait_s?.toFixed(1)} s` : 'warm'}</b></div>
                    <div><span>Inference</span><b>{r.timing?.inference_s != null ? `${r.timing.inference_s.toFixed(1)} s` : ''}</b></div>
                    <div><span>Pipeline</span><b>{r.timing?.pipeline_s != null ? `${r.timing.pipeline_s.toFixed(1)} s` : ''}</b></div>
                    <button type="button" className="btnText" onClick={() => download(r)}>
                      <Icon name="download" size={13} /> Original
                    </button>
                  </div>
                </article>
              ))}
              {data && !data.runs.length && <p className="aEmpty">No runs yet.</p>}
            </div>
          </>
        )}
      </main>

      {big && (
        <div className="aScrim" onMouseDown={(e) => e.target === e.currentTarget && setBig(null)}>
          <figure className="aBig">
            <img src={big.url} alt="" />
            <figcaption>
              <b>{big.run.filename}</b>
              <span>{new Date((big.run.submitted_at || 0) * 1000).toLocaleString()} · IP {big.run.ip}</span>
              <button type="button" className="btnText" onClick={() => setBig(null)}>Close</button>
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
