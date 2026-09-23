'use client';

import { useState } from 'react';
import SAMPLES from '@/lib/samples.json';
import { loadSample } from '@/lib/load';
import Icon from './Icons';

/**
 * Precomputed scenes from final_test_files/, run through the model once and
 * served as static files (scripts/package_samples.py). Opening one goes through
 * the same loaders as an upload -- it just skips the minute of inference, and
 * works with no model service reachable at all.
 */
export default function SampleGallery({ onReady }) {
  const [loading, setLoading] = useState(null);
  const [progress, setProgress] = useState('');
  const [error, setError] = useState('');

  const open = async (s) => {
    if (loading) return;
    setError('');
    setLoading(s.id);
    try {
      onReady(await loadSample(s, setProgress));
    } catch (e) {
      console.error(e);
      setError(`${s.name}: ${e?.message || 'could not load'}`);
      setLoading(null);
    }
  };

  return (
    <section className="gallery" aria-labelledby="samples-title">
      <div className="galleryHead">
        <h2 id="samples-title">Sample scenes</h2>
        <span>Precomputed with the live model · open instantly · no upload needed</span>
      </div>

      <div className="sampleGrid">
        {SAMPLES.map((s) => (
          <button
            key={s.id}
            type="button"
            className={`sample ${loading === s.id ? 'isLoading' : ''}`}
            onClick={() => open(s)}
            disabled={!!loading && loading !== s.id}
            aria-busy={loading === s.id}
          >
            <span className="sampleThumb">
              <img src={`${s.base}/thumb.jpg`} alt="" loading="lazy" draggable={false} />
              <span className={`sampleBadge ${s.quantity === 'DSM' ? 'dsm' : ''}`}>
                {s.quantity === 'DSM' ? 'DSM · terrain' : 'AGL'}
              </span>
              {loading === s.id && (
                <span className="sampleLoading"><span className="spinner" />{progress || 'Opening…'}</span>
              )}
            </span>
            <span className="sampleBody">
              <b>{s.name}</b>
              <small>{s.place}</small>
              <span className="sampleMeta">
                <span>{s.extent[0]} × {s.extent[1]} m</span>
                <span>{s.relief} m relief</span>
                <span>{s.gsd.toFixed(2)} m/px</span>
              </span>
              <span className={`sampleGeo ${s.georeferenced ? 'yes' : ''}`}>
                <Icon name="globe" size={12} />
                {s.georeferenced ? s.crs : 'Not georeferenced'}
              </span>
            </span>
          </button>
        ))}
      </div>

      {error && <p className="errLine">{error}</p>}
    </section>
  );
}
