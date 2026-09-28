import M from '@/lib/metrics.json';

/**
 * Measured performance, laid out against the two SIH26175 evaluation criteria.
 *
 * Every number comes from lib/metrics.json, built by scripts/make_site_metrics.py.
 * The accuracy figures come from results/validation.json (the deployed model scored
 * against swisstopo LiDAR by scripts/evaluate_live.py) and results/offline_eval.json
 * (GAMUS test strata from the training cluster). The rendering and
 * deployment facts are measurements from tests against the live service, each noted
 * in that script with how it was measured.
 */
function Bar({ value, max }) {
  const w = Math.max(3, Math.min(100, (value / max) * 100));
  return <span className="mBar" aria-hidden="true"><i style={{ width: `${w}%` }} /></span>;
}

export default function Metrics() {
  if (!M?.headline?.length || !M?.landscapes?.length) return null;
  const maxMae = Math.max(...M.landscapes.map((l) => l.mae));
  return (
    <section className="metrics" aria-labelledby="metrics-title">
      <div className="galleryHead">
        <h2 id="metrics-title">Measured performance</h2>
        <span>{M.subtitle}</span>
      </div>

      <div className="kpiGrid">
        {M.headline.map((k) => (
          <div key={k.label} className="kpi">
            <span className="kpiLabel">{k.label}</span>
            <b className="kpiValue">{k.value}<small>{k.unit}</small></b>
            <span className="kpiNote">{k.note}</span>
          </div>
        ))}
      </div>

      <div className="mGrid">
        <div className="mCard">
          <div className="mCardHead">
            <span className="mCrit">Criterion 1 · 50%</span>
            <h3>DSM estimation: accuracy across landscapes</h3>
          </div>
          <table className="mTable">
            <thead>
              <tr><th>Landscape</th><th>MAE</th><th>RMSE</th><th className="mSym" title="Pearson correlation">r</th><th aria-hidden="true" /></tr>
            </thead>
            <tbody>
              {M.landscapes.map((l) => (
                <tr key={l.name}>
                  <td><b>{l.name}</b><small>{l.detail}</small></td>
                  <td>{l.mae.toFixed(2)} m</td>
                  <td>{l.rmse.toFixed(2)} m</td>
                  <td>{l.r.toFixed(3)}</td>
                  <td><Bar value={l.mae} max={maxMae} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mDsm">
            {M.dsm.map((d) => (
              <div key={d.name}>
                <span>{d.name}</span>
                <b>{d.mae.toFixed(2)} m MAE · {d.rmse.toFixed(2)} m RMSE · r {d.r.toFixed(3)}</b>
                <small>{d.note}</small>
              </div>
            ))}
          </div>
        </div>

        <div className="mCard">
          <div className="mCardHead">
            <span className="mCrit">Criterion 2 · 50%</span>
            <h3>Visualization: rendering, experience, deployment</h3>
          </div>
          <dl className="mList">
            {M.rendering.map((r) => (
              <div key={r.label}>
                <dt>{r.label}</dt>
                <dd><b>{r.value}</b><small>{r.note}</small></dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      <p className="mFoot">{M.footnote}</p>
    </section>
  );
}
