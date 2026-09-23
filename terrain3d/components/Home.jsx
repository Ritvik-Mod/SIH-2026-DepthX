'use client';

import Brand from './Brand';
import UploadPanel from './UploadPanel';
import SampleGallery from './SampleGallery';

const STEPS = [
  {
    title: 'Open a sample',
    body: 'Pick any scene below. It was reconstructed ahead of time, so it opens in seconds and needs no model service.',
  },
  {
    title: 'Or upload your own',
    body: 'Drop a top-down GeoTIFF, PNG or JPEG. A georeferenced GeoTIFF keeps its real scale and gets terrain added; other images assume 0.33 m/px.',
  },
  {
    title: 'Generate',
    body: 'The image is sent to the DepthX model service. Reconstruction takes about 30–90 seconds depending on size.',
  },
  {
    title: 'Explore and compare',
    body: 'Drag to orbit, scroll to zoom, or switch to Fly / Walk. Toggle layers from the top bar, and click the source image in the corner to compare photo and height.',
  },
];

export default function Home({ onReady }) {
  return (
    <div className="home">
      <header className="topbar">
        <Brand />
        <div className="topMeta">
          <span>Smart India Hackathon 2026</span>
          <span className="psChip">SIH26175</span>
        </div>
      </header>

      <main className="homeMain">
        <section className="hero">
          <p className="eyebrow">Single-image height estimation</p>
          <h1>One overhead image in. A metric 3D scene out.</h1>
          <p className="lede">
            DepthWizard estimates the height of every pixel, in metres, from a single nadir
            satellite or aerial image — then places it on real terrain and renders it as a scene
            you can fly through.
          </p>
          <dl className="specs">
            <div><dt>Model</dt><dd>DINOv2 ViT-L + DPT</dd></div>
            <div><dt>Output</dt><dd>Height in metres (AGL / DSM)</dd></div>
            <div><dt>Terrain</dt><dd>Copernicus GLO-30 DTM</dd></div>
          </dl>
        </section>

        <div className="homeGrid">
          <UploadPanel onReady={onReady} />

          <section className="steps" aria-labelledby="steps-title">
            <h2 id="steps-title">How to test</h2>
            <ol>
              {STEPS.map((s, i) => (
                <li key={s.title}>
                  <span className="stepNum">{i + 1}</span>
                  <span>
                    <b>{s.title}</b>
                    <small>{s.body}</small>
                  </span>
                </li>
              ))}
            </ol>
          </section>
        </div>

        <SampleGallery onReady={onReady} />
      </main>

      <footer className="homeFoot">
        <span>Team DepthX</span>
        <span>DepthWizard v0.1</span>
        <span>Problem statement SIH26175</span>
      </footer>
    </div>
  );
}
