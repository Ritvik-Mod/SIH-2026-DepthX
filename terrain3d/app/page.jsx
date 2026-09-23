'use client';

import { useState } from 'react';
import Home from '@/components/Home';
import TerrainViewer from '@/components/TerrainViewer';
import ErrorBoundary from '@/components/ErrorBoundary';
import Fallback2D from '@/components/Fallback2D';

export default function Page() {
  const [dataset, setDataset] = useState(null);

  // The boundary matters most around the viewer: if anything in the 3D path
  // throws, we still have the height data in memory, so fall back to the 2D
  // relief view rather than letting the exception blank the whole site.
  return dataset
    ? (
      <ErrorBoundary
        fallback={(err) => (
          <Fallback2D
            dataset={dataset}
            onReset={() => setDataset(null)}
            reason={err?.message || String(err)}
          />
        )}
      >
        <TerrainViewer dataset={dataset} onReset={() => setDataset(null)} />
      </ErrorBoundary>
    )
    : (
      <ErrorBoundary>
        <Home onReady={setDataset} />
      </ErrorBoundary>
    );
}
