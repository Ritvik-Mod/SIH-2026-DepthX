'use client';

import { useState } from 'react';
import UploadPanel from '@/components/UploadPanel';
import TerrainViewer from '@/components/TerrainViewer';

export default function Page() {
  const [dataset, setDataset] = useState(null);
  return dataset
    ? <TerrainViewer dataset={dataset} onReset={() => setDataset(null)} />
    : <UploadPanel onReady={setDataset} />;
}
