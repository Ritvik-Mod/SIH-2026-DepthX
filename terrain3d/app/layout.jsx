import './globals.css';

export const metadata = {
  title: '2D → 3D Terrain Viewer',
  description: 'Interactive fly-through of a DSM reconstructed from a single overhead image',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
