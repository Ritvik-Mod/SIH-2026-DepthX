import './globals.css';

export const metadata = {
  title: 'DepthWizard · Team DepthX',
  description:
    'Single-image height estimation for SIH26175: one nadir satellite or aerial image in, '
    + 'a metric height map and a navigable 3D scene out.',
};

export const viewport = {
  themeColor: '#f5f6f8',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
