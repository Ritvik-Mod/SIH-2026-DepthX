import { Analytics } from '@vercel/analytics/next';
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
      <body>
        {children}
        {/* Page views and visitors, reported to the Vercel dashboard. Inert on
            localhost and until Analytics is enabled for the project in Vercel. */}
        <Analytics />
      </body>
    </html>
  );
}
