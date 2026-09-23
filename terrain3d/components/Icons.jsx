/**
 * One small stroke-icon set, drawn on a 16 px grid.
 *
 * Inline SVG rather than an icon font or unicode glyphs: the old panel used
 * characters like ◇ ▦ ▧ ◒, which render differently on every OS and at every
 * size, and never line up with the text next to them.
 */
const PATHS = {
  windows: (
    <>
      <rect x="3" y="2.5" width="10" height="11" rx="1" />
      <path d="M6 5.5h1M9 5.5h1M6 8h1M9 8h1M6 10.5h1M9 10.5h1" />
    </>
  ),
  edges: <path d="M2 11.5 6 5l3.5 4.5L12 7l2 4.5M2 13.5h12" />,
  trees: (
    <>
      <path d="M8 2.5 4.5 8h2L4 11.5h8L9.5 8h2z" />
      <path d="M8 11.5v2.5" />
    </>
  ),
  level: <path d="M3 5c1.5-2 3.5-2 5 0s3.5 2 5 0M8 7.5v4.5M5.5 10 8 12.5 10.5 10M2.5 14h11" />,
  hud: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <path d="M5 6.5h3M5 9.5h6" />
    </>
  ),
  orbit: (
    <>
      <circle cx="8" cy="8" r="2" />
      <path d="M8 2.5a5.5 5.5 0 1 1-5.2 3.7" />
      <path d="M1.8 3.6 2.8 6.2l2.6-.8" />
    </>
  ),
  fly: <path d="M2 9.5 14 4 10.5 13 8.5 9.5zM8.5 9.5 14 4" />,
  walk: (
    <>
      <circle cx="8.5" cy="3" r="1.2" />
      <path d="m6 14 1.5-4.5L9.5 11v3M7.5 9.5 8 6l2.5 2 2 .5M8 6 5.5 7.5 5 10" />
    </>
  ),
  replay: <path d="M2.5 8a5.5 5.5 0 1 0 1.6-3.9M2.5 2.5v3h3" />,
  download: <path d="M8 2.5v8M4.5 7 8 10.5 11.5 7M3 13.5h10" />,
  back: <path d="M9.5 3.5 5 8l4.5 4.5M5 8h8" />,
  chevronLeft: <path d="M10 3.5 5.5 8l4.5 4.5" />,
  chevronRight: <path d="M6 3.5 10.5 8 6 12.5" />,
  expand: <path d="M9.5 2.5h4v4M6.5 13.5h-4v-4M13.5 2.5 9 7M2.5 13.5 7 9" />,
  close: <path d="m3.5 3.5 9 9M12.5 3.5l-9 9" />,
  plus: <path d="M8 3v10M3 8h10" />,
  minus: <path d="M3 8h10" />,
  fit: <path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" />,
  sun: (
    <>
      <circle cx="8" cy="8" r="2.6" />
      <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1 1M11.6 11.6l1 1M3.4 12.6l1-1M11.6 4.4l1-1" />
    </>
  ),
  image: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="1.5" />
      <path d="m2.5 11.5 3.5-3.5 3 3 2-2 2.5 2.5" />
      <circle cx="10.5" cy="6" r="1" />
    </>
  ),
  upload: <path d="M8 10.5v-8M4.5 6 8 2.5 11.5 6M3 10.5v3h10v-3" />,
  folder: <path d="M2 4.5V12a1 1 0 0 0 1 1h10a1 1 0 0 0 1-1V6a1 1 0 0 0-1-1H8L6.5 3.5H3a1 1 0 0 0-1 1z" />,
  globe: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <path d="M2.5 8h11M8 2.5c1.6 1.6 2.3 3.4 2.3 5.5S9.6 11.9 8 13.5M8 2.5C6.4 4.1 5.7 5.9 5.7 8s.7 3.9 2.3 5.5" />
    </>
  ),
  arrowRight: <path d="M3 8h10M9 4l4 4-4 4" />,
  layers: <path d="M8 2.5 14 6 8 9.5 2 6zM2 9l6 3.5L14 9" />,
};

export default function Icon({ name, size = 16, className, strokeWidth = 1.4 }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {PATHS[name] || null}
    </svg>
  );
}
