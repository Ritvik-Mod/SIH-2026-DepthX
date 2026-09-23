/**
 * DepthWizard wordmark. The mark is three stacked height layers seen in
 * isometric -- the product in one glyph: a flat image lifted into strata.
 */
export function BrandMark({ size = 22 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="brandMark">
      <path d="M12 3 21 8l-9 5-9-5z" fill="var(--mark-top)" />
      <path d="M3 12l9 5 9-5" fill="none" stroke="var(--mark-mid)" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 16l9 5 9-5" fill="none" stroke="var(--mark-low)" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function Brand({ sub = true }) {
  return (
    <span className="brand">
      <BrandMark />
      <span className="brandText">
        <b>DepthWizard</b>
        {sub && <small>Team DepthX</small>}
      </span>
    </span>
  );
}
