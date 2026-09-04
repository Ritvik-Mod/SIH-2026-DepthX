'use client';

// A single uncaught client-side exception used to replace the whole site with
// Next's generic "Application error: a client-side exception has occurred".
// That took the upload form and the prepared-files loader down with it -- the
// two things that still work when the 3D view cannot start. This keeps the
// blast radius to the component that actually failed.

import { Component } from 'react';

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Still surface it for debugging; we just do not let it kill the page.
    console.error('[DepthX] caught by boundary:', error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    if (this.props.fallback) return this.props.fallback(this.state.error, () => this.setState({ error: null }));
    return (
      <div style={{
        position: 'fixed', inset: 0, background: '#0c0f13', color: '#e8edf2',
        display: 'flex', flexDirection: 'column', gap: 12,
        alignItems: 'center', justifyContent: 'center', padding: 24,
        font: '13px/1.6 ui-sans-serif, system-ui, sans-serif', textAlign: 'center',
      }}>
        <div style={{ fontWeight: 600 }}>Something in the viewer failed.</div>
        <div style={{ opacity: 0.7, maxWidth: 560, fontFamily: 'ui-monospace, monospace',
                      fontSize: 11 }}>
          {this.state.error?.message || String(this.state.error)}
        </div>
        <button
          onClick={() => this.setState({ error: null })}
          style={{
            marginTop: 6, padding: '7px 16px', borderRadius: 999, cursor: 'pointer',
            border: '1px solid rgba(95,178,255,0.55)', background: 'rgba(95,178,255,0.16)',
            color: '#8ecbff', fontSize: 12,
          }}>
          try again
        </button>
      </div>
    );
  }
}
