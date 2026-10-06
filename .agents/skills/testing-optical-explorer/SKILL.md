---
name: testing-optical-explorer
description: Browser verification of the standalone optical explorer, synchronized signal controls, responsive framing, CSV, and offline single-file build.
---

# Optical explorer runtime testing

## Setup
- Work in `optical-explorer`, separate from the Python simulator.
- Use a compatible Node version (22.18+ or 24+), run `npm ci` if needed.
- Start `npm run dev -- --host 0.0.0.0`; Vite normally uses port 5173.
- No backend, login, feature flag or OpenFOAM installation is needed.
- `npm run build` creates the standalone `dist/index.html`. Use a current build
  when checking offline behavior; do not duplicate checks owned by another agent.

## Devin Secrets Needed
None.

## High-signal runtime flow
- Jump to phase 4 (20 seconds), select Keyhole, play and pause. Match the
  dimension tag to Live depth, and inspect cutaway/graph separation visually.
- Compare paused screenshots after camera damping settles to check that beam
  particles, keyhole and signal stop, not just the displayed timestamp.
- Drag backward while holding the mouse button. Capture before release:
  playback should pause and both graph and intensity history must shorten.
- Reset or scrub before phase 4: history clears and CSV export is unavailable.
- CSV exports visible history only. Full acquisition has 241 samples from
  0 through 120ms in 0.5ms steps. Header:
  `time_ms,depth_um,trailing_10ms_mean_um`.
- Test Overview, Scanner and Keyhole at desktop and 390x844. Inspect actual
  pixels for label/toolbar overlap; DOM presence does not prove visibility.
- Test camera orbit, right-drag pan and wheel zoom separately from presets.
- Help is opened by “Controls & a few notes”; verify close-button focus,
  Tab trapping, Escape dismissal and trigger-focus restoration.

## Offline and console verification
- Open the built HTML with `file://`, select Offline in DevTools Network,
  disable cache and reload. Keep DevTools open for the offline check.
- Verify `navigator.onLine === false`; instrumentation from a separate CDP
  session may be reset or overridden when DevTools or navigation changes state.
- Network should show the local HTML only and no external subresources.
  Exercise phase-4 playback and scrub, not just initial document rendering.
- Restore browser network settings after testing.
- Separate real app exceptions or missing assets from environment warnings.
  Software WebGL can report SwiftShader fallback deprecation and ReadPixels
  performance warnings without an application failure.
