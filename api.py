from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse

from config import Settings
from state import ALL_TYPES, PersonStateStore
from scorer import BODY_TYPES
from face_scorer import FACE_TYPES


MJPEG_BOUNDARY = "frame"


def create_app(store: PersonStateStore, settings: Settings) -> FastAPI:
    app = FastAPI(title="Seizure Detection", version="0.2.0")

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        h = store.health()
        return JSONResponse(
            {
                "ok": h.ok,
                "fps": round(h.fps, 2),
                "model": h.model,
                "uptime_seconds": round(h.uptime_seconds, 1),
                "tracked_people": h.tracked_people,
                "types": list(ALL_TYPES),
            }
        )

    @app.get("/people")
    def people() -> JSONResponse:
        snapshot, fps = store.snapshot()
        now = time.time()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fps": round(fps, 2),
            "alert_threshold": settings.alert_threshold,
            "types": {"body": list(BODY_TYPES), "face": list(FACE_TYPES)},
            "people": [
                {
                    "track_id": p.track_id,
                    "bbox": [round(v, 1) for v in p.bbox],
                    "seizure_probabilities": {
                        k: round(v, 4) for k, v in p.probabilities.items()
                    },
                    "max_probability": (
                        round(max(p.probabilities.values()), 4)
                        if p.probabilities
                        else None
                    ),
                    "dominant_type": (
                        max(p.probabilities, key=lambda k: p.probabilities[k])
                        if p.probabilities
                        else None
                    ),
                    "features": {k: round(v, 4) for k, v in p.features.items()},
                    "window_seconds": round(p.window_seconds, 2),
                    "last_seen_seconds_ago": round(now - p.last_seen_ts, 3),
                }
                for p in sorted(snapshot, key=lambda x: x.track_id)
            ],
        }
        return JSONResponse(payload)

    @app.get("/history")
    def history() -> JSONResponse:
        hist = store.history_snapshot()
        _, fps = store.snapshot()
        now = time.time()
        types = list(ALL_TYPES)
        return JSONResponse(
            {
                "now": round(now, 3),
                "fps": round(fps, 2),
                "window_seconds": store.history_seconds,
                "alert_threshold": settings.alert_threshold,
                "types": types,
                "series": [
                    {
                        "track_id": tid,
                        "points": [
                            [round(t, 3)] + [round(probs.get(k, 0.0), 4) for k in types]
                            for t, probs in points
                        ],
                    }
                    for tid, points in sorted(hist.items())
                ],
            }
        )

    @app.get("/video.mjpg")
    def video() -> StreamingResponse:
        async def gen():
            last_version = -1
            loop = asyncio.get_event_loop()
            while True:
                jpeg, last_version = await loop.run_in_executor(
                    None, store.wait_for_new_jpeg, last_version, 1.0
                )
                if jpeg is None:
                    continue
                yield (
                    b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )

        return StreamingResponse(
            gen(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )

    @app.get("/", include_in_schema=False)
    def root() -> Response:
        return Response(content=DASHBOARD_HTML, media_type="text/html")

    return app


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Seizure Detection — live</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #0e0e10; color: #e6e6e6; margin: 0; padding: 14px; }
  header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  h1 { font-size: 18px; margin: 0; }
  #meta { font-family: ui-monospace, monospace; font-size: 12px; color: #888; }
  .links { font-size: 12px; color: #71717a; margin-left: auto; }
  .links a { color: #93c5fd; text-decoration: none; margin-right: 10px; }
  .layout { display: grid; grid-template-columns: minmax(320px, 480px) minmax(0, 1fr); gap: 14px; }
  @media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
  .panel { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; }
  .panel h2 { font-size: 12px; margin: 0 0 8px; color: #a1a1aa; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
  img.stream { width: 100%; display: block; border-radius: 4px; background: #000; }
  #people-panels { display: flex; flex-direction: column; gap: 12px; }
  .person { background: #18181b; border: 1px solid #27272a; border-radius: 8px; padding: 12px; }
  .person-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px; flex-wrap: wrap; }
  .person-head .id { font-weight: 700; font-size: 14px; }
  .person-head .dom { font-family: ui-monospace, monospace; font-size: 12px; color: #a1a1aa; }
  .person-head .dom.hot { color: #f87171; font-weight: 700; }
  canvas { width: 100%; height: 220px; display: block; }
  .legend { font-family: ui-monospace, monospace; font-size: 11px; display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 4px 12px; margin-top: 8px; }
  .legend .item { cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; opacity: 1; }
  .legend .item.off { opacity: 0.35; text-decoration: line-through; }
  .legend .sw { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
  .legend .item.hot { color: #f87171; font-weight: 700; }
  .empty { color: #52525b; font-size: 13px; text-align: center; padding: 18px; }
</style>
</head>
<body>
<header>
  <h1>Seizure Detection — live</h1>
  <div id="meta">connecting…</div>
  <div class="links">
    <a href="/healthz">/healthz</a>
    <a href="/people">/people</a>
    <a href="/history">/history</a>
    <a href="/video.mjpg">/video.mjpg</a>
  </div>
</header>
<div class="layout">
  <div class="panel">
    <h2>Annotated stream</h2>
    <img class="stream" src="/video.mjpg" alt="annotated video stream"/>
  </div>
  <div>
    <div id="people-panels"><div class="empty">no people detected</div></div>
  </div>
</div>
<script>
(() => {
  const TYPE_COLORS = {
    tonic_clonic:      '#ef4444',
    clonic:            '#f97316',
    myoclonic:         '#ec4899',
    atonic:            '#8b5cf6',
    focal_motor:       '#f59e0b',
    versive:           '#06b6d4',
    eyelid_myoclonia:  '#3b82f6',
    oral_automatism:   '#10b981',
    hemifacial_clonic: '#14b8a6',
  };
  const peopleEl = document.getElementById('people-panels');
  const metaEl = document.getElementById('meta');
  let WINDOW = 120;
  let THRESHOLD = 0.6;
  let TYPES = [];
  const panels = new Map();           // track_id -> {root, canvas, ctx, legend, head, hidden:Set}

  function fmt(p) { return p === null || p === undefined ? '—' : p.toFixed(2); }

  function setupHiDPI(canvas, ctx) {
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 600;
    const cssH = canvas.clientHeight || 220;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function drawChart(panel, points, now) {
    const { canvas, ctx, hidden } = panel;
    const W = canvas.clientWidth, H = canvas.clientHeight;
    ctx.clearRect(0, 0, W, H);
    const padL = 30, padR = 8, padT = 6, padB = 18;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.font = '10px ui-monospace, monospace';
    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#71717a';
    for (let yv = 0; yv <= 1.001; yv += 0.25) {
      const y = padT + plotH * (1 - yv);
      ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(W - padR, y); ctx.stroke();
      ctx.fillText(yv.toFixed(2), 2, y + 3);
    }
    for (let xs = 0; xs <= WINDOW; xs += 30) {
      const x = padL + plotW * (1 - xs / WINDOW);
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + plotH); ctx.stroke();
      const lbl = xs === 0 ? '0' : '-' + xs;
      ctx.fillText(lbl, x - lbl.length * 3, H - 4);
    }
    ctx.strokeStyle = '#ef4444';
    ctx.setLineDash([3, 3]);
    const yT = padT + plotH * (1 - THRESHOLD);
    ctx.beginPath(); ctx.moveTo(padL, yT); ctx.lineTo(W - padR, yT); ctx.stroke();
    ctx.setLineDash([]);

    if (!points || !points.length) return;

    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    for (let ti = 0; ti < TYPES.length; ti++) {
      const t = TYPES[ti];
      if (hidden.has(t)) continue;
      ctx.strokeStyle = TYPE_COLORS[t] || '#999';
      ctx.beginPath();
      let started = false;
      for (const row of points) {
        const dt = now - row[0];
        if (dt > WINDOW || dt < 0) continue;
        const p = row[ti + 1] || 0;
        const x = padL + plotW * (1 - dt / WINDOW);
        const y = padT + plotH * (1 - Math.max(0, Math.min(1, p)));
        if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
      }
      ctx.stroke();
    }
  }

  function ensurePanel(trackId) {
    if (panels.has(trackId)) return panels.get(trackId);
    const root = document.createElement('div');
    root.className = 'person';
    root.innerHTML = `
      <div class="person-head">
        <span class="id" style="color:#fff">id ${trackId}</span>
        <span class="dom">—</span>
      </div>
      <canvas></canvas>
      <div class="legend"></div>
    `;
    peopleEl.appendChild(root);
    const canvas = root.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    const panel = {
      root,
      canvas,
      ctx,
      legend: root.querySelector('.legend'),
      head: root.querySelector('.dom'),
      hidden: new Set(),
    };
    setupHiDPI(canvas, ctx);
    panels.set(trackId, panel);
    return panel;
  }

  function removePanel(trackId) {
    const panel = panels.get(trackId);
    if (!panel) return;
    panel.root.remove();
    panels.delete(trackId);
  }

  function renderLegend(panel, points) {
    const last = points && points.length ? points[points.length - 1] : null;
    let dominantType = null, dominantP = -1;
    panel.legend.innerHTML = TYPES.map((t, i) => {
      const p = last ? (last[i + 1] || 0) : 0;
      if (p > dominantP) { dominantP = p; dominantType = t; }
      const isHot = p > THRESHOLD;
      const isOff = panel.hidden.has(t);
      return `<span class="item${isOff ? ' off' : ''}${isHot ? ' hot' : ''}" data-type="${t}">
        <span class="sw" style="background:${TYPE_COLORS[t] || '#999'}"></span>
        <span style="flex:1">${t.replace(/_/g, ' ')}</span>
        <span>${p.toFixed(2)}</span>
      </span>`;
    }).join('');
    panel.legend.querySelectorAll('.item').forEach(el => {
      el.onclick = () => {
        const t = el.dataset.type;
        if (panel.hidden.has(t)) panel.hidden.delete(t); else panel.hidden.add(t);
        el.classList.toggle('off');
      };
    });
    if (dominantType !== null) {
      const hot = dominantP > THRESHOLD;
      panel.head.className = 'dom' + (hot ? ' hot' : '');
      panel.head.textContent = `dominant: ${dominantType.replace(/_/g, ' ')} = ${dominantP.toFixed(2)}` + (hot ? '  ⚠' : '');
    } else {
      panel.head.textContent = '—';
    }
  }

  async function tick() {
    try {
      const r = await fetch('/history', { cache: 'no-store' });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const j = await r.json();
      WINDOW = j.window_seconds || WINDOW;
      THRESHOLD = j.alert_threshold ?? THRESHOLD;
      TYPES = j.types || [];
      const series = j.series || [];

      // remove panels whose tracks vanished
      const liveIds = new Set(series.map(s => s.track_id));
      for (const tid of Array.from(panels.keys())) {
        if (!liveIds.has(tid)) removePanel(tid);
      }
      // remove "empty" placeholder when we have people
      if (series.length && peopleEl.querySelector('.empty')) peopleEl.innerHTML = '';
      if (!series.length && !peopleEl.querySelector('.empty')) {
        peopleEl.innerHTML = '<div class="empty">no people detected</div>';
      }

      for (const s of series) {
        const panel = ensurePanel(s.track_id);
        drawChart(panel, s.points, j.now);
        renderLegend(panel, s.points);
      }
      metaEl.textContent = `fps=${j.fps.toFixed(1)} • tracked=${series.length} • ${new Date().toLocaleTimeString()}`;
    } catch (err) {
      metaEl.textContent = 'fetch error: ' + err.message;
    }
  }

  window.addEventListener('resize', () => {
    for (const p of panels.values()) setupHiDPI(p.canvas, p.ctx);
  });
  tick();
  setInterval(tick, 500);
})();
</script>
</body>
</html>
"""
