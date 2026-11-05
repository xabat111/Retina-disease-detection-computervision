"""Tema visual: variables CSS, tipografía (2 familias) y clases reutilizables.

Dirección estética: sobria y técnica, como un visor clínico. Las superficies son
translúcidas y el texto hereda color, así que funciona con el tema claro u oscuro de
Streamlit; las insignias de estado llevan fondo sólido propio para garantizar contraste AA.
Para recolorear la app basta con editar las variables de `:root` (y `.streamlit/config.toml`).
"""

from __future__ import annotations

import streamlit as st

# Incertidumbre: Low (seguro) = verde, Medium = ámbar, High (duda) = rojo.
# badge-high/low CSS names are historical (green/red); mapped to uncertainty levels.
UNCERTAINTY_CSS_CLASS = {"Low": "badge-high", "Medium": "badge-medium", "High": "badge-low"}

_STYLESHEET = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
  --font-ui: 'IBM Plex Sans', system-ui, 'Segoe UI', sans-serif;
  --font-number: 'IBM Plex Mono', ui-monospace, Consolas, monospace;
  --accent: #3FB8AF;
  --state-high: #5FD39A;
  --state-medium: #F2B84B;
  --state-low: #FF8A80;
  --on-state: #0B1014;
  --surface: rgba(128, 128, 128, 0.10);
  --border: rgba(128, 128, 128, 0.30);
  --muted-opacity: 0.78;
  /* escala de espaciado 4/8/16/24/32 */
  --space-1: 4px; --space-2: 8px; --space-3: 16px; --space-4: 24px; --space-5: 32px;
  --radius: 8px;
}

html, body, [class*="st-"], .stMarkdown { font-family: var(--font-ui); }
.block-container { padding-top: var(--space-5); padding-bottom: var(--space-5); max-width: 1400px; }

h1 { font-weight: 600; letter-spacing: -0.01em; margin-bottom: var(--space-1); }
h2, h3 { font-weight: 600; margin-top: var(--space-4); }
.subtitle { opacity: var(--muted-opacity); margin-bottom: var(--space-4); font-size: 1.02rem; }
.caption-muted { opacity: var(--muted-opacity); font-size: 0.85rem; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem;
           opacity: var(--muted-opacity); font-weight: 600; }

.kpi { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
       padding: var(--space-3); height: 100%; }
.kpi-value { font-family: var(--font-number); font-size: 1.6rem; line-height: 1.25;
             margin: var(--space-1) 0; overflow-wrap: anywhere; }
.kpi-note { opacity: var(--muted-opacity); font-size: 0.82rem; }

.certainty-strip { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
                   padding: var(--space-3); margin-bottom: var(--space-3); }
.certainty-chips { display: flex; flex-wrap: wrap; gap: var(--space-2); margin-top: var(--space-1); }
.certainty-chip { display: inline-flex; align-items: center; gap: var(--space-2);
                  padding: var(--space-1) var(--space-2); border: 1px solid var(--border);
                  border-radius: var(--radius); background: transparent; }
.certainty-chip-muted { opacity: 0.72; border-style: dashed; }
.certainty-chip-label { font-size: 0.95rem; font-weight: 500; }
.certainty-chip-meta { font-size: 0.78rem; opacity: var(--muted-opacity); font-weight: 500; }

.badge { display: inline-block; padding: 2px var(--space-2); border-radius: 999px;
         font-weight: 600; font-size: 0.85rem; color: var(--on-state); }
.badge-high { background: var(--state-high); }
.badge-medium { background: var(--state-medium); }
.badge-low { background: var(--state-low); }
.badge-neutral { background: var(--border); color: inherit; }

.notice { border-left: 4px solid var(--accent); background: var(--surface);
          border-radius: 0 var(--radius) var(--radius) 0; padding: var(--space-3);
          margin: var(--space-3) 0; }
.notice-warning { border-left-color: var(--state-medium); }
.notice-error { border-left-color: var(--state-low); }

.empty-state { border: 1px dashed var(--border); border-radius: var(--radius);
               padding: var(--space-5); text-align: center; }
.empty-state h3 { margin-top: 0; }

/* tablas anchas: scroll propio, nunca scroll horizontal de la página */
[data-testid="stDataFrame"] { max-width: 100%; overflow-x: auto; }
</style>
"""


def inject_styles() -> None:
    st.markdown(_STYLESHEET, unsafe_allow_html=True)


def badge_html(uncertainty: str, text: str | None = None) -> str:
    css_class = UNCERTAINTY_CSS_CLASS.get(uncertainty, "badge-neutral")
    return f'<span class="badge {css_class}">{text or uncertainty}</span>'
