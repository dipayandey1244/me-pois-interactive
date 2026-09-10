# 🧬 Research Work — Replicable Prompt Template

A master prompt you can copy-paste to build any interactive research paper walkthrough
in the exact same format as the ME-POIs "Research Work" publication.

---

## ✦ MASTER PROMPT (Copy This Verbatim)

```
Build me a single-file interactive HTML research publication called "Research Work"
replicating the Distill.pub academic style for the paper:

  PAPER TITLE: [INSERT PAPER TITLE HERE]
  PAPER URL / BLOG: [INSERT GOOGLE RESEARCH OR ARXIV URL HERE]

The page must follow ALL of the design rules, structure, and technical specifications below.

═══════════════════════════════════════════
SECTION 1: VISUAL DESIGN SYSTEM & THEME
═══════════════════════════════════════════

Color Palette (dark navy + accent):
  - Background: #fefefe
  - Dark body text: #1e293b
  - Section accent labels: #64748b (uppercase, 0.7rem, letter-spaced)
  - Primary blue: #4285F4 (hyperlinks, callout accents)
  - Green success: #22c55e
  - Warning amber: #f59e0b
  - Error red: #ef4444

Header Bar:
  - Background gradient: linear-gradient(135deg, #1a1a2e, #16213e)
  - Logo left: "Research Work" in Inter 800 weight, white, links to Home (#)
  - Nav links right: all steps as anchor links, flex row, white 70% opacity
  - Sticky top, flex-wrap: nowrap, single row enforced

Typography:
  - Body: 'Source Serif 4' (serif), 18px, line-height 1.85
  - Headings + UI: 'Inter' (sans-serif), weight 800 for h1, 700 for h2
  - Code: 'Roboto Mono', 0.8rem, line-height 1.7
  - Hand-drawn notes: 'Caveat' (cursive), orange #b45309, border-left 3px dashed
  Google Fonts: Source Serif 4, Inter, Roboto Mono, Caveat (all loaded from CDN)

Favicon:
  favicon.svg — dark navy gradient circle (#1a1a2e → #16213e) with paper-relevant icon
  Add <link rel="icon" type="image/svg+xml" href="favicon.svg"> in <head>

Page Title:
  <title>Research Work · [SHORT PAPER TITLE]</title>

KaTeX Math Rendering:
  Load KaTeX 0.16.9 (CDN) with auto-render for $$...$$ (display) and $...$ (inline)
  IMPORTANT: Never use $$ in prose — use &dollar; for currency symbols to avoid conflicts

═══════════════════════════════════════════
SECTION 2: UI COMPONENTS REQUIRED
═══════════════════════════════════════════

① HEADER (sticky top bar):
  <header class="site-header">
    <div class="wide">
      <div class="logo"><a href="#">Research Work</a></div>
      <nav> [one <a> per step] </nav>
    </div>
  </header>

② HERO SECTION:
  Full-width dark banner with:
    - Bold h1 (2 lines max, centered)
    - Subtitle in opacity-70 text
    - Meta line: "Based on [PAPER LINK in blue] · Replicated in PyTorch · By [NAME] · [YEAR]"

③ TABLE OF CONTENTS (toc):
  Grid of clickable toc-items linking to each step, numbered circles

④ ARTICLE SECTIONS (.article-section):
  Each step starts with:
    <div class="section-label">Step N of M</div>
    <h2>...</h2>
  Background alternates: #fff and #f8fafc

⑤ HAND-DRAWN ANALOGY NOTES (.hand-note):
  Background: #fffbeb, border-left: 3px dashed #f59e0b
  Font: Caveat 1.05rem, starts with ✏️ emoji
  Purpose: plain-English, 10-year-old friendly explanations

⑥ SIMPLE CALLOUT BOXES (.simple-box):
  Dashed border, light background, .heading div in Inter 700
  Use for: "What just happened?", "Why this matters", key insights

⑦ GREEN/BLUE CALLOUT (.callout.green):
  Background: #f0fdf4, border: 1px solid #bbf7d0
  Use for: algorithm steps, process explanations

⑧ STEP CARDS (.step-card):
  Border-left 3px solid color, flex row with number circle + content
  Use for: numbered procedures, benchmark comparisons

⑨ MACOS TERMINAL CODE WINDOWS (.terminal-window):
  Structure:
    <div class="terminal-window">
      <div class="terminal-header">
        <div class="terminal-left">
          <div class="terminal-dots">
            <span class="terminal-dot red"></span>
            <span class="terminal-dot yellow"></span>
            <span class="terminal-dot green"></span>
          </div>
          <span class="terminal-title">filename.py</span>
        </div>
        <div class="terminal-right">
          <span class="terminal-lang">PyTorch 2.1</span>
          <button class="copy-btn" onclick="copyCode(this)">📋 Copy Code</button>
        </div>
      </div>
      <pre class="terminal-body"><code>... SYNTAX-HIGHLIGHTED CODE ...</code></pre>
    </div>

  Background: #0d1117 (GitHub Dark), header: #161b22, border: #30363d
  Syntax colors:
    .kw (keywords): #ff7b72
    .cl (class names): #79c0ff
    .fn (functions): #d2a8ff
    .cm (comments): #8b949e italic
    .st (strings): #a5d6ff
    .num (numbers): #79c0ff
  Copy button: calls window.copyCode(btn) which copies innerText to clipboard
              with 2s ✅ Copied! success animation

⑩ MATH FORMULAS (KaTeX block):
  <div class="math-display" style="text-align:center; padding:16px; line-height:2.0;">
    $$\mathcal{L}_{\text{total}} = ...$$
  </div>
  Display math always inside .math-display div wrapper

⑪ RESULTS TABLE (.results-table):
  thead th: background #f1f5f9, text #475569, uppercase 0.68rem
  Best values highlighted: background #f0fdf4, color #166534, font-weight 700
  Striped rows: odd rows slightly darker

⑫ INTERACTIVE LEAFLET MAP (#leaflet-map):
  Load Leaflet 1.9.4 (CDN)
  Height: 450px, CARTO Light basemap tiles
  Circle markers with popup showing venue details
  GNN edge lines drawn between venue markers within radius slider value

⑬ PLOTLY CHARTS:
  Load Plotly 2.27.0 (CDN)
  Charts needed: Benchmark comparison bars, Training loss curves, Category pie charts
  Style: Inter font, clean white background, subtle gridlines

⑭ DYNAMIC CITY EXPLORER (OpenStreetMap):
  Fetch real POI data from Overpass API for any city worldwide
  Nominatim geocoding for custom city input
  Simulate ME-POIs classification predictions on live data
  Show: Map + Category pie + Price histogram + Closure risk + F1 comparison

⑮ FOOTER:
  "Research Replication of [PAPER LINK] · Built with PyTorch, ... · [NAME] · [YEAR]"
  Attribution: "POI data © OpenStreetMap contributors (ODbL)"

═══════════════════════════════════════════
SECTION 3: NARRATIVE FORMAT (8 STEPS)
═══════════════════════════════════════════

Build exactly these 8 steps as progressive article sections:

STEP 1 — THE PROBLEM
  • State the real-world failure of existing text-only methods
  • Use a concrete before/after example (e.g., review text vs. actual venue behavior)
  • Include a hand-drawn-style illustration (generate as PNG via image tool)
  • Use the ✏️ hand-note to explain like a 10-year-old

STEP 2 — THE BIG IDEA
  • Explain what 3 modalities/inputs the paper fuses
  • Show a simple visual diagram of the architecture
  • Use the restaurant/coffee shop analogy for text, mobility, spatial

STEP 3 — HOW IT WORKS
  • 3 "Brain" explainer cards (one per encoder/module)
  • Each card: what it reads → what it produces → plain English analogy
  • Example: "Brain 2 reads foot traffic like a heart-rate monitor"

STEP 4 — THE CODE (PyTorch Implementation)
  • 2 Terminal Windows showing the key model classes with full comments
  • Each class has: ASCII box diagrams, tensor shape tracking, "What just happened?" box
  • Annotate every line with inline comments explaining purpose

STEP 5 — THE RESULTS & METRICS EXPLAINER
  • Full benchmark table comparing all model variants
  • Metrics Explainer section: F1 (Doctor analogy), Precision, Recall, IoU, AUROC, MAE, RMSE
  • Per-row breakdown: what each result means in plain English
  • Hand-note with the key insight

STEP 6 — TRY IT LIVE (Interactive Proof)
  • Leaflet map with real/simulated POI markers
  • GNN radius slider that redraws edge connections in real time
  • Training loss curve chart (all variants)
  • Benchmark comparison Plotly bar chart
  • POI inspector with 168-hour mobility fingerprint chart

STEP 7 — DYNAMIC CITY EXPLORER
  • City dropdown (presets: SF, NYC, London, Paris, Tokyo, Mumbai, etc.)
  • Custom city search via Nominatim geocoding
  • Overpass API call to fetch real OSM venues
  • 4 Plotly charts + clickable results table
  • Explainer box: How synthetic dataset was generated + How each prediction (Price, Closure Risk, Peak Hour) is made

STEP 8 — HOW TO TRAIN IT YOURSELF
  • "YES, we ran the full PyTorch training loop on our own environment"
  • Step-by-step: data prep → text embedding → mobility normalization → GNN → multi-task loss
  • Multi-task loss shown as compiled KaTeX equations (two-line formulation)
  • Full train_modelname.py script shown in Terminal Window with copy button
  • Actual training output log shown as text

═══════════════════════════════════════════
SECTION 4: DATA TRANSPARENCY SECTION
═══════════════════════════════════════════

Add a Data Sources section with clearly labeled:
  ✅ OpenStreetMap (ODbL) — open source
  ✅ Nominatim geocoding — open source
  ✅ CARTO basemaps — open source
  ⚠️ Original paper's proprietary training data — NOT open source
  Explanation of what is simulated vs. real

═══════════════════════════════════════════
SECTION 5: JAVASCRIPT REQUIREMENTS
═══════════════════════════════════════════

All JS in a single <script> tag at end of body, wrapped in IIFE:
(function() { 'use strict'; ... })();

Global functions BEFORE the IIFE:
  window.copyCode = function(btn) { /* clipboard copy with 2s success animation */ }

Required data:
  const LOSS = { text: [...], mobility: [...], fusion: [...], mepoi: [...] };  // 30 epochs
  const POIS = [...];  // 50 sample POIs with lat, lon, category, price, etc.

Required init functions:
  initBenchmarkChart()
  initLossChart()
  initLeafletMap()
  initSlider()
  initCityExplorer()
  initCityMap()

═══════════════════════════════════════════
SECTION 6: GITHUB DEPLOYMENT
═══════════════════════════════════════════

At the end, run:
  git init
  git remote add origin https://github.com/[USERNAME]/[REPO-NAME].git
  git add .
  git commit -m "Initial: [PAPER NAME] interactive research publication"
  git push origin main

Enable GitHub Pages on the main branch.
Save a copy of all files inside a Research_implementation/ folder as the reusable skin.

═══════════════════════════════════════════
SECTION 7: QUALITY CHECKLIST
═══════════════════════════════════════════

Before finalizing, verify all of these:
  □ All $$ in prose (price tiers) are escaped as &dollar;&dollar;
  □ KaTeX display math uses literal $$ delimiters only inside .math-display divs
  □ Run verify_latex.py (or equivalent) 3 times — 0 corrupt entries
  □ Header is single-line flex row (no wrapping on desktop)
  □ All terminal windows have syntax-highlighted code
  □ Copy buttons work (navigator.clipboard.writeText)
  □ Leaflet map loads on localhost:8080
  □ Overpass API city explorer fetches and displays real data
  □ train_modelname.py is a standalone runnable script (tested locally)
  □ favicon.svg is in root directory
  □ Page title is "Research Work · [PAPER SHORT NAME]"
```

---

## ✦ DESIGN TOKEN REFERENCE

| Token | Value | Usage |
|---|---|---|
| `--bg-primary` | `#fefefe` | Page background |
| `--text-primary` | `#1e293b` | Body text |
| `--accent-blue` | `#4285F4` | Links, CTAs |
| `--accent-green` | `#22c55e` | Success states |
| `--accent-amber` | `#f59e0b` | Warnings, hand-notes |
| `--header-bg` | `#1a1a2e → #16213e` | Sticky header gradient |
| `--code-bg` | `#0d1117` | Terminal window background |
| `--code-header-bg` | `#161b22` | Terminal window header bar |
| `--code-border` | `#30363d` | Terminal window border |
| `--font-body` | Source Serif 4 | All article prose |
| `--font-ui` | Inter | Nav, buttons, labels |
| `--font-mono` | Roboto Mono | Code blocks |
| `--font-hand` | Caveat | Hand-written notes |

---

## ✦ CDN DEPENDENCIES

```html
<!-- Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;0,700;1,400&family=Inter:wght@400;500;600;700;800&family=Roboto+Mono:wght@400;500&family=Caveat:wght@400;600;700&display=swap" rel="stylesheet">

<!-- Leaflet.js Map -->
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

<!-- Plotly.js Charts -->
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>

<!-- KaTeX Math Rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});"></script>
```

---

## ✦ EXAMPLE INSTANTIATION (ME-POIs)

```
PAPER TITLE: ME-POIs: Multimodal Entity understanding for Points of Interest
PAPER URL:   https://research.google/blog/how-mobility-gives-language-models-a-deeper-understanding-of-place/
MODEL NAME:  ME_POIs_Model
KEY MODULES: TemporalMobilityEncoder, MultiScaleSpatialGNN
KEY METRICS: Macro F1 (Price), IoU (Hours), AUROC (Closure), MAE (Intent), RMSE (Busyness)
DATASET:     500 synthetic SF POIs (5 neighborhoods), 300 NYC POIs for cross-city transfer
```

---

## ✦ REUSE THIS FOR ANY PAPER

Swap out:
- Paper title & URL
- Model architecture names + module names
- Metrics used in evaluation
- Dataset description
- Key analogies in Steps 1–3

Keep everything else (design system, terminal windows, city explorer, math rendering) exactly the same.
