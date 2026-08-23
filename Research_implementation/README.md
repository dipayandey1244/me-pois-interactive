# Research Implementation Skin & Web Framework

A modular, Distill.pub-inspired web design system and PyTorch deep learning framework for presenting research papers interactively.

> **Based on Google Research**: [How Mobility Gives Language Models a Deeper Understanding of Place](https://research.google/blog/how-mobility-gives-language-models-a-deeper-understanding-of-place/)

---

## 🎨 Skin Design System & UI Components

### 1. Typography & Aesthetic
- **Body Font**: `Source Serif 4` (Georgia/serif) — 18px body font for clean academic readability.
- **Headings & UI**: `Inter` (Sans-serif) — 800-weight bold titles and uppercase section labels.
- **Monospace Code**: `Roboto Mono` / `JetBrains Mono` — for tensor shapes, metrics, and PyTorch syntax.
- **Hand-Drawn Notes**: `Caveat` (Cursive) — for margin notes, whiteboard diagrams, and intuitive summaries.

### 2. UI Components Included
- 🔴🟡🟢 **macOS Terminal IDE Code Windows**: Syntax-highlighted code blocks with PyTorch environment badges and one-click copy buttons.
- 💡 **Simple Analogy Callouts**: Dashed yellow containers with light background for 10-year-old friendly intuitive explanations.
- ✏️ **Hand-Drawn Margin Notes**: Orange left-bordered notes mimicking whiteboard annotations.
- 📊 **Distill-Style Results Tables**: Multi-metric benchmark tables with green highlights for winning scores.
- 🗺️ **Interactive Leaflet.js Map**: Dynamic POI pins, holdout region overlays, and adjustable distance-threshold GNN edge drawing.
- 📈 **Plotly.js Dynamic Charts**: Real-time training loss curves, multi-task benchmark comparisons, category pie charts, and closure risk distributions.
- 🌍 **Dynamic City Explorer**: Fetch real open-source POIs via Overpass API (OpenStreetMap) for any city globally and run live multi-task model predictions.

---

## 💻 Included Files in `Research_implementation/`

- `index.html` — Full interactive research publication page.
- `train_mepois.py` — Standalone PyTorch 2.1 implementation script for model training.
- `img_problem.png` — Hand-drawn illustration: text-only mismatch & SF→NYC zero-shot transfer collapse.
- `img_architecture.png` — Hand-drawn illustration: Text, Mobility, and Spatial Graph architecture pipeline.
- `img_gnn.png` — Hand-drawn illustration: Multi-scale spatial graph rings (Local ≤0.5km, Neighborhood 0.5–2km, City-wide >2km).
- `experiment_data.json` — Exported loss curves, holdout metrics, and sample venue predictions.

---

## 🚀 Quick Start

### Run Web Application Locally:
```bash
python3 -m http.server 8080
# Open http://localhost:8080/Research_implementation/index.html in your browser
```

### Train PyTorch Model:
```bash
pip install torch sentence-transformers scikit-learn scipy numpy
python train_mepois.py
```

---

## 📜 Attribution & Open Data
- **Google Research Publication**: [Read the Google Research Blog Post](https://research.google/blog/how-mobility-gives-language-models-a-deeper-understanding-of-place/)
- **Map & POI Data**: OpenStreetMap contributors under the **Open Database License (ODbL)**.
