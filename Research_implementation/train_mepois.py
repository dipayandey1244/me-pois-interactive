"""
ME-POIs: Multimodal Representation Learning for Points of Interest
Complete Standalone Training Script (PyTorch 2.1+)

To run this script:
    pip install torch sentence-transformers scikit-learn scipy numpy

Usage:
    python train_mepois.py
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from scipy import stats

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Device: {device}")

# ── 1. NEIGHBORHOODS & POI GENERATOR ──
SF_NEIGHBORHOODS = [
    ("Financial District", 37.7940, -122.4005, ["Corporate Office", "Coffee Shop", "Transit Hub"]),
    ("Mission District", 37.7600, -122.4194, ["Restaurant", "Nightlife/Bar", "Boutique Retail"]),
    ("SoMa Tech Hub", 37.7785, -122.4056, ["Tech Workspace", "Coffee Shop", "Restaurant"]),
    ("Marina & Presidio", 37.8020, -122.4400, ["Public Park", "Boutique Retail", "Healthcare Facility"]),
    ("Union Square", 37.7880, -122.4075, ["Retail Store", "Restaurant", "Transit Hub"])
]

CATEGORY_PROFILE = {
    "Restaurant": (12, 19, 60),
    "Coffee Shop": (8, 10, 30),
    "Corporate Office": (9, 14, 480),
    "Nightlife/Bar": (22, 23, 150),
    "Tech Workspace": (10, 14, 420),
    "Public Park": (16, 12, 90),
    "Transit Hub": (8, 17, 15),
    "Healthcare Facility": (10, 11, 120),
    "Boutique Retail": (15, 14, 45),
    "Retail Store": (15, 14, 45)
}

print("📥 Loading SentenceTransformer ('all-MiniLM-L6-v2')...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def generate_pois(city_name, neighborhoods, num_pois=500):
    descriptions, coords, cat_list, names = [], [], [], []
    for i in range(num_pois):
        neigh, clat, clon, allowed = neighborhoods[i % len(neighborhoods)]
        cat = np.random.choice(allowed)
        name = f"{neigh} {cat} #{i+1}"
        lat = clat + np.random.normal(0, 0.008)
        lon = clon + np.random.normal(0, 0.008)
        names.append(name)
        cat_list.append(cat)
        coords.append([lat, lon])
        descriptions.append(f"{name} is a {cat} in {neigh}, {city_name}.")

    coords = np.array(coords)
    raw_embeds = embedder.encode(descriptions, batch_size=128, show_progress_bar=False)

    mobility_tensors, hours_labels, price_labels, closure_labels, intent_labels, busyness_labels = [], [], [], [], [], []
    for i in range(num_pois):
        cat = cat_list[i]
        peak_wd, peak_we, avg_dwell = CATEGORY_PROFILE[cat]
        mob = np.zeros((168, 4))
        for h in range(168):
            day, hr = h // 24, h % 24
            is_we = day in [5, 6]
            peak = peak_we if is_we else peak_wd
            val = np.exp(-0.5 * ((hr - peak) / 3.0) ** 2) * 100.0
            arr = max(0, val + np.random.poisson(lam=5))
            dep = max(0, arr * 0.85 + np.random.normal(0, 2))
            dwell = max(10, avg_dwell + np.random.normal(0, 10))
            density = arr * (dwell / 60.0)
            mob[h] = [arr, dep, dwell, density]

        mob_norm = (mob - mob.mean(axis=0)) / (mob.std(axis=0) + 1e-6)
        mobility_tensors.append(mob_norm)
        hours_labels.append((mob[:, 0] > (0.15 * mob[:, 0].max())).astype(np.float32))
        price_labels.append(i % 4)
        closure_labels.append(1.0 if np.random.rand() < 0.05 else 0.0)
        intent_labels.append(float(mob[:, 0].sum() * (1.0 + 0.5 * (i % 4))))
        busyness_labels.append(mob[:24, 3])

    # Haversine distance matrix
    lat_r, lon_r = np.radians(coords[:, 0]), np.radians(coords[:, 1])
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = np.sin(dlat / 2.0)**2 + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2.0)**2
    dist_matrix = 6371.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return {
        "text_embeddings": raw_embeds.astype(np.float32),
        "mobility_tensors": np.array(mobility_tensors, dtype=np.float32),
        "dist_matrix": dist_matrix.astype(np.float32),
        "coords": coords,
        "hours": np.array(hours_labels, dtype=np.float32),
        "price": np.array(price_labels, dtype=np.int64),
        "closure": np.array(closure_labels, dtype=np.float32),
        "intent": np.array(intent_labels, dtype=np.float32),
        "busyness": np.array(busyness_labels, dtype=np.float32)
    }


# ── 2. PYTORCH MODEL ARCHITECTURE ──
class TemporalMobilityEncoder(nn.Module):
    def __init__(self, in_ch=4, seq_len=168, d_model=128):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=3, padding=1)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.out = nn.Linear(d_model, 128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x)).transpose(1, 2) + self.pos
        return torch.relu(self.out(self.transformer(x).mean(dim=1)))


class MultiScaleSpatialGNN(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.local_proj = nn.Linear(dim, dim)
        self.neigh_proj = nn.Linear(dim, dim)
        self.fuse = nn.Linear(dim * 3, dim)

    def forward(self, x, dist_matrix):
        adj_local = (dist_matrix <= 0.5).float()
        deg_l = adj_local.sum(1, keepdim=True) + 1e-6
        h_local = torch.relu(self.local_proj(adj_local @ x / deg_l))

        adj_neigh = ((dist_matrix > 0.5) & (dist_matrix <= 2.0)).float()
        deg_n = adj_neigh.sum(1, keepdim=True) + 1e-6
        h_neigh = torch.relu(self.neigh_proj(adj_neigh @ x / deg_n))

        return torch.relu(self.fuse(torch.cat([x, h_local, h_neigh], dim=-1)))


class ME_POIs_Model(nn.Module):
    def __init__(self, mode="me_poi", text_dim=384):
        super().__init__()
        self.mode = mode
        self.text_proj = nn.Linear(text_dim, 128)
        self.mob_encoder = TemporalMobilityEncoder()
        self.fusion = nn.Linear(256, 256)
        self.gnn = MultiScaleSpatialGNN(dim=256)

        # Multi-task prediction heads
        self.head_hours = nn.Linear(256, 168)
        self.head_price = nn.Linear(256, 4)
        self.head_closure = nn.Linear(256, 1)
        self.head_intent = nn.Linear(256, 1)
        self.head_busyness = nn.Linear(256, 24)

    def forward(self, text_emb, mob_tensor, dist_matrix):
        t_feat = torch.relu(self.text_proj(text_emb)) if self.mode != "mobility_only" else torch.zeros(text_emb.size(0), 128, device=text_emb.device)
        m_feat = self.mob_encoder(mob_tensor) if self.mode != "text_only" else torch.zeros(text_emb.size(0), 128, device=text_emb.device)

        fused = torch.relu(self.fusion(torch.cat([t_feat, m_feat], dim=-1)))
        if self.mode == "me_poi":
            fused = self.gnn(fused, dist_matrix)

        return {
            "hours": self.head_hours(fused),
            "price": self.head_price(fused),
            "closure": self.head_closure(fused).squeeze(-1),
            "intent": self.head_intent(fused).squeeze(-1),
            "busyness": self.head_busyness(fused)
        }


# ── 3. LOSS & EVALUATION ──
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self, preds, targets):
        lh = self.bce(preds["hours"], targets["hours"])
        lp = self.ce(preds["price"], targets["price"])
        lc = self.bce(preds["closure"], targets["closure"])
        li = self.huber(preds["intent"], targets["intent"])
        lb = self.huber(preds["busyness"], targets["busyness"])
        return lh + 2.0 * lp + 3.0 * lc + 0.01 * li + 1.0 * lb


# ── 4. MAIN TRAINING LOOP ──
if __name__ == "__main__":
    print("\n🏙️ Generating San Francisco POI Dataset (N=500)...")
    data = generate_pois("San Francisco", SF_NEIGHBORHOODS, num_pois=500)

    # Spatial split: hold out center area
    coords = data["coords"]
    test_mask = (coords[:, 0] >= 37.75) & (coords[:, 0] <= 37.77) & (coords[:, 1] >= -122.43) & (coords[:, 1] <= -122.41)
    train_idx = np.where(~test_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"📊 Dataset Split: {len(train_idx)} Train POIs | {len(test_idx)} Test Holdout POIs")

    # Tensors
    text_t = torch.tensor(data["text_embeddings"], device=device)
    mob_t = torch.tensor(data["mobility_tensors"], device=device)
    dist_t = torch.tensor(data["dist_matrix"], device=device)
    targets = {k: torch.tensor(data[k], device=device) for k in ["hours", "price", "closure", "intent", "busyness"]}

    modes = ["text_only", "mobility_only", "fusion", "me_poi"]
    results = {}

    print("\n🏋️ Training All 4 Model Variants (30 Epochs Each)...")
    for mode in modes:
        model = ME_POIs_Model(mode=mode).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        criterion = MultiTaskLoss()

        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            preds = model(text_t, mob_t, dist_t)
            train_preds = {k: v[train_idx] for k, v in preds.items()}
            train_targets = {k: v[train_idx] for k, v in targets.items()}
            loss = criterion(train_preds, train_targets)
            loss.backward()
            optimizer.step()

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            preds = model(text_t, mob_t, dist_t)
            test_preds = {k: v[test_idx] for k, v in preds.items()}
            test_targets = {k: v[test_idx] for k, v in targets.items()}

            price_pred = torch.argmax(test_preds["price"], dim=-1).cpu().numpy()
            price_f1 = f1_score(test_targets["price"].cpu().numpy(), price_pred, average="macro")

            sig_hours = (torch.sigmoid(test_preds["hours"]) > 0.5).float().cpu().numpy()
            tgt_hours = test_targets["hours"].cpu().numpy()
            hours_iou = (np.logical_and(sig_hours, tgt_hours).sum()) / (np.logical_or(sig_hours, tgt_hours).sum() + 1e-6)

            prob_closure = torch.sigmoid(test_preds["closure"]).cpu().numpy()
            try:
                closure_auroc = roc_auc_score(test_targets["closure"].cpu().numpy(), prob_closure)
            except Exception:
                closure_auroc = 0.5

            results[mode] = {"price_f1": round(price_f1, 4), "hours_iou": round(hours_iou, 4), "closure_auroc": round(closure_auroc, 4)}
            print(f"  [{mode.upper().ljust(13)}] → Price F1: {price_f1:.4f} | Hours IoU: {hours_iou:.4f} | Closure AUROC: {closure_auroc:.4f}")

    print("\n✅ Training Complete! All models evaluated on unseen holdout venues.")
