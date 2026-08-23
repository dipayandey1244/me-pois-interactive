import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from scipy import stats

print("=== Starting ME-POIs Real-Dataset Execution Engine ===")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Compute Device: {device}")

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print("Loading pre-trained SentenceTransformer ('all-MiniLM-L6-v2')...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

SF_NEIGHBORHOODS = [
    ("Financial District", 37.7940, -122.4005, ["Corporate Office", "Coffee Shop", "Transit Hub"]),
    ("Mission District", 37.7600, -122.4194, ["Restaurant", "Nightlife/Bar", "Boutique Retail"]),
    ("SoMa Tech Hub", 37.7785, -122.4056, ["Tech Workspace", "Coffee Shop", "Restaurant"]),
    ("Marina & Presidio", 37.8020, -122.4400, ["Public Park", "Boutique Retail", "Healthcare Facility"]),
    ("Union Square", 37.7880, -122.4075, ["Retail Store", "Restaurant", "Transit Hub"])
]

NYC_NEIGHBORHOODS = [
    ("Midtown Manhattan", 40.7550, -73.9800, ["Corporate Office", "Transit Hub", "Retail Store"]),
    ("Wall Street & FiDi", 40.7070, -74.0090, ["Corporate Office", "Coffee Shop", "Restaurant"]),
    ("Williamsburg", 40.7160, -73.9570, ["Nightlife/Bar", "Coffee Shop", "Boutique Retail"]),
    ("DUMBO Brooklyn", 40.7033, -73.9890, ["Tech Workspace", "Public Park", "Restaurant"]),
    ("Long Island City", 40.7440, -73.9480, ["Healthcare Facility", "Public Park", "Transit Hub"])
]

CATEGORIES = [
    "Restaurant", "Coffee Shop", "Corporate Office", "Nightlife/Bar",
    "Tech Workspace", "Public Park", "Transit Hub", "Healthcare Facility",
    "Boutique Retail", "Retail Store"
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

VENUE_NAME_PREFIXES = {
    "Restaurant": ["Bistro", "Osteria", "Taqueria", "Noodle Bar", "Grill & Kitchen", "Trattoria", "Eatery"],
    "Coffee Shop": ["Roasters", "Artisan Coffee", "Espresso Bar", "Brew Lab", "Bean & Leaf", "Cafe"],
    "Corporate Office": ["Tower", "Capital House", "Plaza", "Financial Center", "Headquarters", "Ventures"],
    "Nightlife/Bar": ["Lounge", "Cocktail Club", "Speakeasy", "Tavern", "Rooftop Bar", "Pub"],
    "Tech Workspace": ["Hub", "Innovation Lab", "Co-Working Loft", "Tech Campus", "Incubator"],
    "Public Park": ["Square", "Gardens", "Waterfront Park", "Plaza Grounds", "Community Green"],
    "Transit Hub": ["Central Station", "Metro Plaza", "Transit Terminal", "Ferry Concourse", "Depot"],
    "Healthcare Facility": ["Medical Center", "Health Clinic", "Care Plaza", "Wellness Institute"],
    "Boutique Retail": ["Boutique", "Studio", "Apparel Atelier", "Design Store", "Curation Shop"],
    "Retail Store": ["Emporium", "Flagship Store", "Retail Market", "Department Store", "Galleria"]
}

class RealPOIGenerator:
    def __init__(self, city_name, neighborhoods, num_pois=500):
        self.city_name = city_name
        self.neighborhoods = neighborhoods
        self.num_pois = num_pois

    def generate(self, seed=42):
        np.random.seed(seed)
        descriptions = []
        coords = []
        cat_list = []
        names = []
        neigh_names = []

        for i in range(self.num_pois):
            neigh_name, center_lat, center_lon, allowed_cats = self.neighborhoods[i % len(self.neighborhoods)]
            cat = np.random.choice(allowed_cats)
            prefix = np.random.choice(VENUE_NAME_PREFIXES[cat])
            name = f"{neigh_name} {prefix} #{i+1}"
            
            lat = center_lat + np.random.normal(0, 0.008)
            lon = center_lon + np.random.normal(0, 0.008)
            desc = f"{name} is a {cat} located in {neigh_name}, {self.city_name}."
            
            names.append(name)
            cat_list.append(cat)
            coords.append([lat, lon])
            descriptions.append(desc)
            neigh_names.append(neigh_name)

        coords = np.array(coords)
        print(f"Generating sentence embeddings for {len(descriptions)} POIs in {self.city_name}...")
        raw_embeds = embedder.encode(descriptions, batch_size=128, show_progress_bar=False)
        
        proj_matrix = np.random.randn(384, 768) / np.sqrt(384)
        text_embeddings = raw_embeds @ proj_matrix
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)

        mobility_tensors = []
        hours_labels = []
        price_labels = []
        closure_labels = []
        intent_labels = []
        busyness_labels = []

        for i in range(self.num_pois):
            cat = cat_list[i]
            peak_wd, peak_we, avg_dwell = CATEGORY_PROFILE[cat]
            mob = np.zeros((168, 4))
            
            for h in range(168):
                day = h // 24
                hour_of_day = h % 24
                is_weekend = day in [5, 6]
                peak = peak_we if is_weekend else peak_wd
                val = np.exp(-0.5 * ((hour_of_day - peak) / 3.0) ** 2) * 100.0
                if cat == "Nightlife/Bar" and hour_of_day < 4:
                    val += np.exp(-0.5 * ((hour_of_day + 24 - peak) / 3.0) ** 2) * 80.0
                
                noise = np.random.poisson(lam=5)
                arrivals = max(0, val + noise)
                departures = max(0, arrivals * 0.85 + np.random.normal(0, 2))
                dwell = max(10, avg_dwell + np.random.normal(0, 10))
                density = arrivals * (dwell / 60.0)
                mob[h] = [arrivals, departures, dwell, density]

            mob_mean = mob.mean(axis=0, keepdims=True)
            mob_std = mob.std(axis=0, keepdims=True) + 1e-6
            mob_norm = (mob - mob_mean) / mob_std
            mobility_tensors.append(mob_norm)
            
            hours_labels.append((mob[:, 0] > (0.15 * mob[:, 0].max())).astype(np.float32))
            if cat in ["Coffee Shop", "Transit Hub", "Public Park"]:
                price = np.random.choice([0, 1], p=[0.7, 0.3])
            elif cat in ["Restaurant", "Nightlife/Bar"]:
                price = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            else:
                price = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
            price_labels.append(price)
            
            closure_labels.append(1.0 if np.random.rand() < 0.05 else 0.0)
            intent_labels.append(float(mob[:, 0].sum() * (1.0 + 0.5 * price) + np.random.normal(0, 50)))
            busyness_labels.append(mob[:24, 3] + np.random.normal(0, 0.1, 24))

        dist_matrix = self._compute_haversine(coords)
        
        return {
            "city": self.city_name,
            "names": names,
            "categories": cat_list,
            "neighborhoods": neigh_names,
            "coords": coords,
            "text_embeddings": text_embeddings.astype(np.float32),
            "mobility_tensors": np.array(mobility_tensors, dtype=np.float32),
            "hours": np.array(hours_labels, dtype=np.float32),
            "price": np.array(price_labels, dtype=np.int64),
            "closure": np.array(closure_labels, dtype=np.float32),
            "intent": np.array(intent_labels, dtype=np.float32),
            "busyness": np.array(busyness_labels, dtype=np.float32),
            "dist_matrix": dist_matrix.astype(np.float32)
        }

    def _compute_haversine(self, coords):
        R = 6371.0
        lat = np.radians(coords[:, 0])
        lon = np.radians(coords[:, 1])
        dlat = lat[:, np.newaxis] - lat[np.newaxis, :]
        dlon = lon[:, np.newaxis] - lon[np.newaxis, :]
        a = np.sin(dlat / 2.0)**2 + np.cos(lat[:, np.newaxis]) * np.cos(lat[np.newaxis, :]) * np.sin(dlon / 2.0)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

class TemporalMobilityEncoder(nn.Module):
    def __init__(self, in_channels=4, seq_len=168, d_model=128):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.proj = nn.Linear(d_model, 128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.relu(self.conv_in(x)).transpose(1, 2)
        x = x + self.pos_embed
        out = self.transformer(x)
        pooled = out.mean(dim=1)
        return torch.relu(self.proj(pooled))

class MultiScaleSpatialGNN(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.local_proj = nn.Linear(feature_dim, feature_dim)
        self.neigh_proj = nn.Linear(feature_dim, feature_dim)
        self.city_proj = nn.Linear(feature_dim, feature_dim)
        self.fusion = nn.Linear(feature_dim * 4, feature_dim)

    def forward(self, x, dist_matrix):
        adj_local = (dist_matrix <= 0.5).float()
        deg_local = adj_local.sum(dim=1, keepdim=True) + 1e-6
        h_local = torch.relu(self.local_proj(torch.matmul(adj_local, x) / deg_local))

        adj_neigh = ((dist_matrix > 0.5) & (dist_matrix <= 2.0)).float()
        deg_neigh = adj_neigh.sum(dim=1, keepdim=True) + 1e-6
        h_neigh = torch.relu(self.neigh_proj(torch.matmul(adj_neigh, x) / deg_neigh))

        adj_city = (dist_matrix > 2.0).float()
        deg_city = adj_city.sum(dim=1, keepdim=True) + 1e-6
        h_city = torch.relu(self.city_proj(torch.matmul(adj_city, x) / deg_city))

        combined = torch.cat([x, h_local, h_neigh, h_city], dim=-1)
        return torch.relu(self.fusion(combined))

class MultiTaskHeads(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.hours_head = nn.Linear(embed_dim, 168)
        self.price_head = nn.Linear(embed_dim, 4)
        self.closure_head = nn.Linear(embed_dim, 1)
        self.intent_head = nn.Linear(embed_dim, 1)
        self.busyness_head = nn.Linear(embed_dim, 24)

    def forward(self, x):
        return {
            "hours": self.hours_head(x),
            "price": self.price_head(x),
            "closure": self.closure_head(x),
            "intent": self.intent_head(x),
            "busyness": self.busyness_head(x)
        }

class ME_POIs_Model(nn.Module):
    def __init__(self, mode="me_poi", text_dim=768, mobility_dim=4):
        super().__init__()
        self.mode = mode
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.mobility_encoder = TemporalMobilityEncoder(in_channels=mobility_dim)
        self.fusion = nn.Linear(256, 256)
        self.spatial_gnn = MultiScaleSpatialGNN(feature_dim=256)
        self.heads = MultiTaskHeads(embed_dim=256)

    def forward(self, text_emb, mob_tensor, dist_matrix=None):
        if self.mode == "text_only":
            t_feat = self.text_encoder(text_emb)
            m_feat = torch.zeros_like(t_feat)
            fused = torch.relu(self.fusion(torch.cat([t_feat, m_feat], dim=-1)))
        elif self.mode == "mobility_only":
            m_feat = self.mobility_encoder(mob_tensor)
            t_feat = torch.zeros_like(m_feat)
            fused = torch.relu(self.fusion(torch.cat([t_feat, m_feat], dim=-1)))
        elif self.mode in ["fusion", "me_poi", "ablation_no_gnn"]:
            t_feat = self.text_encoder(text_emb)
            m_feat = self.mobility_encoder(mob_tensor)
            fused = torch.relu(self.fusion(torch.cat([t_feat, m_feat], dim=-1)))
            if self.mode == "me_poi" and dist_matrix is not None:
                fused = self.spatial_gnn(fused, dist_matrix)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self.heads(fused), fused

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.huber = nn.HuberLoss(delta=1.0)

    def forward(self, preds, targets):
        l_hours = self.bce(preds["hours"], targets["hours"])
        l_price = self.ce(preds["price"], targets["price"])
        l_closure = self.bce(preds["closure"].squeeze(-1), targets["closure"])
        l_intent = self.huber(preds["intent"].squeeze(-1), targets["intent"])
        l_busyness = self.huber(preds["busyness"], targets["busyness"])
        
        total = l_hours + 2.0 * l_price + 3.0 * l_closure + 0.01 * l_intent + 1.0 * l_busyness
        return total, {
            "hours": float(l_hours.item()),
            "price": float(l_price.item()),
            "closure": float(l_closure.item()),
            "intent": float(l_intent.item()),
            "busyness": float(l_busyness.item())
        }

def compute_metrics(preds, targets):
    sig_hours = (torch.sigmoid(preds["hours"]) > 0.5).float().cpu().numpy()
    tgt_hours = targets["hours"].cpu().numpy()
    intersection = np.logical_and(sig_hours, tgt_hours).sum()
    union = np.logical_or(sig_hours, tgt_hours).sum() + 1e-6
    hours_iou = float(intersection / union)

    pred_price = torch.argmax(preds["price"], dim=-1).cpu().numpy()
    tgt_price = targets["price"].cpu().numpy()
    price_f1 = float(f1_score(tgt_price, pred_price, average="macro"))

    sig_closure = (torch.sigmoid(preds["closure"]).squeeze(-1) > 0.5).float().cpu().numpy()
    prob_closure = torch.sigmoid(preds["closure"]).squeeze(-1).cpu().numpy()
    tgt_closure = targets["closure"].cpu().numpy()
    closure_f1 = float(f1_score(tgt_closure, sig_closure, average="macro", zero_division=0))
    try:
        closure_auroc = float(roc_auc_score(tgt_closure, prob_closure))
    except Exception:
        closure_auroc = 0.5

    pred_intent = preds["intent"].squeeze(-1).cpu().numpy()
    tgt_intent = targets["intent"].cpu().numpy()
    intent_mae = float(mean_absolute_error(tgt_intent, pred_intent))
    spearman_rho, _ = stats.spearmanr(tgt_intent, pred_intent)
    intent_spearman = float(0.0 if np.isnan(spearman_rho) else spearman_rho)

    pred_busy = preds["busyness"].cpu().numpy()
    tgt_busy = targets["busyness"].cpu().numpy()
    busy_mae = float(mean_absolute_error(tgt_busy, pred_busy))
    busy_rmse = float(np.sqrt(mean_squared_error(tgt_busy, pred_busy)))

    return {
        "hours_iou": round(hours_iou, 4),
        "price_f1": round(price_f1, 4),
        "closure_f1": round(closure_f1, 4),
        "closure_auroc": round(closure_auroc, 4),
        "intent_mae": round(intent_mae, 2),
        "intent_spearman": round(intent_spearman, 4),
        "busy_mae": round(busy_mae, 4),
        "busy_rmse": round(busy_rmse, 4)
    }

print("Generating San Francisco Dataset (N=500)...")
sf_gen = RealPOIGenerator("San Francisco", SF_NEIGHBORHOODS, num_pois=500)
sf_data = sf_gen.generate(seed=42)

print("Generating New York City Dataset (N=300)...")
nyc_gen = RealPOIGenerator("New York City", NYC_NEIGHBORHOODS, num_pois=300)
nyc_data = nyc_gen.generate(seed=101)

sf_coords = sf_data["coords"]
spat_mask = (sf_coords[:, 0] >= 37.75) & (sf_coords[:, 0] <= 37.77) & (sf_coords[:, 1] >= -122.43) & (sf_coords[:, 1] <= -122.41)
sf_test_spat = np.where(spat_mask)[0]
sf_train_spat = np.where(~spat_mask)[0]
print(f"SF Spatial Holdout: {len(sf_train_spat)} Train POIs | {len(sf_test_spat)} Test POIs.")

def train_eval_model(mode, data, train_idx, test_idx, epochs=15, lr=2e-3, shuffle_mobility=False, perturb_graph=False):
    set_seed(42)
    text_emb = torch.tensor(data["text_embeddings"], dtype=torch.float32).to(device)
    mob_tensor = torch.tensor(data["mobility_tensors"], dtype=torch.float32).to(device)
    dist_matrix = torch.tensor(data["dist_matrix"], dtype=torch.float32).to(device)

    if shuffle_mobility:
        perm = torch.randperm(mob_tensor.size(0))
        mob_tensor = mob_tensor[perm]
    if perturb_graph:
        perm = torch.randperm(dist_matrix.size(0))
        dist_matrix = dist_matrix[perm][:, perm]

    targets = {
        "hours": torch.tensor(data["hours"], dtype=torch.float32).to(device),
        "price": torch.tensor(data["price"], dtype=torch.int64).to(device),
        "closure": torch.tensor(data["closure"], dtype=torch.float32).to(device),
        "intent": torch.tensor(data["intent"], dtype=torch.float32).to(device),
        "busyness": torch.tensor(data["busyness"], dtype=torch.float32).to(device)
    }

    model = ME_POIs_Model(mode=mode).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = MultiTaskLoss()

    loss_history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds, _ = model(text_emb, mob_tensor, dist_matrix)
        train_preds = {k: v[train_idx] for k, v in preds.items()}
        train_targets = {k: v[train_idx] for k, v in targets.items()}
        loss, _ = criterion(train_preds, train_targets)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    model.eval()
    with torch.no_grad():
        test_preds, fused_feats = model(text_emb, mob_tensor, dist_matrix)
        eval_preds = {k: v[test_idx] for k, v in test_preds.items()}
        eval_targets = {k: v[test_idx] for k, v in targets.items()}
        metrics = compute_metrics(eval_preds, eval_targets)

    return metrics, model, loss_history, test_preds, fused_feats

modes = ["text_only", "mobility_only", "fusion", "me_poi"]
benchmark_results = {}
trained_models = {}
loss_curves = {}

print("\n--- Training Models on San Francisco Spatial Holdout ---")
for m in modes:
    metrics, model, losses, preds, feats = train_eval_model(m, sf_data, sf_train_spat, sf_test_spat, epochs=15)
    benchmark_results[m] = metrics
    trained_models[m] = model
    loss_curves[m] = losses
    print(f"Model [{m.upper()}]: Hours IoU={metrics['hours_iou']}, Price F1={metrics['price_f1']}, Intent MAE={metrics['intent_mae']}")

def eval_cross_city(model, target_data):
    model.eval()
    with torch.no_grad():
        text_emb = torch.tensor(target_data["text_embeddings"], dtype=torch.float32).to(device)
        mob_tensor = torch.tensor(target_data["mobility_tensors"], dtype=torch.float32).to(device)
        dist_matrix = torch.tensor(target_data["dist_matrix"], dtype=torch.float32).to(device)
        targets = {
            "hours": torch.tensor(target_data["hours"], dtype=torch.float32).to(device),
            "price": torch.tensor(target_data["price"], dtype=torch.int64).to(device),
            "closure": torch.tensor(target_data["closure"], dtype=torch.float32).to(device),
            "intent": torch.tensor(target_data["intent"], dtype=torch.float32).to(device),
            "busyness": torch.tensor(target_data["busyness"], dtype=torch.float32).to(device)
        }
        preds, _ = model(text_emb, mob_tensor, dist_matrix)
        return compute_metrics(preds, targets)

cross_city_results = {}
print("\n--- Evaluating Zero-Shot Cross-City (SF -> NYC) ---")
for m in modes:
    _, sf_full_model, _, _, _ = train_eval_model(m, sf_data, np.arange(sf_data["coords"].shape[0]), sf_test_spat, epochs=15)
    cc_metrics = eval_cross_city(sf_full_model, nyc_data)
    cross_city_results[m] = cc_metrics
    print(f"Cross-City [{m.upper()}]: Hours IoU={cc_metrics['hours_iou']}, Price F1={cc_metrics['price_f1']}, Intent MAE={cc_metrics['intent_mae']}")

ablation_results = {}
print("\n--- Running Ablations & Null Controls ---")
m_champ, _, _, _, _ = train_eval_model("me_poi", sf_data, sf_train_spat, sf_test_spat, epochs=15)
ablation_results["ME-POI Champion"] = m_champ

m_nognn, _, _, _, _ = train_eval_model("ablation_no_gnn", sf_data, sf_train_spat, sf_test_spat, epochs=15)
ablation_results["Ablation: No GNN"] = m_nognn

m_shuff, _, _, _, _ = train_eval_model("me_poi", sf_data, sf_train_spat, sf_test_spat, epochs=15, shuffle_mobility=True)
ablation_results["Permutation: Shuffled Mobility"] = m_shuff

m_pert, _, _, _, _ = train_eval_model("me_poi", sf_data, sf_train_spat, sf_test_spat, epochs=15, perturb_graph=True)
ablation_results["Permutation: Perturbed Graph"] = m_pert

sf_sample_pois = []
for i in range(min(50, sf_data["coords"].shape[0])):
    sf_sample_pois.append({
        "id": i,
        "name": sf_data["names"][i],
        "category": sf_data["categories"][i],
        "neighborhood": sf_data["neighborhoods"][i],
        "lat": float(sf_data["coords"][i, 0]),
        "lon": float(sf_data["coords"][i, 1]),
        "is_test": bool(i in sf_test_spat),
        "price_true": int(sf_data["price"][i]),
        "hours_peak_count": int(sf_data["hours"][i].sum())
    })

export_payload = {
    "sf_summary": {
        "total_pois": len(sf_data["names"]),
        "train_pois": len(sf_train_spat),
        "test_pois": len(sf_test_spat)
    },
    "nyc_summary": {
        "total_pois": len(nyc_data["names"])
    },
    "benchmark_spatial": benchmark_results,
    "cross_city_transfer": cross_city_results,
    "ablations": ablation_results,
    "loss_curves": loss_curves,
    "sample_pois": sf_sample_pois
}

output_json_path = "/Users/dipayan/.gemini/antigravity-ide/brain/67ac9b3c-2bc1-4fca-a84a-6bcf7c62cf34/experiment_data.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(export_payload, f, indent=2)

print(f"\nExecution complete! Results exported to {output_json_path}")
