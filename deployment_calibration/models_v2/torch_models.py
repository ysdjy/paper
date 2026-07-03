"""Torch history models for models_v2: a genuine DeepSets and a GRU.

Both encode the variable-length probe history into a fixed embedding, concatenate it with the
candidate's static features, and predict three targets (success logit, task error, exec time)
from a shared trunk. Training: Adam, early stopping on validation multitask loss, fixed seed.

DeepSets is a REAL permutation-invariant set encoder (shared per-probe MLP phi, then mean+sum
pooling, then rho) -- not a hand-crafted mean/max feature. GRU consumes probes in order and is
order-sensitive. With K=0 (no probes) both fall back to a learned empty-set embedding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from . import features as F
from .base import Model, clip_predictions


def _standardize_fit(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(0) if len(X) else np.zeros(X.shape[1])
    sd = (X.std(0) + 1e-8) if len(X) else np.ones(X.shape[1])
    return mu, sd


class _PhiMLP(nn.Module):
    def __init__(self, d_in, d_hid, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(),
                                 nn.Linear(d_hid, d_out), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class _DeepSetsEncoder(nn.Module):
    """phi over each probe, mean+sum pool, then a small rho. Permutation invariant."""
    def __init__(self, d_probe, d_hid, d_emb):
        super().__init__()
        self.phi = _PhiMLP(d_probe, d_hid, d_hid)
        self.rho = nn.Sequential(nn.Linear(2 * d_hid, d_emb), nn.ReLU())
        self.empty = nn.Parameter(torch.zeros(d_emb))
        self.d_emb = d_emb

    def forward(self, probes, mask):
        # probes: (B, K, d_probe), mask: (B, K) 1 for real probe
        if probes.shape[1] == 0:
            return self.empty.expand(probes.shape[0], self.d_emb)
        phi = self.phi(probes) * mask.unsqueeze(-1)          # (B,K,H)
        s = phi.sum(1)
        cnt = mask.sum(1, keepdim=True).clamp(min=1.0)
        mean = s / cnt
        pooled = torch.cat([mean, s], dim=-1)
        emb = self.rho(pooled)
        has = (mask.sum(1, keepdim=True) > 0).float()
        return has * emb + (1 - has) * self.empty


class _GRUEncoder(nn.Module):
    def __init__(self, d_probe, d_hid, d_emb):
        super().__init__()
        self.gru = nn.GRU(d_probe, d_hid, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d_hid, d_emb), nn.ReLU())
        self.empty = nn.Parameter(torch.zeros(d_emb))
        self.d_emb = d_emb

    def forward(self, probes, mask):
        if probes.shape[1] == 0:
            return self.empty.expand(probes.shape[0], self.d_emb)
        lengths = mask.sum(1).long().clamp(min=0)
        B = probes.shape[0]
        emb = torch.zeros(B, self.d_emb, device=probes.device)
        nonempty = lengths > 0
        if nonempty.any():
            packed = nn.utils.rnn.pack_padded_sequence(
                probes[nonempty], lengths[nonempty].cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            emb_ne = self.out(h[-1])
            emb[nonempty] = emb_ne
        if (~nonempty).any():
            emb[~nonempty] = self.empty
        return emb


class _Net(nn.Module):
    def __init__(self, encoder, d_static, d_emb, d_hid=32):
        super().__init__()
        self.encoder = encoder
        self.trunk = nn.Sequential(nn.Linear(d_static + d_emb, d_hid), nn.ReLU(),
                                   nn.Linear(d_hid, d_hid), nn.ReLU())
        self.head_succ = nn.Linear(d_hid, 1)
        self.head_err = nn.Linear(d_hid, 1)
        self.head_time = nn.Linear(d_hid, 1)

    def forward(self, static, probes, mask):
        emb = self.encoder(probes, mask)
        z = self.trunk(torch.cat([static, emb], dim=-1))
        return self.head_succ(z).squeeze(-1), self.head_err(z).squeeze(-1), self.head_time(z).squeeze(-1)


class _TorchHistoryModel(Model):
    reads_history = True
    encoder_kind = "deepsets"

    def __init__(self, d_hid=32, d_emb=16, lr=1e-2, max_epochs=300, patience=30, l2=1e-4):
        self.hp = dict(d_hid=d_hid, d_emb=d_emb, lr=lr, max_epochs=max_epochs,
                       patience=patience, l2=l2)

    # ---- tensor prep ----
    def _prep(self, pairs, fit_stats=False):
        static = np.array([F.static_features(e) for e, _ in pairs], dtype=float)
        mats = [F.history_matrix(H) for _, H in pairs]
        Kmax = max([m.shape[0] for m in mats] + [0])
        y = np.array([F.targets(e) for e, _ in pairs], dtype=float)
        if fit_stats:
            self.s_mu, self.s_sd = _standardize_fit(static)
            allp = np.concatenate([m for m in mats if m.shape[0] > 0], axis=0) \
                if any(m.shape[0] for m in mats) else np.zeros((1, F.PROBE_DIM))
            self.p_mu, self.p_sd = _standardize_fit(allp)
            self.t_err_mu, self.t_err_sd = float(y[:, 1].mean()), float(y[:, 1].std() + 1e-8)
            self.t_tim_mu, self.t_tim_sd = float(y[:, 2].mean()), float(y[:, 2].std() + 1e-8)
        static = (static - self.s_mu) / self.s_sd
        B = len(pairs)
        probes = np.zeros((B, Kmax, F.PROBE_DIM), dtype=float)
        mask = np.zeros((B, Kmax), dtype=float)
        for i, m in enumerate(mats):
            k = m.shape[0]
            if k:
                probes[i, :k] = (m - self.p_mu) / self.p_sd
                mask[i, :k] = 1.0
        return (torch.tensor(static, dtype=torch.float32),
                torch.tensor(probes, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    def _build(self):
        if self.encoder_kind == "deepsets":
            enc = _DeepSetsEncoder(F.PROBE_DIM, self.hp["d_hid"], self.hp["d_emb"])
        else:
            enc = _GRUEncoder(F.PROBE_DIM, self.hp["d_hid"], self.hp["d_emb"])
        return _Net(enc, F.STATIC_DIM, self.hp["d_emb"], self.hp["d_hid"])

    def _loss(self, out, y):
        logit, err, tim = out
        ys = y[:, 0]
        err_n = (y[:, 1] - self.t_err_mu) / self.t_err_sd
        tim_n = (y[:, 2] - self.t_tim_mu) / self.t_tim_sd
        bce = nn.functional.binary_cross_entropy_with_logits(logit, ys)
        mse_e = nn.functional.mse_loss(err, err_n)
        mse_t = nn.functional.mse_loss(tim, tim_n)
        return bce + mse_e + mse_t

    def fit(self, train_pairs, val_pairs=None, seed: int = 0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.net = self._build()
        tr = self._prep(train_pairs, fit_stats=True)
        va = self._prep(val_pairs, fit_stats=False) if val_pairs else tr
        opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["l2"])
        best_val, best_state, bad = float("inf"), None, 0
        for _ in range(self.hp["max_epochs"]):
            self.net.train()
            opt.zero_grad()
            loss = self._loss(self.net(tr[0], tr[1], tr[2]), tr[3])
            loss.backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                vloss = float(self._loss(self.net(va[0], va[1], va[2]), va[3]))
            if vloss < best_val - 1e-5:
                best_val, bad = vloss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.hp["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self._best_val = best_val
        return self

    def predict(self, e, H):
        self.net.eval()
        s, p, m, _ = self._prep([(e, H)], fit_stats=False)
        with torch.no_grad():
            logit, err, tim = self.net(s, p, m)
        return clip_predictions({
            "p_success": float(torch.sigmoid(logit)[0]),
            "pred_error": float(err[0]) * self.t_err_sd + self.t_err_mu,
            "pred_time": float(tim[0]) * self.t_tim_sd + self.t_tim_mu,
        })

    def config(self):
        c = super().config()
        c.update({"encoder": self.encoder_kind, "hp": self.hp})
        return c

    def state_dict(self):
        return {"hp": self.hp, "encoder_kind": self.encoder_kind,
                "net": self.net.state_dict(),
                "stats": {"s_mu": self.s_mu.tolist(), "s_sd": self.s_sd.tolist(),
                          "p_mu": self.p_mu.tolist(), "p_sd": self.p_sd.tolist(),
                          "t_err": [self.t_err_mu, self.t_err_sd],
                          "t_tim": [self.t_tim_mu, self.t_tim_sd]}}


class DeepSets(_TorchHistoryModel):
    name = "B2_deepsets"
    encoder_kind = "deepsets"


class GRU(_TorchHistoryModel):
    name = "B2_gru"
    encoder_kind = "gru"
