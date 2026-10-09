# phase3/security/security_gateway.py
"""
Simulated Network Security Gateway for Phase 3.

IMPORTANT DISCLAIMER:
This is a controlled research simulation. It does NOT perform real
network interception, packet inspection, or offensive actions.
The gateway simulates a defensive security monitor for a hypothetical
networked roadside camera system.

Architecture:
    Network flow features → Random Forest → NORMAL / SUSPICIOUS

Usage:
    gw = SecurityGateway.load()
    result = gw.classify(flow_features_array)
    if result.is_suspicious:
        log_and_reject()
    else:
        pass_to_cv_pipeline()
"""

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

CKPT_DIR  = Path("D:/road obstacle detection/road_damage_research/phase3/checkpoints")
LOG_DIR   = Path("D:/road obstacle detection/road_damage_research/results/phase3/logs")


@dataclass
class GatewayDecision:
    """Result of a single gateway classification."""
    label: str                   # "NORMAL" or "SUSPICIOUS"
    confidence: float            # probability of SUSPICIOUS (class 1)
    is_suspicious: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "label":        self.label,
            "confidence":   round(self.confidence, 4),
            "is_suspicious": self.is_suspicious,
            "latency_ms":   round(self.latency_ms, 4),
            "timestamp":    self.timestamp,
        }


class SecurityGateway:
    """
    Lightweight security gateway wrapping the trained Random Forest.

    Parameters
    ----------
    threshold : float
        Probability threshold above which traffic is classified as SUSPICIOUS.
        Default 0.5.
    """

    def __init__(self, model, scaler, feature_names: list,
                 threshold: float = 0.5, log_to_file: bool = True):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.threshold = threshold
        self._n_total = 0
        self._n_suspicious = 0
        self._n_normal = 0
        self._total_latency_ms = 0.0

        if log_to_file:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=str(LOG_DIR / "gateway_events.log"),
                level=logging.INFO,
                format="%(asctime)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        self._log_to_file = log_to_file

    @classmethod
    def load(cls, threshold: float = 0.5, log_to_file: bool = True) -> "SecurityGateway":
        """Load trained model and scaler from checkpoint directory."""
        model = joblib.load(CKPT_DIR / "random_forest.joblib")
        scaler = joblib.load(CKPT_DIR / "ids_scaler.joblib")
        meta_path = CKPT_DIR / "ids_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = meta["feature_columns"]
        print(f"  SecurityGateway loaded | model: RandomForest | features: {len(feature_names)}")
        return cls(model, scaler, feature_names, threshold=threshold, log_to_file=log_to_file)

    def classify(self, flow_features: np.ndarray) -> GatewayDecision:
        """
        Classify a single network flow.

        Parameters
        ----------
        flow_features : np.ndarray of shape (n_features,) — RAW (unscaled)

        Returns
        -------
        GatewayDecision
        """
        t0 = time.perf_counter()
        x = self.scaler.transform(flow_features.reshape(1, -1))
        prob = self.model.predict_proba(x)[0, 1]
        is_suspicious = bool(prob >= self.threshold)
        latency_ms = (time.perf_counter() - t0) * 1000

        label = "SUSPICIOUS" if is_suspicious else "NORMAL"
        decision = GatewayDecision(
            label=label, confidence=float(prob),
            is_suspicious=is_suspicious, latency_ms=latency_ms,
        )

        # Update stats
        self._n_total += 1
        self._total_latency_ms += latency_ms
        if is_suspicious:
            self._n_suspicious += 1
            if self._log_to_file:
                logging.warning(f"SUSPICIOUS | confidence={prob:.3f} | latency={latency_ms:.2f}ms")
        else:
            self._n_normal += 1

        return decision

    def classify_batch(self, X: np.ndarray, already_scaled: bool = False):
        """
        Classify a batch of flows (efficiently).

        Parameters
        ----------
        X : np.ndarray of shape (N, n_features)
        already_scaled : bool — if True, skip scaling

        Returns
        -------
        labels: np.ndarray of str, probs: np.ndarray of float
        """
        t0 = time.perf_counter()
        if not already_scaled:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        probs = self.model.predict_proba(X_scaled)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000 / len(X)

        labels = np.where(probs >= self.threshold, "SUSPICIOUS", "NORMAL")
        self._n_total += len(X)
        self._n_suspicious += int((probs >= self.threshold).sum())
        self._n_normal += int((probs < self.threshold).sum())
        self._total_latency_ms += latency_ms * len(X)
        return labels, probs, latency_ms

    def get_stats(self) -> dict:
        avg_lat = self._total_latency_ms / max(self._n_total, 1)
        return {
            "total_classified":       self._n_total,
            "n_normal":               self._n_normal,
            "n_suspicious":           self._n_suspicious,
            "suspicious_rate":        round(self._n_suspicious / max(self._n_total, 1), 4),
            "avg_latency_ms":         round(avg_lat, 4),
        }


if __name__ == "__main__":
    gw = SecurityGateway.load()
    # Smoke test with random features
    n_feat = len(gw.feature_names)
    dummy = np.zeros(n_feat, dtype=np.float32)
    result = gw.classify(dummy)
    print(f"Test classification: {result.label} (confidence={result.confidence:.3f})")
