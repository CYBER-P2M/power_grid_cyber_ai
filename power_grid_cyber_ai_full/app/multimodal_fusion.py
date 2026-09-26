from __future__ import annotations

from typing import Any


class MultimodalFusion:
    """
    Combines cyber-attack risk and physical / computer-vision risk
    into one final power-grid security risk score.
    """

    def __init__(
        self,
        cyber_weight: float = 0.45,
        physical_weight: float = 0.55,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.cyber_weight = float(cyber_weight)
        self.physical_weight = float(physical_weight)

        total_weight = self.cyber_weight + self.physical_weight

        if total_weight <= 0:
            self.cyber_weight = 0.50
            self.physical_weight = 0.50
        else:
            # Ensure the weights always sum to 1.0.
            self.cyber_weight /= total_weight
            self.physical_weight /= total_weight

    @staticmethod
    def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
        """Limit a numeric risk/confidence value to the 0-1 range."""
        return max(minimum, min(maximum, float(value)))

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """Convert a value safely to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _extract_score(
        cls,
        result: Any,
        possible_keys: tuple[str, ...],
        default: float = 0.0,
    ) -> float:
        """
        Accept either a direct number or a result dictionary.
        It lets the class work with various IDS / vision output formats.
        """
        if isinstance(result, (int, float)):
            return cls._clamp(result)

        if isinstance(result, dict):
            for key in possible_keys:
                if key in result:
                    return cls._clamp(cls._to_float(result[key], default))

        return default

    @staticmethod
    def _extract_labels(vision_result: Any) -> list[str]:
        """Read detected labels from a vision result dictionary."""
        if not isinstance(vision_result, dict):
            return []

        labels = vision_result.get(
            "detected_labels",
            vision_result.get("labels", vision_result.get("classes", [])),
        )

        if isinstance(labels, str):
            return [labels]

        if isinstance(labels, (list, tuple, set)):
            return [str(label) for label in labels]

        return []

    def fuse(
        self,
        cyber_result: Any,
        vision_result: Any,
        cyber_confidence: float | None = None,
        attack_type: str | None = None,
        detected_labels: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Fuse IDS output and computer-vision output.

        Compatible with your Streamlit call:
            fused = fusion.fuse(cyber_result, vision_result)

        Accepted example inputs:
            cyber_result = {"risk_score": 0.30, "confidence": 0.25, "attack_type": "normal"}
            vision_result = {"risk_score": 0.15, "labels": ["normal"]}
        """
        cyber_risk = self._extract_score(
            cyber_result,
            (
                "cyber_risk",
                "risk_score",
                "risk",
                "score",
                "anomaly_score",
                "probability",
            ),
        )

        physical_risk = self._extract_score(
            vision_result,
            (
                "physical_risk",
                "risk_score",
                "risk",
                "score",
                "threat_score",
                "probability",
            ),
        )

        if cyber_confidence is None:
            if isinstance(cyber_result, dict):
                cyber_confidence = cyber_result.get(
                    "cyber_confidence",
                    cyber_result.get("confidence", 1.0),
                )
            else:
                cyber_confidence = 1.0

        cyber_confidence = self._clamp(
            self._to_float(cyber_confidence, default=1.0)
        )

        if attack_type is None:
            if isinstance(cyber_result, dict):
                attack_type = str(
                    cyber_result.get(
                        "attack_type",
                        cyber_result.get("prediction", cyber_result.get("label", "normal")),
                    )
                )
            else:
                attack_type = "normal"

        if detected_labels is None:
            detected_labels = self._extract_labels(vision_result)

        # Confidence reduces the cyber score only if it is lower than 1.
        adjusted_cyber_risk = cyber_risk * cyber_confidence

        fused_risk_score = (
            self.cyber_weight * adjusted_cyber_risk
            + self.physical_weight * physical_risk
        )

        fused_risk_score = self._clamp(fused_risk_score)

        if fused_risk_score >= 0.75:
            risk_level = "critical"
        elif fused_risk_score >= 0.50:
            risk_level = "high"
        elif fused_risk_score >= 0.25:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "fused_risk_score": round(fused_risk_score, 4),
            "risk_level": risk_level,
            "cyber_risk": round(cyber_risk, 4),
            "physical_risk": round(physical_risk, 4),
            "cyber_confidence": round(cyber_confidence, 4),
            "adjusted_cyber_risk": round(adjusted_cyber_risk, 4),
            "attack_type": attack_type,
            "detected_labels": detected_labels or [],
            "cyber_weight": round(self.cyber_weight, 4),
            "physical_weight": round(self.physical_weight, 4),
        }

    def predict(
        self,
        cyber_result: Any,
        vision_result: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Alias for fuse(), for compatibility with prediction pages."""
        return self.fuse(cyber_result, vision_result, **kwargs)

    def combine(
        self,
        cyber_result: Any,
        vision_result: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Alias for fuse(), for compatibility with older code."""
        return self.fuse(cyber_result, vision_result, **kwargs)