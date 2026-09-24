"""
Multimodal Fusion Module for Cyber-Physical Power Grid Security

Combines cyber (network/SCADA) and physical (computer vision) signals
into a unified risk assessment and diagnostic decision.

Fusion Strategy: Late Fusion with Contradiction Detection
"""

from typing import Dict, Any, Optional, Tuple
import time


class MultimodalFusion:
    """
    Fuses cyber and physical modalities for comprehensive security assessment.
    
    Decision categories:
    - CYBER_ATTACK: High cyber, low physical (e.g., FDIA)
    - PHYSICAL_FAULT: Low cyber, high physical (e.g., equipment failure)
    - CYBER_PHYSICAL_EVENT: Both high (coordinated attack or cascade)
    - INTRUSION: Physical threat only (unauthorized access)
    - NORMAL: Both low
    """
    
    # Decision thresholds
    CYBER_HIGH = 0.7
    PHYSICAL_HIGH = 0.6
    CONTRADICTION_THRESHOLD = 0.4
    
    # Fusion weights (tunable per deployment)
    WEIGHT_CYBER = 0.55
    WEIGHT_PHYSICAL = 0.45
    
    # Risk multipliers for decision categories
    CATEGORY_MULTIPLIERS = {
        'CYBER_ATTACK': 1.0,
        'PHYSICAL_FAULT': 0.9,
        'CYBER_PHYSICAL_EVENT': 1.2,  # Highest priority
        'INTRUSION': 0.8,
        'NORMAL': 0.1
    }
    
    def __init__(self, cyber_weight: float = 0.55, physical_weight: float = 0.45):
        """
        Initialize fusion engine.
        
        Args:
            cyber_weight: Weight for cyber modality (0-1)
            physical_weight: Weight for physical modality (0-1)
        """
        self.WEIGHT_CYBER = cyber_weight
        self.WEIGHT_PHYSICAL = physical_weight
        assert abs(cyber_weight + physical_weight - 1.0) < 0.01, "Weights must sum to 1"
    
    def fuse(self, cyber_result: Dict[str, Any], vision_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuse cyber and vision results into unified decision.
        
        Args:
            cyber_result: Output from cyber IDS model with keys:
                - risk_score: float 0-1
                - attack_type: str (e.g., 'FDIA', 'DoS', 'normal')
                - confidence: float
            vision_result: Output from VisionDetector with keys:
                - risk_score: float 0-1
                - labels: List[str]
                - is_physical_threat: bool
                - confidence: Dict[str, float]
        
        Returns:
            dict with keys:
                - fused_risk_score: float 0-1
                - decision_category: str
                - confidence: float
                - cyber_contribution: float
                - physical_contribution: float
                - is_contradiction: bool
                - explanation: str
                - timestamp: float
        """
        # Extract scores
        cyber_score = cyber_result.get('risk_score', 0.0)
        physical_score = vision_result.get('risk_score', 0.0)
        
        # Weighted fusion
        fused_score = (
            self.WEIGHT_CYBER * cyber_score +
            self.WEIGHT_PHYSICAL * physical_score
        )
        
        # Detect contradiction (one high, one low)
        is_contradiction = self._is_contradiction(cyber_score, physical_score)
        
        # Classify decision category
        category = self._classify_decision(cyber_score, physical_score, cyber_result, vision_result)
        
        # Apply category multiplier
        adjusted_score = min(1.0, fused_score * self.CATEGORY_MULTIPLIERS.get(category, 1.0))
        
        # Generate explanation
        explanation = self._generate_explanation(category, cyber_result, vision_result, is_contradiction)
        
        return {
            'fused_risk_score': float(adjusted_score),
            'decision_category': category,
            'confidence': float(max(
                cyber_result.get('confidence', 0.5),
                max(vision_result.get('confidence', {}).values(), default=0.5)
            )),
            'cyber_contribution': float(self.WEIGHT_CYBER * cyber_score),
            'physical_contribution': float(self.WEIGHT_PHYSICAL * physical_score),
            'is_contradiction': bool(is_contradiction),
            'explanation': explanation,
            'timestamp': time.time()
        }
    
    def _is_contradiction(self, cyber: float, physical: float) -> bool:
        """
        Detect if cyber and physical signals contradict.
        Contradiction = one is high, other is low (suggests spoofing or sensor fault)
        """
        cyber_high = cyber > self.CYBER_HIGH
        physical_high = physical > self.PHYSICAL_HIGH
        
        # Contradiction: exactly one is high
        return (cyber_high and not physical_high) or (physical_high and not cyber_high)
    
    def _classify_decision(self, cyber: float, physical: float, 
                          cyber_result: Dict, vision_result: Dict) -> str:
        """
        Classify the fused decision into a category.
        """
        cyber_high = cyber > self.CYBER_HIGH
        physical_high = physical > self.PHYSICAL_HIGH
        
        if cyber_high and physical_high:
            return 'CYBER_PHYSICAL_EVENT'
        elif cyber_high and not physical_high:
            # Check if cyber attack type suggests data manipulation
            attack_type = cyber_result.get('attack_type', 'unknown')
            if attack_type in ['FDIA', 'data_injection', 'spoofing']:
                return 'CYBER_ATTACK'  # Likely false data injection
            return 'CYBER_ATTACK'
        elif not cyber_high and physical_high:
            # Physical threat without cyber anomaly
            labels = vision_result.get('labels', [])
            if 'intruder' in labels or 'no_ppe' in labels:
                return 'INTRUSION'
            return 'PHYSICAL_FAULT'
        else:
            return 'NORMAL'
    
    def _generate_explanation(self, category: str, cyber_result: Dict, 
                             vision_result: Dict, is_contradiction: bool) -> str:
        """
        Generate human-readable explanation for the decision.
        """
        explanations = {
            'CYBER_PHYSICAL_EVENT': (
                "⚠️ CRITICAL: Coordinated cyber-physical event detected. "
                f"Cyber indicators ({cyber_result.get('attack_type', 'anomaly')}) "
                f"combined with physical threats ({vision_result.get('labels', [])}). "
                "Immediate response required."
            ),
            'CYBER_ATTACK': (
                "🔴 Cyber attack detected without physical corroboration. "
                f"Attack type: {cyber_result.get('attack_type', 'unknown')}. "
                "Likely false data injection or remote intrusion."
            ),
            'PHYSICAL_FAULT': (
                "🟠 Physical anomaly detected. "
                f"Vision detected: {vision_result.get('labels', [])}. "
                "No concurrent cyber attack indicators. Possible equipment failure."
            ),
            'INTRUSION': (
                "🟡 Security intrusion detected. "
                f"Unauthorized presence: {vision_result.get('labels', [])}. "
                "Cyber systems normal. Dispatch security."
            ),
            'NORMAL': (
                "🟢 All systems normal. "
                f"Cyber risk: {cyber_result.get('risk_score', 0):.2f}, "
                f"Physical risk: {vision_result.get('risk_score', 0):.2f}."
            )
        }
        
        base = explanations.get(category, "Unknown category")
        
        if is_contradiction:
            base += " ⚠️ Signal contradiction detected - verify sensors."
        
        return base
    
    def batch_fuse(self, cyber_batch: list, vision_batch: list) -> list:
        """
        Fuse multiple timestamped pairs.
        
        Args:
            cyber_batch: List of cyber results
            vision_batch: List of vision results
        
        Returns:
            List of fused results
        """
        assert len(cyber_batch) == len(vision_batch), "Batches must be same length"
        return [self.fuse(c, v) for c, v in zip(cyber_batch, vision_batch)]


# Convenience function
def quick_fuse(cyber_result: Dict[str, Any], vision_result: Dict[str, Any]) -> Dict[str, Any]:
    """Quick fusion for testing."""
    fusion = MultimodalFusion()
    return fusion.fuse(cyber_result, vision_result)


if __name__ == '__main__':
    # Demo test
    fusion = MultimodalFusion()
    
    # Test case 1: FDIA without physical threat
    cyber1 = {'risk_score': 0.85, 'attack_type': 'FDIA', 'confidence': 0.9}
    vision1 = {'risk_score': 0.1, 'labels': ['normal'], 'is_physical_threat': False, 'confidence': {'normal': 0.95}}
    result1 = fusion.fuse(cyber1, vision1)
    print(f"Test 1 (FDIA): {result1['decision_category']} - {result1['fused_risk_score']:.2f}")
    print(f"  {result1['explanation']}\n")
    
    # Test case 2: Coordinated attack
    cyber2 = {'risk_score': 0.9, 'attack_type': 'DoS', 'confidence': 0.88}
    vision2 = {'risk_score': 0.85, 'labels': ['fire_smoke'], 'is_physical_threat': True, 'confidence': {'fire_smoke': 0.92}}
    result2 = fusion.fuse(cyber2, vision2)
    print(f"Test 2 (Coordinated): {result2['decision_category']} - {result2['fused_risk_score']:.2f}")
    print(f"  {result2['explanation']}\n")
    
    # Test case 3: Normal
    cyber3 = {'risk_score': 0.1, 'attack_type': 'normal', 'confidence': 0.95}
    vision3 = {'risk_score': 0.05, 'labels': ['normal'], 'is_physical_threat': False, 'confidence': {'normal': 0.98}}
    result3 = fusion.fuse(cyber3, vision3)
    print(f"Test 3 (Normal): {result3['decision_category']} - {result3['fused_risk_score']:.2f}")
    print(f"  {result3['explanation']}")
