"""
Prior Generator Module

Uses a Small Language Model (SLM) to generate initial bluff probability
priors for new players, solving the cold-start problem.

Note: This module requires additional dependencies:
    pip install transformers torch bitsandbytes
"""

import json
import re
from typing import Dict, Optional


class PriorGenerator:
    """
    Generates bluff probability priors using a local Small Language Model.
    
    Uses Microsoft Phi-3-mini or similar model to analyze player characteristics
    and generate initial probability estimates for each context bucket.
    
    Usage:
        generator = PriorGenerator()
        priors = generator.generate_prior({
            'baseline_hr': 75,
            'blink_rate': 12,
            'fidgeting': 'low'
        })
        # Returns: {'HH': 0.7, 'HL': 0.5, 'MM': 0.45, ...}
    """
    
    # Default model (quantized for CPU)
    DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"
    
    def __init__(self, model_id: str = None, device: str = "auto"):
        """
        Initialize the prior generator.
        
        Args:
            model_id: HuggingFace model ID (default: Phi-3-mini)
            device: Device to run on ("auto", "cpu", "cuda")
        """
        self.model_id = model_id or self.DEFAULT_MODEL
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def _lazy_init(self):
        """Lazy initialization to defer loading until first use."""
        if self._initialized:
            return True
            
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"[PriorGenerator] Loading model: {self.model_id}")
            print(f"[PriorGenerator] This may take a few minutes on first run...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=self.device,
                load_in_4bit=True,  # Quantize for speed
                trust_remote_code=True
            )
            
            self._initialized = True
            print(f"[PriorGenerator] Model loaded successfully")
            return True
            
        except ImportError as e:
            print(f"[PriorGenerator] Missing dependencies: {e}")
            print("[PriorGenerator] Run: pip install transformers torch bitsandbytes")
            return False
        except Exception as e:
            print(f"[PriorGenerator] Failed to load model: {e}")
            return False
    
    def generate_prior(self, observed_features: Dict) -> Dict[str, float]:
        """
        Generate bluff probability priors for a new player.
        
        Args:
            observed_features: Dict of observed player characteristics
                - baseline_hr: Resting heart rate
                - baseline_stress: Baseline stress level
                - blink_rate: Blinks per minute
                - fidgeting: 'low', 'medium', 'high'
                - age_estimate: Approximate age
                - experience_level: 'novice', 'intermediate', 'expert'
                
        Returns:
            Dict mapping context bucket to P(bluffing)
            e.g., {'HH': 0.7, 'HM': 0.6, 'HL': 0.5, ...}
        """
        if not self._lazy_init():
            return self._fallback_priors()
        
        prompt = self._build_prompt(observed_features)
        
        try:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            priors = self._parse_response(response)
            
            if priors:
                return priors
            else:
                print("[PriorGenerator] Failed to parse response, using fallback")
                return self._fallback_priors()
                
        except Exception as e:
            print(f"[PriorGenerator] Generation failed: {e}")
            return self._fallback_priors()
    
    def _build_prompt(self, features: Dict) -> str:
        """Build the prompt for the LLM."""
        prompt = f"""You are a poker psychology expert analyzing a player's likely bluffing patterns.

Player Observations:
- Baseline heart rate: {features.get('baseline_hr', 'unknown')} BPM
- Baseline stress level: {features.get('baseline_stress', 'unknown')}
- Blink rate: {features.get('blink_rate', 'unknown')}/min
- Fidgeting level: {features.get('fidgeting', 'unknown')}
- Estimated experience: {features.get('experience_level', 'unknown')}

Based on these observations, estimate the probability of bluffing in each stress scenario.
Output ONLY a JSON object with probabilities (0.0 to 1.0):

{{"HH": X.X, "HM": X.X, "HL": X.X, "MH": X.X, "MM": X.X, "ML": X.X, "LH": X.X, "LM": X.X, "LL": X.X}}

Where the first letter is HR deviation (H=high, M=medium, L=low) and second is stress deviation.
For example, "HH" means both heart rate AND stress are elevated from baseline.

Output ONLY the JSON, no explanation:"""
        
        return prompt
    
    def _parse_response(self, response: str) -> Optional[Dict[str, float]]:
        """Parse LLM response to extract priors."""
        try:
            # Find JSON object in response
            match = re.search(r'\{[^}]+\}', response)
            if not match:
                return None
            
            json_str = match.group()
            priors = json.loads(json_str)
            
            # Validate and clean
            valid_buckets = {'HH', 'HM', 'HL', 'MH', 'MM', 'ML', 'LH', 'LM', 'LL'}
            cleaned = {}
            
            for bucket, value in priors.items():
                bucket = bucket.upper()
                if bucket in valid_buckets:
                    # Clamp to valid probability range
                    cleaned[bucket] = max(0.0, min(1.0, float(value)))
            
            # Fill missing buckets with 0.5
            for bucket in valid_buckets:
                if bucket not in cleaned:
                    cleaned[bucket] = 0.5
            
            return cleaned
            
        except Exception as e:
            print(f"[PriorGenerator] Parse error: {e}")
            return None
    
    def _fallback_priors(self) -> Dict[str, float]:
        """Return default priors when LLM is unavailable."""
        # Heuristic: high stress usually correlates with bluffing
        return {
            'HH': 0.65,  # High HR + High stress = likely bluffing
            'HM': 0.55,
            'HL': 0.50,
            'MH': 0.55,
            'MM': 0.50,  # Neutral = uncertain
            'ML': 0.45,
            'LH': 0.50,
            'LM': 0.45,
            'LL': 0.35   # Low HR + Low stress = likely strong hand
        }
    
    def is_available(self) -> bool:
        """Check if the model is available."""
        try:
            import transformers
            import torch
            return True
        except ImportError:
            return False


class SimplePriorGenerator:
    """
    Lightweight prior generator using heuristics instead of LLM.
    
    Use this as a fallback when the full LLM is not available.
    """
    
    def generate_prior(self, observed_features: Dict) -> Dict[str, float]:
        """Generate priors using simple heuristics."""
        
        # Base priors
        priors = {
            'HH': 0.65, 'HM': 0.55, 'HL': 0.50,
            'MH': 0.55, 'MM': 0.50, 'ML': 0.45,
            'LH': 0.50, 'LM': 0.45, 'LL': 0.35
        }
        
        # Adjust based on experience level
        experience = observed_features.get('experience_level', 'unknown')
        if experience == 'expert':
            # Experts mask better, harder to read
            for k in priors:
                priors[k] = 0.5 + (priors[k] - 0.5) * 0.5  # Move toward 0.5
        elif experience == 'novice':
            # Novices are more readable
            for k in priors:
                priors[k] = 0.5 + (priors[k] - 0.5) * 1.5  # Amplify
                priors[k] = max(0.0, min(1.0, priors[k]))
        
        # Adjust based on baseline HR
        baseline_hr = observed_features.get('baseline_hr', 70)
        if baseline_hr and baseline_hr > 80:
            # High baseline HR = naturally anxious, stress signals less meaningful
            for k in priors:
                priors[k] = 0.5 + (priors[k] - 0.5) * 0.7
        
        return priors
