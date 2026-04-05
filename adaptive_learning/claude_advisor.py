"""
Claude Advisor — Physiological signal analyst for poker tell detection.

Uses the Claude Code CLI (already installed, no API key) via subprocess.
Python pre-computes rich signal statistics; Claude reasons over them and
returns a structured verdict stored alongside each showdown.

No API key required — uses the existing Claude Code OAuth session.
"""

import json
import os
import re
import shutil
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

_CLAUDE_EXE: str = (
    os.environ.get("CLAUDE_EXE")
    or shutil.which("claude")
    or ""
)
_CLAUDE_MODEL       = os.environ.get("CLAUDE_CLI_MODEL", "haiku")
_SUBPROCESS_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "45"))


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = (
    "You are a physiological signal analyst for a poker tell-detection system. "
    "You receive pre-computed statistics from a player's facial physiological signals "
    "(heart rate via rPPG, stress score, facial Action Units) recorded during a poker hand. "
    "Your job is to determine whether the player was BLUFFING or held a STRONG HAND.\n\n"
    "Key rules:\n"
    "- Every player is different. Use their personal baseline and history.\n"
    "- Bluffing typically: elevated HR, stress spike, AU23 (lip tighten), AU4 (brow furrow), "
    "AU20 (lip stretch), suppressed AU12 (smile).\n"
    "- Strong hands: calm HR near baseline, stable AUs, low variance.\n"
    "- If HR is marked NOISY, rely on stress and AU signals instead.\n"
    "- Tells appearing in the last third of a hand (when pressure peaks) are more reliable.\n"
    "- If history is available, describe whether this player's pattern matches their usual tells.\n\n"
    "Respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON:\n"
    '{"prediction":"BLUFFING"|"STRONG","confidence":0.0-1.0,"p_bluff":0.0-1.0,'
    '"primary_signals":["signal1","signal2",...],"reasoning":"one paragraph"}'
)

# ---------------------------------------------------------------------------
# ClaudeAdvisor
# ---------------------------------------------------------------------------

class ClaudeAdvisor:
    """
    Physiological analyst that calls Claude via the local CLI (no API key).

    Python handles all statistical computation. Claude handles reasoning —
    weighing signals against each other, cross-referencing history, and
    producing a calibrated prediction with explicit justification.
    """

    def __init__(self, profile_store):
        self.store = profile_store
        if self._cli_available():
            print(f"[ClaudeAdvisor] Ready — model: {_CLAUDE_MODEL}")
        else:
            print(f"[ClaudeAdvisor] WARNING: Claude CLI not found at {_CLAUDE_EXE}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_hand(
        self,
        player_id: str,
        hand_buffer: List[Dict],
        baseline: Dict,
        showdown_history: List[Dict],
    ) -> Optional[Dict]:
        """
        Analyse a full hand buffer and return a structured prediction.

        Python computes all signal statistics; Claude reasons over them.

        Returns dict: prediction, confidence, p_bluff, primary_signals, reasoning
        Returns None if CLI unavailable or call fails.
        """
        if not hand_buffer:
            return None

        prompt = self._build_prompt(player_id, hand_buffer, baseline, showdown_history)
        raw = self._call_claude_cli(prompt)
        if raw is None:
            return None
        return self._parse_response(raw)

    def analyze_baseline(self, samples: List[Dict]) -> Optional[Dict]:
        """
        Analyse raw calibration samples to identify clean resting frames.

        Downsamples to at most 150 frames and asks Claude to flag noisy frames
        (talking, moving, head turns) so the baseline can be recomputed from
        only the clean subset.

        Returns dict with keys: clean_pct, clean_ranges, verdict, notes, step
        Returns None on failure.
        """
        if not samples:
            return None

        step = max(1, len(samples) // 150)
        frames = []
        for orig_i, s in enumerate(samples):
            if orig_i % step != 0:
                continue
            ds_i = orig_i // step
            au = s.get('au', {})
            frames.append({
                'i':    ds_i,
                'hr':   round(float(s['hr'])   if s['hr']   is not None else 0.0, 1),
                'stress': round(float(s['stress']) if s['stress'] is not None else 0.0, 3),
                'AU12': round(au.get('AU12', 0.0), 2),
                'AU26': round(au.get('AU26', 0.0), 2),
                'AU45': round(au.get('AU45', 0.0), 2),
                'AU1':  round(au.get('AU1',  0.0), 2),
                'AU2':  round(au.get('AU2',  0.0), 2),
            })

        prompt = (
            f"TASK: Baseline quality filtering.\n"
            f"This is a {len(samples)}-frame resting baseline calibration window downsampled to "
            f"{len(frames)} frames (step={step}).\n"
            f"Identify which frames are clean resting frames vs noisy frames.\n\n"
            f"NOISE RULES:\n"
            f"- Talking/laughing: AU12 AND AU26 both > 0.15\n"
            f"- Head turn/surprise: AU1 AND AU2 both > 0.2\n"
            f"- HR spike: HR more than 12 BPM from the window median\n\n"
            f"FRAMES (downsampled index, physiological values):\n"
            f"{json.dumps(frames)}\n\n"
            f"Return ONLY a valid JSON object with these keys:\n"
            f'  "clean_pct": fraction of frames that are clean (0.0-1.0)\n'
            f'  "clean_ranges": list of [ds_start, ds_end] inclusive index pairs covering clean frames\n'
            f'  "verdict": "GOOD" (>0.8 clean), "FAIR" (0.5-0.8), or "POOR" (<0.5)\n'
            f'  "notes": brief one-sentence description of what noise was found\n'
            f'Example: {{"clean_pct":0.85,"clean_ranges":[[0,12],[18,44]],"verdict":"GOOD","notes":"brief"}}'
        )

        raw = self._call_claude_cli(prompt)
        if raw is None:
            return None

        clean = re.sub(r'```(?:json)?', '', raw).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if not match:
            print(f'[ClaudeAdvisor] analyze_baseline: no JSON found in response: {raw[:200]}')
            return None
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f'[ClaudeAdvisor] analyze_baseline: JSON parse error: {e}')
            return None

        result['step'] = step
        result['original_n'] = len(samples)
        return result

    def generate_cold_start_priors(self, baseline_dict: Dict, player_id: str) -> Optional[Dict]:
        """
        Generate per-context-bucket bluff probability priors from a player's
        physiological baseline, for use before any showdown data exists.

        Returns dict mapping bucket keys (HH, HM, ..., LL) to p_bluff floats,
        or None on failure.
        """
        hr     = baseline_dict.get('hr', 70)
        stress = baseline_dict.get('stress')
        aus    = baseline_dict.get('au', {})

        stress_str = f'{stress:.3f}' if stress is not None else 'N/A'
        sig_aus = {k: round(v, 3) for k, v in aus.items() if abs(v) > 0.05}

        prompt = (
            f"TASK: Cold-start bluff priors for a new poker player.\n"
            f"PLAYER: {player_id[:8]}\n\n"
            f"RESTING PHYSIOLOGICAL BASELINE:\n"
            f"  HR: {hr:.1f} BPM\n"
            f"  Stress: {stress_str}\n"
            f"  Significant resting AUs (abs > 0.05): {json.dumps(sig_aus) if sig_aus else 'none'}\n\n"
            f"CONTEXT BUCKETS: First letter = HR deviation (H=high, M=medium, L=low), "
            f"Second letter = stress deviation (H/M/L).\n\n"
            f"INTERPRETATION GUIDE:\n"
            f"- High resting HR or stress = naturally anxious player; physiological signals are "
            f"less discriminative for them (priors closer to 0.5).\n"
            f"- Low resting HR = calm baseline; stress/HR spikes during play are more meaningful "
            f"(priors can be further from 0.5).\n"
            f"- MM bucket (medium HR delta, medium stress delta) should be near 0.5 (ambiguous).\n\n"
            f"Estimate the probability this player is BLUFFING in each context bucket.\n"
            f"Return ONLY a valid JSON object:\n"
            f'{{"HH":X,"HM":X,"HL":X,"MH":X,"MM":X,"ML":X,"LH":X,"LM":X,"LL":X}}\n'
            f'where each value is a float between 0.1 and 0.9.'
        )

        raw = self._call_claude_cli(prompt)
        if raw is None:
            return None

        clean = re.sub(r'```(?:json)?', '', raw).strip()
        match = re.search(r'\{[^}]+\}', clean, re.DOTALL)
        if not match:
            print(f'[ClaudeAdvisor] generate_cold_start_priors: no JSON found: {raw[:200]}')
            return None
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f'[ClaudeAdvisor] generate_cold_start_priors: JSON parse error: {e}')
            return None

        all_buckets = ['HH', 'HM', 'HL', 'MH', 'MM', 'ML', 'LH', 'LM', 'LL']
        validated = {}
        for bucket in all_buckets:
            if bucket in result:
                try:
                    validated[bucket] = max(0.1, min(0.9, float(result[bucket])))
                except (TypeError, ValueError):
                    pass

        if len(validated) < 5:
            print(f'[ClaudeAdvisor] generate_cold_start_priors: too few valid buckets ({len(validated)})')
            return None

        # Fill any missing buckets with 0.5
        for bucket in all_buckets:
            validated.setdefault(bucket, 0.5)

        return validated

    def generate_session_debrief(
        self,
        player_id: str,
        showdowns: List[Dict],
        personality_state: Dict,
        context_summary: Dict,
    ) -> Optional[str]:
        """
        Generate a 2-3 sentence debrief note about this opponent after they leave.

        Returns plain text string (stripped), or None on failure.
        """
        if not showdowns:
            return None

        total     = len(showdowns)
        bluff_ct  = sum(1 for h in showdowns if h.get('actual_result') == 'BLUFFING')
        correct   = sum(1 for h in showdowns if h.get('was_correct'))
        bluff_rate = bluff_ct / total if total else 0.0
        accuracy   = correct  / total if total else 0.0

        # Compact recent hand list (last 10)
        recent = showdowns[:10]
        recent_lines = []
        for h in recent:
            recent_lines.append(
                f"  bucket={h.get('context_bucket','?')} actual={h.get('actual_result','?')} "
                f"correct={h.get('was_correct',0)} "
                f"hr_delta={h.get('hr_delta',0):+.1f} stress_delta={h.get('stress_delta',0):+.3f}"
            )

        # Top 5 context buckets by sample count
        top_buckets = []
        if context_summary:
            sorted_buckets = sorted(
                context_summary.items(),
                key=lambda kv: kv[1].get('samples', 0) if isinstance(kv[1], dict) else 0,
                reverse=True,
            )
            for bucket, info in sorted_buckets[:5]:
                if isinstance(info, dict):
                    top_buckets.append(
                        f"  {bucket}: p_bluff={info.get('p_bluff', 0.5):.2f} samples={info.get('samples', 0)}"
                    )

        ps = personality_state or {}
        prompt = (
            f"TASK: Write a 2-3 sentence session debrief note about a poker opponent.\n"
            f"Write in second-person observer style: 'This player bluffs...'\n\n"
            f"SESSION STATS:\n"
            f"  Hands observed: {total}\n"
            f"  Bluff rate: {bluff_rate:.0%} ({bluff_ct}/{total})\n"
            f"  Model accuracy: {accuracy:.0%} ({correct}/{total})\n\n"
            f"PERSONALITY STATE:\n"
            f"  bluff_base_rate: {ps.get('bluff_base_rate', 'N/A')}\n"
            f"  stress_bluff_slope: {ps.get('stress_bluff_slope', 'N/A')}\n"
            f"  hr_bluff_slope: {ps.get('hr_bluff_slope', 'N/A')}\n"
            f"  consistency: {ps.get('consistency', 'N/A')}\n\n"
            f"RECENT HANDS (newest first):\n" + '\n'.join(recent_lines) + '\n\n'
            f"TOP CONTEXT BUCKETS:\n" + ('\n'.join(top_buckets) if top_buckets else '  none') + '\n\n'
            f"Address: which physiological signal is most predictive for this player specifically, "
            f"their bluff tendency, and any timing pattern. "
            f"Return ONLY the plain text note (no JSON, no markdown)."
        )

        raw = self._call_claude_cli(prompt)
        if raw is None:
            return None
        return raw.strip() or None

    # ------------------------------------------------------------------
    # Prompt construction — pre-computed stats, not raw data
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        player_id: str,
        buf: List[Dict],
        baseline: Dict,
        history: List[Dict],
    ) -> str:
        n = len(buf)
        secs = n / 30.0
        stress_ok = baseline.get("stress") is not None

        lines = [
            f"PLAYER: {player_id[:8]}",
            f"HAND:   {n} frames (~{secs:.0f}s at 30 fps)",
            f"BASELINE HR: {baseline.get('hr', 'N/A'):.1f} BPM | "
            f"STRESS BASELINE: {'calibrated' if stress_ok else 'NOT CALIBRATED — stress deltas unreliable'}",
            "",
            "=== SIGNAL ANALYSIS ===",
        ]

        # HR analysis
        hr = self._series(buf, "hr")
        hr_var = float(np.var(hr))
        hr_noisy = hr_var > 100.0
        lines += self._format_signal("HR (rPPG)", hr,
                                      note="⚠ HIGH NOISE — weight AU/stress signals more" if hr_noisy else "")

        # Stress analysis
        if stress_ok:
            st = self._series(buf, "stress")
            lines += self._format_signal("STRESS", st)

        # AU analysis — only show AUs with meaningful activation
        au_names = sorted({au for f in buf for au in f.get("au_delta", {})})
        active_aus = []
        for au in au_names:
            s = self._au_series(buf, au)
            if float(np.percentile(np.abs(s), 90)) > 0.05:  # skip near-zero AUs
                active_aus.append((au, s))

        if active_aus:
            lines.append("\nACTIVE ACTION UNITS (p90 abs > 0.05):")
            for au, s in active_aus:
                lines += self._format_signal(au, s, indent="  ")

        # History
        lines += ["", "=== PLAYER HISTORY ==="]
        if history:
            bluff_hands  = [h for h in history if h.get("actual_result") == "BLUFFING"]
            strong_hands = [h for h in history if h.get("actual_result") == "STRONG"]

            lines.append(f"Total hands on record: {len(history)} "
                         f"({len(bluff_hands)} bluffs, {len(strong_hands)} strong)")

            if bluff_hands:
                avg_hr_bluff = np.mean([h["hr_delta"] for h in bluff_hands if h.get("hr_delta")])
                avg_st_bluff = np.mean([h["stress_delta"] for h in bluff_hands if h.get("stress_delta")])
                lines.append(f"When BLUFFING avg: HR delta={avg_hr_bluff:+.2f}, stress delta={avg_st_bluff:+.3f}")

            if strong_hands:
                avg_hr_str = np.mean([h["hr_delta"] for h in strong_hands if h.get("hr_delta")])
                avg_st_str = np.mean([h["stress_delta"] for h in strong_hands if h.get("stress_delta")])
                lines.append(f"When STRONG avg:   HR delta={avg_hr_str:+.2f}, stress delta={avg_st_str:+.3f}")

            correct = sum(1 for h in history if h.get("was_correct"))
            lines.append(f"Model accuracy on this player so far: {correct}/{len(history)}")
        else:
            lines.append("No prior history — cold start, rely on signal patterns only.")

        lines += ["", "Based on the above analysis, return your verdict as a JSON object."]
        return "\n".join(lines)

    def _format_signal(self, name: str, s: np.ndarray, note: str = "", indent: str = "") -> List[str]:
        if len(s) == 0:
            return [f"{indent}{name}: no data"]
        n = len(s)
        t1, t2 = s[:n//3], s[2*n//3:]
        trend = float(np.polyfit(np.arange(n), s, 1)[0]) if n > 1 else 0.0
        sustained = float(np.mean(s > float(np.std(s)))) if np.std(s) > 0 else 0.0
        late_mean = float(np.mean(t2)) if len(t2) else 0.0
        early_mean = float(np.mean(t1)) if len(t1) else 0.0

        lines = [
            f"{indent}{name}:{f'  [{note}]' if note else ''}",
            f"{indent}  mean_delta={np.mean(s):+.3f}  std={np.std(s):.3f}  "
            f"peak_delta={s[np.argmax(np.abs(s))]:+.3f}  p90_abs={np.percentile(np.abs(s),90):.3f}",
            f"{indent}  early_mean={early_mean:+.3f}  late_mean={late_mean:+.3f}  "
            f"trend={'rising' if trend>0.0005 else 'falling' if trend<-0.0005 else 'flat'}  "
            f"sustained_elevation={sustained:.0%}",
        ]
        return lines

    # ------------------------------------------------------------------
    # Signal extraction helpers
    # ------------------------------------------------------------------

    def _series(self, buf: List[Dict], key: str) -> np.ndarray:
        k = "hr_delta" if key == "hr" else "stress_delta"
        return np.array([f.get(k, 0.0) for f in buf], dtype=float)

    def _au_series(self, buf: List[Dict], au: str) -> np.ndarray:
        return np.array([f.get("au_delta", {}).get(au, 0.0) for f in buf], dtype=float)

    # ------------------------------------------------------------------
    # Claude CLI call
    # ------------------------------------------------------------------

    def _cli_available(self) -> bool:
        try:
            r = subprocess.run([_CLAUDE_EXE, "--version"], capture_output=True, text=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _call_claude_cli(self, prompt: str) -> Optional[str]:
        """
        Call Claude Code CLI in non-interactive mode.
        No API key required — uses the existing OAuth session.

        Prompt is passed via stdin (avoids Windows 8 KB command-line limit).
        --no-session-persistence keeps this call out of your session history.
        --tools "" disables all built-in tools (filesystem, bash, etc.)
        --json-schema enforces structured output so parsing never fails.
        """
        try:
            result = subprocess.run(
                [
                    _CLAUDE_EXE,
                    "--print",
                    "--model",                 _CLAUDE_MODEL,
                    "--append-system-prompt",  _SYSTEM,
                    "--output-format",         "text",
                    "--no-session-persistence",
                    "--tools",                 "",
                ],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
            if result.returncode != 0:
                print(f"[ClaudeAdvisor] CLI error (code {result.returncode}): {result.stderr[:200]}")
                return None
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print("[ClaudeAdvisor] CLI timed out")
            return None
        except Exception as e:
            print(f"[ClaudeAdvisor] CLI call failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> Optional[Dict]:
        """Extract JSON from Claude's response, tolerating minor formatting noise."""
        clean = re.sub(r"```(?:json)?", "", raw).strip()

        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            print(f"[ClaudeAdvisor] Could not find JSON in response: {raw[:200]}")
            return None

        try:
            result = json.loads(match.group())
        except json.JSONDecodeError as e:
            print(f"[ClaudeAdvisor] JSON parse error: {e} | raw: {raw[:200]}")
            return None

        required = {"prediction", "confidence", "p_bluff", "primary_signals", "reasoning"}
        if not required.issubset(result):
            print(f"[ClaudeAdvisor] Missing keys in response: {required - result.keys()}")
            return None

        result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
        result["p_bluff"]    = max(0.0, min(1.0, float(result["p_bluff"])))

        print(
            f"[ClaudeAdvisor] {result['prediction']} "
            f"(p_bluff={result['p_bluff']:.2f}, conf={result['confidence']:.2f})"
        )
        return result
