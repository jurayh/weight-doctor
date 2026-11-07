"""Minimal Meta Model API client for the kill-3 run. Not run without approval.

Pattern follows the proven Sluice live benchmark: OpenAI-compatible
base URL https://api.meta.ai/v1, POST /responses, model id passed in,
credential via the dynamic-credentials surrogate helper (never
printed, logged, or written to any output file; sent only to
api.meta.ai). Token counts and dollars come from the provider's
`usage` field, never from estimates.

Cap enforcement: before each call, the estimated cost (prompt chars
/ 4 x 1.4 upper bound, plus the max-output cap) is checked against
the remaining budget; a call that would cross the cap is refused
before it is made. After each call the measured usage is added to
the running tally, and the tally is returned with every result so
the runner can stop the moment the cap is reached.
"""

import json
import math
import urllib.request

BASE_URL = "https://api.meta.ai/v1"
MODEL_CONTRIBUTOR = "muse-spark-1.3-contributor"
MODEL_STANDARD = "muse-spark-1.3"
# Contributor tier: $0.10 input / $0.20 output per 1M tokens.
IN_RATE_CONTRIBUTOR = 0.10 / 1_000_000
OUT_RATE_CONTRIBUTOR = 0.20 / 1_000_000
# Standard tier, for the comparison price only: $1.25 / $4.25 per 1M.
IN_RATE_STANDARD = 1.25 / 1_000_000
OUT_RATE_STANDARD = 4.25 / 1_000_000
# 4096, not 1024: the output budget must cover reasoning tokens
# PLUS the answer. The fixed-client smoke run showed the model
# spending 1022 of 1024 output tokens on reasoning (effort low)
# and returning status "incomplete" with no answer text at all.
# A JSON diff needs only a few hundred tokens, so 4096 leaves
# ~3000 for reasoning with headroom. Worst-case cost at this cap
# is still ~$0.007 on contributor tier (see KILL3_PLAN.md).
MAX_OUTPUT_TOKENS = 4096
BPE_UPPER = 1.4  # chars/4 underestimates tokens; x1.4 is the upper bound


class CapExceeded(RuntimeError):
    pass


def extract_text(payload):
    """Extract answer text from a Responses API payload.

    Raises RuntimeError with a truncated shape dump when no
    output_text part exists (e.g. the output budget was spent on
    reasoning items), so the failure is diagnosable from the record.
    """
    text = ""
    for item in payload.get("output", []):
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text"):
                text += part.get("text", "")
    if not text:
        shape = str({k: (str(v)[:300]) for k, v in payload.items()})[:2000]
        raise RuntimeError(f"no output_text in response; shape: {shape}")
    return text


class MetaClient:
    """One model id, one cap, one running tally. Cap is in USD."""

    def __init__(self, model=MODEL_CONTRIBUTOR, cap_usd=0.10,
                 max_output_tokens=MAX_OUTPUT_TOKENS):
        self.model = model
        self.cap_usd = cap_usd
        self.max_output_tokens = max_output_tokens
        if "contributor" in model:
            self.in_rate, self.out_rate = (IN_RATE_CONTRIBUTOR, OUT_RATE_CONTRIBUTOR)
        else:
            self.in_rate, self.out_rate = (IN_RATE_STANDARD, OUT_RATE_STANDARD)
        self.spent_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._surrogate = None

    def _get_surrogate(self):
        if self._surrogate is None:
            import sys
            sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
            from dynamic_credentials import dynamic_credential_entry
            entry = dynamic_credential_entry("custom.meta-model-api", "access_token")
            surrogate = str(entry["surrogate"]).strip()
            if not surrogate.startswith("hsurr:"):
                raise RuntimeError(
                    "meta-model-api credential surrogate not in expected form; refusing to call.")
            self._surrogate = surrogate
        return self._surrogate

    def estimate_cost_usd(self, prompt):
        """Upper-bound cost of one call, before it is made."""
        in_tokens = math.ceil(len(prompt) / 4 * BPE_UPPER)
        return in_tokens * self.in_rate + self.max_output_tokens * self.out_rate

    def complete(self, prompt):
        """One call. Returns {"text": str, "usage": {...}, "spent_usd": float}."""
        est = self.estimate_cost_usd(prompt)
        if self.spent_usd + est > self.cap_usd:
            raise CapExceeded(
                f"refusing call: estimated ${est:.4f} would take spend to "
                f"${self.spent_usd + est:.4f}, over the ${self.cap_usd:.2f} cap.")
        # Request shape follows the proven Sluice live-benchmark
        # worker: typed input items, temperature 0, reasoning effort
        # low. The first smoke-test run omitted both settings and got
        # 1024 output tokens with zero extractable text on both calls:
        # with default reasoning the model spends the output budget on
        # reasoning items (which carry no output_text) before any
        # answer text is produced.
        body = json.dumps({
            "model": self.model,
            "input": [{"type": "message", "role": "user",
                       "content": [{"type": "input_text", "text": prompt}]}],
            "stream": False,
            "temperature": 0,
            "reasoning": {"effort": "low"},
            "max_output_tokens": self.max_output_tokens,
        }).encode()
        req = urllib.request.Request(
            BASE_URL + "/responses", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._get_surrogate()}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode())
        # Tally FIRST, before extracting text: a call that returns no
        # answer text was still made and billed. (The first fixed-
        # client run recorded $0.00 because the tally happened after
        # extraction; the provider had in fact billed 282 in / 1024
        # out tokens.)
        usage = payload.get("usage") or {}
        in_tok = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        out_tok = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        cost = in_tok * self.in_rate + out_tok * self.out_rate
        self.spent_usd += cost
        self.prompt_tokens += in_tok
        self.completion_tokens += out_tok
        self.calls += 1
        usage_out = {"input_tokens": in_tok, "output_tokens": out_tok,
                     "cost_usd": cost}
        try:
            text = extract_text(payload)
        except RuntimeError as e:
            e.usage = usage_out
            raise
        return {"text": text, "usage": usage_out,
                "spent_usd": self.spent_usd}
