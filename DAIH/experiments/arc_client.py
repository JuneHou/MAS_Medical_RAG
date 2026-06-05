"""
ARC API client — drop-in replacement for GPTClient used by run_condition_*.py.

Wraps VT ARC's OpenAI-compatible endpoint (https://llm-api.arc.vt.edu/api/v1/) with
a sliding-window rate limiter that honors ARC fairshare (30/min, 1000/hr, 3000/3hr)
and exponential-backoff retries. Exposes the same `.generate(prompt, max_tokens,
temperature) -> str` interface as KARE/gpt/src/gpt_utils.py:GPTClient so the
existing process_sample_condition_*() functions work unchanged.

Auth:
    set -a; source /data/wang/junh/.cache/keys/arc_llm_api.sh; set +a
    export ARC_LLM_API_KEY="$API_KEY"

Reference implementation borrowed from:
    /data/wang/junh/githubs/MAST/eval/full_run_eval_graph_inject_api_arc.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Optional

from openai import OpenAI


ARC_BASE_URL = "https://llm-api.arc.vt.edu/api/v1/"
DEFAULT_MODEL = "gpt-oss-120b"

# ARC fairshare ceilings
DEFAULT_RPM = 30
DEFAULT_RPH = 1000
DEFAULT_RP3H = 3000

# Note: we intentionally do NOT cap or pass max_tokens to the ARC endpoint.
# gpt-oss-120b emits reasoning tokens before the user-visible content, and any
# explicit cap risks the reasoning channel consuming the entire budget — leaving
# `message.content` as an empty string. Let the server use its own default.


class RateLimiter:
    """Sliding-window limiter for multiple (max_count, window_seconds) rules.

    Sequential use only — no thread safety. On acquire() we sleep just long
    enough that every rule permits one more call, then record the call.
    """

    def __init__(self, limits):
        self.limits = list(limits)
        self.longest_window = max(w for _, w in self.limits)
        self.times: deque = deque()
        self.n_total = 0

    def acquire(self) -> None:
        while True:
            now = time.time()
            while self.times and now - self.times[0] > self.longest_window:
                self.times.popleft()
            sleep_for = 0.0
            offender = None
            for max_n, window in self.limits:
                in_win = [t for t in self.times if now - t < window]
                if len(in_win) >= max_n:
                    target = in_win[len(in_win) - max_n]
                    delta = target + window - now + 0.5
                    if delta > sleep_for:
                        sleep_for = delta
                        offender = (max_n, window, len(in_win))
            if sleep_for <= 0:
                self.times.append(time.time())
                self.n_total += 1
                return
            mx, win, cur = offender
            print(
                f"[arc rate-limit] {cur}/{mx} in last {win}s — sleeping "
                f"{sleep_for:.1f}s  (calls so far: {self.n_total})",
                flush=True,
            )
            time.sleep(sleep_for)


class ARCClient:
    """Drop-in replacement for KARE/gpt/src/gpt_utils.py:GPTClient.

    Same `.generate(prompt, max_tokens, temperature) -> str` signature, so
    process_sample_condition_a() and friends accept an instance of this class
    interchangeably with GPTClient. Internally routes to the ARC endpoint with
    rate-limiting + retries.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        rpm: int = DEFAULT_RPM,
        rph: int = DEFAULT_RPH,
        rp3h: int = DEFAULT_RP3H,
        max_retries: int = 5,
        base_url: str = ARC_BASE_URL,
    ):
        self.api_key = api_key or os.environ.get("ARC_LLM_API_KEY") or os.environ.get("API_KEY")
        if not self.api_key:
            raise ValueError("ARC API key not set (ARC_LLM_API_KEY or API_KEY env var).")
        masked = self.api_key[:8] + "..." + self.api_key[-4:] if len(self.api_key) > 12 else "***"
        print(f"[ARCClient] key={masked} model={model} base={base_url}", flush=True)

        self.model = model
        self.base_url = base_url
        self.max_retries = max_retries
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.limiter = RateLimiter([(rpm, 60), (rph, 3600), (rp3h, 10800)])

    @property
    def n_calls(self) -> int:
        return self.limiter.n_total

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,  # accepted for GPTClient parity; IGNORED
        temperature: float = 0.7,
    ) -> str:
        """Single-turn completion. Mirrors GPTClient.generate() signature.

        max_tokens is accepted but intentionally not forwarded to the ARC server —
        see module docstring. Returns the model's text response, or "" on terminal
        failure (matches GPTClient behavior so downstream code's None-checks still
        work). Emits a warning when the server returns empty content with a
        non-stop finish_reason so we can spot reasoning-channel exhaustion.
        """
        messages = [{"role": "user", "content": prompt}]
        last_err = None
        for attempt in range(self.max_retries):
            self.limiter.acquire()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                choice = resp.choices[0]
                content = choice.message.content or ""
                if not content.strip():
                    finish = getattr(choice, "finish_reason", "?")
                    print(
                        f"[ARCClient WARN] empty content (finish_reason={finish}) "
                        f"on attempt {attempt + 1}; prompt_len={len(prompt)} chars",
                        flush=True,
                    )
                return content
            except Exception as e:
                last_err = e
                backoff = min(60, 2 ** attempt)
                print(
                    f"[ARCClient retry {attempt + 1}/{self.max_retries}] "
                    f"{type(e).__name__}: {str(e)[:200]} — sleeping {backoff}s",
                    flush=True,
                )
                time.sleep(backoff)
        print(f"[ARCClient FAILED] after {self.max_retries} retries: {last_err}", flush=True)
        return ""


if __name__ == "__main__":
    # Tiny self-test: requires ARC_LLM_API_KEY env var
    c = ARCClient()
    print("response:", repr(c.generate("Say 'hello' in one word.", max_tokens=16, temperature=0.0)))
    print(f"total calls: {c.n_calls}")
