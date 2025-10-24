#!/usr/bin/env python
"""Utility to launch two vLLM OpenAI-compatible servers on dedicated GPUs.

This script spawns two ``python -m vllm.entrypoints.openai.api_server``
processes, each bound to a different CUDA device.  It keeps running until the
user terminates it (Ctrl+C).  Logs from each server are streamed to stdout so
that you can monitor loading progress.

Example
-------
python scripts/launch_vllm_servers.py \
    --llama-model meta-llama/Meta-Llama-3-8B-Instruct \
    --llama-gpu 3 --llama-port 8000 \
    --pmc-model chaoyi-wu/PMC_LLAMA_7B \
    --pmc-gpu 4 --pmc-port 8001

The API servers expose the OpenAI compatible REST endpoints.  Set the
``OPENAI_API_KEY`` environment variable (any string) when calling them.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from typing import Dict, List


def _assert_python_version() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit(
            "Python 3.10 or newer is required to run vLLM. "
            "Detected {}.{}.{}".format(
                sys.version_info.major,
                sys.version_info.minor,
                sys.version_info.micro,
            )
        )


def build_server_command(model: str, port: int, extra_args: List[str] | None = None) -> List[str]:
    """Build the command list for starting a vLLM OpenAI server."""

    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def launch_server(command: List[str], cuda_visible_devices: str) -> subprocess.Popen:
    """Launch a vLLM server with the provided CUDA device string."""

    env: Dict[str, str] = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # vLLM only needs a placeholder API key when running the OpenAI server.
    env.setdefault("OPENAI_API_KEY", "unused-key")

    process = subprocess.Popen(command, env=env)
    return process


def main() -> None:
    _assert_python_version()
    parser = argparse.ArgumentParser(description="Launch two vLLM servers on dedicated GPUs.")
    parser.add_argument("--llama-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="HF repo or local path for the general-purpose Llama model.")
    parser.add_argument("--llama-gpu", default="3",
                        help="GPU id for the Llama model (passed to CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--llama-port", type=int, default=8000,
                        help="Port for the Llama OpenAI server.")
    parser.add_argument("--pmc-model", default="chaoyi-wu/PMC_LLAMA_7B",
                        help="HF repo or local path for the PMC-LLaMA model.")
    parser.add_argument("--pmc-gpu", default="4",
                        help="GPU id for the PMC-LLaMA model (passed to CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--pmc-port", type=int, default=8001,
                        help="Port for the PMC-LLaMA OpenAI server.")
    parser.add_argument("--pmc-extra-args", nargs=argparse.REMAINDER,
                        help="Optional extra args appended to the PMC server command.")

    args = parser.parse_args()

    llama_command = build_server_command(args.llama_model, args.llama_port)
    pmc_command = build_server_command(args.pmc_model, args.pmc_port, args.pmc_extra_args)

    llama_process = launch_server(llama_command, args.llama_gpu)
    pmc_process = launch_server(pmc_command, args.pmc_gpu)

    print("Started Llama server:", " ".join(llama_command))
    print("Started PMC-LLaMA server:", " ".join(pmc_command))
    print("Press Ctrl+C to stop both servers.")

    def shutdown(*_: int) -> None:
        for proc in (llama_process, pmc_process):
            if proc.poll() is None:
                proc.terminate()
        for proc in (llama_process, pmc_process):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Wait indefinitely while child processes run.
    for proc in (llama_process, pmc_process):
        proc.wait()


if __name__ == "__main__":
    main()
