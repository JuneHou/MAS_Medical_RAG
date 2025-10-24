#!/usr/bin/env python
"""Run a multi-agent debate between two vLLM-hosted models on MedQA.

The script assumes that both models are exposed via the OpenAI-compatible REST
API provided by ``vllm.entrypoints.openai.api_server``.  The debate logic
alternates between a generalist Llama-3.1-8B agent and a domain-specialised
PMC-LLaMA-7B agent.  Each model receives the conversation history and is asked
to refine its answer after every rebuttal.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from datasets import load_dataset
from tqdm import tqdm

Choice = Tuple[str, str]


@dataclass
class AgentConfig:
    name: str
    system_prompt: str
    api_url: str
    api_key: str


def call_chat_completion(agent: AgentConfig, messages: List[Dict[str, str]],
                         temperature: float = 0.2, max_tokens: int = 512) -> str:
    payload = {
        "model": "unused",  # vLLM ignores this when single model served per process
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {agent.api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{agent.api_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def format_question(question: str, choices: Iterable[Choice]) -> str:
    formatted_choices = "\n".join(f"{label}. {text}" for label, text in choices)
    return f"Question: {question}\nOptions:\n{formatted_choices}\nProvide a single best answer with reasoning."


def parse_choices(sample: Dict[str, str]) -> Tuple[str, List[Choice], str]:
    question = sample.get("question") or sample.get("prompt")
    if question is None:
        raise KeyError("Could not find 'question' field in the MedQA sample.")

    if "options" in sample:
        options = sample["options"]
    elif "choices" in sample:
        options = sample["choices"]
    elif "options_text" in sample:
        options = sample["options_text"]
    else:
        raise KeyError("Could not find answer options in the MedQA sample.")

    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    formatted: List[Choice] = []
    for idx, option in enumerate(options):
        label = labels[idx]
        if isinstance(option, dict):
            text = option.get("text") or option.get("answer") or str(option)
        else:
            text = str(option)
        formatted.append((label, text))

    answer = sample.get("answer")
    if isinstance(answer, int):
        answer_label = labels[answer]
    elif isinstance(answer, str) and len(answer.strip()) == 1 and answer.strip().upper() in labels:
        answer_label = answer.strip().upper()
    elif isinstance(answer, str) and answer.strip() in labels[: len(options)]:
        answer_label = answer.strip()
    elif "answer_idx" in sample:
        idx_value = sample["answer_idx"]
        if isinstance(idx_value, int):
            answer_label = labels[idx_value]
        elif isinstance(idx_value, str):
            idx_str = idx_value.strip()
            if idx_str.isdigit():
                answer_label = labels[int(idx_str)]
            elif len(idx_str) == 1 and idx_str.upper() in labels[: len(options)]:
                answer_label = idx_str.upper()
    else:
        # If the dataset stores textual answer, try to map back to a label.
        answer_label = None
        if isinstance(answer, str):
            normalized = answer.lower().strip()
            for label, text in formatted:
                if normalized == text.lower().strip():
                    answer_label = label
                    break
    if answer_label is None:
        answer_label = ""

    return question, formatted, answer_label


ANSWER_PATTERN = re.compile(r"\b([A-H])\b")


def extract_answer_letter(text: str, valid_labels: Iterable[str]) -> str:
    match = ANSWER_PATTERN.search(text)
    if match and match.group(1) in valid_labels:
        return match.group(1)
    return ""


@dataclass
class DebateResult:
    question: str
    options: List[Choice]
    ground_truth: str
    generalist_responses: List[str]
    specialist_responses: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "question": self.question,
            "options": self.options,
            "ground_truth": self.ground_truth,
            "generalist_responses": self.generalist_responses,
            "specialist_responses": self.specialist_responses,
        }


def run_debate_round(question: str, options: List[Choice], rounds: int,
                     generalist: AgentConfig, specialist: AgentConfig) -> DebateResult:
    prompt = format_question(question, options)

    generalist_messages = [
        {"role": "system", "content": generalist.system_prompt},
        {"role": "user", "content": prompt},
    ]
    specialist_messages = [
        {"role": "system", "content": specialist.system_prompt},
        {"role": "user", "content": prompt},
    ]

    generalist_history: List[str] = []
    specialist_history: List[str] = []

    for round_idx in range(rounds):
        generalist_reply = call_chat_completion(generalist, generalist_messages)
        generalist_history.append(generalist_reply)
        generalist_messages.append({"role": "assistant", "content": generalist_reply})

        specialist_messages.append({
            "role": "user",
            "content": (
                f"The generalist agent responded with:\n{generalist_reply}\n"
                "Critique their reasoning and refine your own answer."
            ),
        })
        specialist_reply = call_chat_completion(specialist, specialist_messages)
        specialist_history.append(specialist_reply)
        specialist_messages.append({"role": "assistant", "content": specialist_reply})

        generalist_messages.append({
            "role": "user",
            "content": (
                f"The medical specialist replied with:\n{specialist_reply}\n"
                "Respond with an improved argument if needed."
            ),
        })

    # Ask both agents for a final concise answer after the debate history.
    generalist_messages[-1]["content"] += (
        "\nDebate complete. Provide your final answer letter (A/B/...) and a "
        "one-sentence justification."
    )
    final_generalist = call_chat_completion(generalist, generalist_messages)
    generalist_history.append(final_generalist)

    specialist_messages.append({
        "role": "user",
        "content": (
            "Debate complete. Provide your final answer letter (A/B/...) and a one-sentence "
            "justification."
        ),
    })
    final_specialist = call_chat_completion(specialist, specialist_messages)
    specialist_history.append(final_specialist)

    return DebateResult(
        question=question,
        options=options,
        ground_truth="",
        generalist_responses=generalist_history,
        specialist_responses=specialist_history,
    )


def evaluate_results(results: List[DebateResult]) -> Dict[str, float]:
    total = 0
    generalist_correct = 0
    specialist_correct = 0

    for result in results:
        labels = [label for label, _ in result.options]
        generalist_answer = extract_answer_letter(result.generalist_responses[-1], labels)
        specialist_answer = extract_answer_letter(result.specialist_responses[-1], labels)

        if result.ground_truth:
            total += 1
            if generalist_answer == result.ground_truth:
                generalist_correct += 1
            if specialist_answer == result.ground_truth:
                specialist_correct += 1

    accuracy = {
        "evaluated": total,
        "generalist_accuracy": generalist_correct / total if total else 0.0,
        "specialist_accuracy": specialist_correct / total if total else 0.0,
    }
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a LLM debate on MedQA.")
    parser.add_argument("--llama-url", default="http://localhost:8000",
                        help="Base URL of the Llama vLLM OpenAI server.")
    parser.add_argument("--pmc-url", default="http://localhost:8001",
                        help="Base URL of the PMC-LLaMA vLLM OpenAI server.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "token"),
                        help="API key to send in the Authorization header.")
    parser.add_argument("--medqa-split", default="validation",
                        help="MedQA split to evaluate (train/validation/test).")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of MedQA samples to run.")
    parser.add_argument("--rounds", type=int, default=2,
                        help="Number of debate rounds (each round is generalist followed by specialist).")
    parser.add_argument("--output", type=Path, default=Path("debate_results.json"),
                        help="Path to write the JSONL debate transcripts.")

    args = parser.parse_args()

    generalist = AgentConfig(
        name="llama",
        system_prompt=(
            "You are a thoughtful general reasoning assistant without specialised medical "
            "knowledge. Rely on first principles and logical deduction to answer questions."
        ),
        api_url=args.llama_url,
        api_key=args.api_key,
    )
    specialist = AgentConfig(
        name="pmc-llama",
        system_prompt=(
            "You are a medical domain expert trained on clinical literature. Provide precise "
            "medical reasoning grounded in domain knowledge."
        ),
        api_url=args.pmc_url,
        api_key=args.api_key,
    )

    dataset = load_dataset("medalpaca/medqa", split=args.medqa_split)

    results: List[DebateResult] = []
    for sample in tqdm(dataset.select(range(min(args.num_samples, len(dataset))))):
        question, options, answer_label = parse_choices(sample)
        debate = run_debate_round(question, options, args.rounds, generalist, specialist)
        debate.ground_truth = answer_label
        results.append(debate)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result.to_dict(), ensure_ascii=False)
            f.write(json_line + "\n")

    metrics = evaluate_results(results)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved debate transcripts to", args.output)
    print("Saved accuracy metrics to", metrics_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
