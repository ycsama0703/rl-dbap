import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, Optional

import requests


DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def _read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt and prompt_file:
        raise ValueError("Specify either --prompt or --prompt-file, not both")
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    if prompt:
        return prompt
    # fallback: read stdin if piped
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("No prompt provided. Use --prompt, --prompt-file, or pipe stdin.")


def _post_non_stream(
    api_base: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _post_stream(
    api_base: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: float,
) -> Iterable[str]:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    with requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    break
                yield data


def main() -> int:
    ap = argparse.ArgumentParser(description="Call DeepSeek chat completions API (OpenAI-compatible)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model name, e.g. deepseek-chat or deepseek-reasoner")
    ap.add_argument("--api-base", default=DEFAULT_API_BASE, help="API base URL (default: https://api.deepseek.com)")
    ap.add_argument("--api-key", default=os.getenv("DEEPSEEK_API_KEY"), help="API key or set env DEEPSEEK_API_KEY")

    prompt_group = ap.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", help="User prompt text")
    prompt_group.add_argument("--prompt-file", help="Path to a file containing the prompt")

    ap.add_argument("--system", default=None, help="Optional system instruction")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    ap.add_argument("--stream", action="store_true", help="Stream tokens to stdout")
    ap.add_argument("--json-output", help="If set, write full JSON response(s) to this file")
    ap.add_argument("--print-reasoning", action="store_true", help="If present and available, print reasoning content")
    ap.add_argument(
        "--metadata",
        nargs="*",
        help="Optional key=val pairs to attach to request metadata",
    )

    args = ap.parse_args()

    if not args.api_key:
        print("[error] Missing API key. Set --api-key or env DEEPSEEK_API_KEY.", file=sys.stderr)
        return 2

    try:
        user_prompt = _read_prompt(args.prompt, args.prompt_file)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": user_prompt})

    md: Dict[str, Any] = {}
    if args.metadata:
        for kv in args.metadata:
            if "=" in kv:
                k, v = kv.split("=", 1)
                md[k] = v

    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": bool(args.stream),
    }
    if md:
        payload["metadata"] = md

    json_out_file = None
    if args.json_output:
        json_out_file = open(args.json_output, "a", encoding="utf-8")

    try:
        if args.stream:
            # Stream and print incrementally
            acc_content = []
            acc_reason = []
            start = time.time()
            for data_line in _post_stream(args.api_base, args.api_key, payload, args.timeout):
                try:
                    obj = json.loads(data_line)
                except Exception:
                    continue
                if json_out_file:
                    json_out_file.write(json.dumps(obj, ensure_ascii=False) + "\n")

                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                # Some models expose reasoning deltas separately
                if args.print_reasoning:
                    rc = delta.get("reasoning_content")
                    if rc:
                        acc_reason.append(rc)
                        # Print reasoning to stderr to separate from final answer text
                        print(rc, end="", file=sys.stderr)
                cc = delta.get("content")
                if cc:
                    acc_content.append(cc)
                    print(cc, end="", flush=True)
            print()
            dur = time.time() - start
            # Summary line to stderr
            print(f"\n[done] streamed {len(''.join(acc_content))} chars in {dur:.2f}s", file=sys.stderr)
        else:
            resp = _post_non_stream(args.api_base, args.api_key, payload, args.timeout)
            if json_out_file:
                json_out_file.write(json.dumps(resp, ensure_ascii=False) + "\n")
            choices = resp.get("choices") or []
            if not choices:
                print(json.dumps(resp, ensure_ascii=False, indent=2))
                return 0
            msg = (choices[0].get("message") or {})
            # Print reasoning if available
            if args.print_reasoning:
                rc = msg.get("reasoning_content")
                if rc:
                    print(rc, file=sys.stderr)
            content = msg.get("content") or ""
            print(content)
    except requests.HTTPError as e:
        try:
            err_json = e.response.json()
        except Exception:
            err_json = {"error": str(e)}
        print(json.dumps(err_json, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    finally:
        if json_out_file:
            json_out_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

