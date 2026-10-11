"""Write docs/MODELS.md: what each model is, what it was trained on, how it was trained, and how it was evaluated.

Every figure is read or computed from files in the working tree: the weights and their configs, the coordinator's
state and logs, the data shards and their metadata, and the result files. A figure that cannot be read is written as
"not recorded". An estimate is labelled with its method. Nothing is filled in by hand.

    .venv/bin/python -m training.model_card

Run it from the repository root. It reads only the files named below. It never reads kaggle/secrets, run.env, or the
notebooks, because those hold keys and the coordinator token.
"""

from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import math
import re
import statistics
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "docs" / "MODELS.md"
LM_DIR = ROOT / "models" / "lm-global"
MATCHER_DIRS = [ROOT / "models" / "matcher-demo", ROOT / "models" / "matcher-v2"]
RESULTS = ROOT / "reports" / "2026-10-11" / "results"
MATCH_DATA = ROOT / "data" / "match"
UTC_OFFSET = dt.timedelta(hours=5, minutes=30)  # IST


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_events(path: Path) -> list[dict]:
    events = []
    if path.exists():
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue
    return events


def safetensors_summary(path: Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return {
        "tensors": len(header),
        "parameters": sum(math.prod(t["shape"]) for t in header.values()),
        "dtypes": dict(Counter(t["dtype"] for t in header.values())),
    }


def mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB" if n < 1e9 else f"{n / 1e9:,.2f} GB"


def num(x) -> str:
    return f"{x:,}" if isinstance(x, int) else "n/a"


def rec(x) -> str:
    return "not recorded" if x is None else str(x)


def lfs_status(path: str) -> str:
    """'LFS' when git-lfs stores the file, 'regular git' when git stores it directly, else what git reports."""
    listing = subprocess.run(["git", "lfs", "ls-files", "-n"], capture_output=True, text=True, cwd=ROOT)
    if listing.returncode != 0:
        return "not recorded"
    if path in set(listing.stdout.splitlines()):
        return "LFS"
    listed = subprocess.run(["git", "ls-files", "--error-unmatch", path], capture_output=True, text=True, cwd=ROOT)
    return "regular git" if listed.returncode == 0 else "not in git"


def active_aliases() -> list[str]:
    names = []
    for line in (ROOT / "kaggle" / "active2.txt").read_text(encoding="utf-8").splitlines():
        parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
        if parts and not parts[0].startswith("#"):
            names.append(parts[0])
    return names


def shard(alias: str) -> dict:
    d = ROOT / "data" / "accounts" / alias
    meta = load_json(d / "train_meta.json") or {}
    run = load_json(d / "run.json") or {}
    return {
        "alias": alias,
        "run_alias": run.get("alias"),
        "examples": meta.get("examples"),
        "tokens": meta.get("tokens"),
        "dropped": meta.get("dropped_too_long"),
        "max_len": meta.get("max_len"),
        "pw_rows": run.get("pw_rows"),
        "pw_share": run.get("pw_token_share"),
        "heldout_examples": run.get("heldout_examples"),
        "heldout_tokens": run.get("heldout_tokens"),
    }


def outer_rounds(paths: list[str]) -> dict[int, tuple[int, int | None, str]]:
    """Round number -> (deltas merged, stale deltas, log file). Read from the coordinator's own log lines."""
    rounds = {}
    for name in paths:
        for line in Path(name).read_text(encoding="utf-8", errors="replace").splitlines():
            if '"ev":"outer"' not in line.replace(" ", ""):
                continue
            v = re.search(r'"version"\s*:\s*(\d+)', line)
            d = re.search(r'"deltas"\s*:\s*(\d+)', line)
            s = re.search(r'"stale"\s*:\s*(\d+)', line)
            if v and d:
                rounds[int(v.group(1))] = (int(d.group(1)), int(s.group(1)) if s else None, Path(name).name)
    return dict(sorted(rounds.items()))


def worker_constants() -> dict:
    """Optimizer settings as the worker code states them (training/group_worker.py, training/coordinator.rs)."""
    out = {}
    text = (ROOT / "training" / "group_worker.py").read_text(encoding="utf-8")
    m = re.search(r"AdamW\(params, lr=args\.lr, betas=\(([\d.]+), ([\d.]+)\), weight_decay=([\d.]+)", text)
    if m:
        out.update(adam_beta1=float(m.group(1)), adam_beta2=float(m.group(2)), weight_decay=float(m.group(3)))
    m = re.search(r"clip_grad_norm_\(params, ([\d.]+)\)", text)
    if m:
        out["grad_clip"] = float(m.group(1))
    m = re.search(r'"--lr", type=float, default=([\de.\-]+)', text)
    if m:
        out["lr_default"] = float(m.group(1))
    m = re.search(r'"--warmup-steps", type=int, default=(\d+)', text)
    if m:
        out["warmup_steps"] = int(m.group(1))
    rust = (ROOT / "crates" / "pw-train" / "src" / "coordinator.rs").read_text(encoding="utf-8")
    m = re.search(r"OuterOpt \{ lr: ([\d.]+), momentum: ([\d.]+) \}", rust)
    if m:
        out.update(outer_lr=float(m.group(1)), outer_momentum=float(m.group(2)))
    return out


def tracker_result(seed: int, kind: str) -> dict | None:
    return load_json(RESULTS / f"{kind}-{seed}.json")


def objective(r: dict) -> int:
    p, x = r["present"], r["deleted"]
    return p["correct"] - 5 * (p["wrong_match"] + x["wrong_match"])


def rate(r: dict) -> float:
    return r["present"]["correct"] / max(1, r["present"]["cases"])


def adoption(heur: dict, comb: dict) -> tuple[bool, list[str]]:
    """The adoption rule of docs/DESIGN.md, section 6.10, as training/adopt did it (same thresholds)."""
    reasons, ok = [], True
    for s in (404, 505):
        a = objective(comb[s]) >= objective(heur[s])
        b = comb[s]["present"]["wrong_match"] <= heur[s]["present"]["wrong_match"]
        c = comb[s]["deleted"]["wrong_match"] <= heur[s]["deleted"]["wrong_match"]
        reasons.append(
            f"seed {s} (mixed): objective {objective(comb[s])} vs heuristic {objective(heur[s])}; "
            f"wrong present {comb[s]['present']['wrong_match']} vs {heur[s]['present']['wrong_match']}; "
            f"wrong deleted {comb[s]['deleted']['wrong_match']} vs {heur[s]['deleted']['wrong_match']}"
        )
        ok = ok and a and b and c
    gain = rate(comb[606]) - rate(heur[606])
    reasons.append(
        f"seed 606 (compound): correct {100 * rate(comb[606]):.1f}% vs {100 * rate(heur[606]):.1f}% "
        f"(gain {100 * gain:.1f} points, needs at least 10)"
    )
    return ok and gain >= 0.10, reasons


def git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=ROOT)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def lm_section(lines: list[str], summary: list[str]) -> None:
    cfg = load_json(LM_DIR / "config.json") or {}
    tok = load_json(LM_DIR / "tokenizer.json") or {}
    weights = LM_DIR / "model.safetensors"
    st = safetensors_summary(weights) if weights.exists() else None
    status = load_json(ROOT / "runs" / "state" / "status.json") or {}
    serve = load_json(ROOT / "runs" / "serve" / "run.json") or {}
    events = read_events(ROOT / "runs" / "state" / "events.jsonl")
    aliases = active_aliases()
    shards = [shard(a) for a in aliases]
    have = [s for s in shards if s["examples"] and s["tokens"]]
    total_examples = sum(s["examples"] for s in have)
    total_tokens = sum(s["tokens"] for s in have)
    pooled_len = total_tokens / total_examples if total_examples else None
    token_hashes = {
        s["alias"]: sha256(ROOT / "data" / "accounts" / s["alias"] / "train_tokens.u16")
        for s in shards
        if (ROOT / "data" / "accounts" / s["alias"] / "train_tokens.u16").exists()
    }
    dupes = sorted({a for a, h in token_hashes.items() if list(token_hashes.values()).count(h) > 1})

    starts = [e for e in events if e.get("ev") == "start"]
    start = starts[0] if starts else {}
    world, micro, accum = start.get("world"), start.get("micro_batch"), start.get("grad_accum")
    inner = start.get("inner_steps")
    seqs_per_step = world * micro * accum if None not in (world, micro, accum) else None

    steps = [e for e in events if e.get("ev") == "step"]
    last_step: dict[str, dict] = {}
    for e in steps:
        g = e.get("group")
        if g and (g not in last_step or e["step"] > last_step[g]["step"]):
            last_step[g] = e
    tok_s = [e["tok_s"] for e in steps if isinstance(e.get("tok_s"), (int, float))]
    lrs = [e["lr"] for e in steps if isinstance(e.get("lr"), (int, float))]
    times = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    sends = [e for e in events if e.get("ev") == "send"]
    send_steps = Counter(e.get("steps") for e in sends)

    rounds = outer_rounds(sorted(glob.glob(str(ROOT / "runs" / "coordinator*.log"))))
    deltas_total = sum(d for d, _, _ in rounds.values())
    stale_total = sum(s or 0 for _, s, _ in rounds.values())
    consts = worker_constants()

    evals: dict[int, list[float]] = defaultdict(list)
    for e in events:
        if e.get("ev") == "eval" and isinstance(e.get("heldout_loss"), (int, float)):
            evals[e["version"]].append(e["heldout_loss"])

    send_steps_list = [e["steps"] for e in sends if isinstance(e.get("steps"), (int, float))]
    mean_steps = statistics.mean(send_steps_list) if send_steps_list else None
    merged_seqs = deltas_total * mean_steps * seqs_per_step if mean_steps and seqs_per_step else None
    merged_tokens = merged_seqs * pooled_len if merged_seqs and pooled_len else None
    logged_tokens = None
    if seqs_per_step and have:
        per_group = {s["alias"]: s["tokens"] / s["examples"] for s in have}
        logged_tokens = sum(last_step[g]["step"] * seqs_per_step * per_group[g] for g in last_step if g in per_group)

    lm_eval = load_json(RESULTS / "lm-round16-eval.json")
    pw_v2_sites = (load_json(ROOT / "data" / "pw-v2" / "meta.json") or {}).get("config", {}).get("sites")
    pw_v1_sites = (load_json(ROOT / "data" / "patchwork" / "meta.json") or {}).get("config", {}).get("sites")
    tok_type = (tok.get("model") or {}).get("type")
    vocab_size = len((tok.get("model") or {}).get("vocab") or {})
    merges = len((tok.get("model") or {}).get("merges") or [])
    lm_hash = sha256(weights) if weights.exists() else None
    summary.append(
        f"| PatchWork LM (`models/lm-global`) | author and relocation model; the coordinator's global model | "
        f"{st['parameters']:,} | {mb(weights.stat().st_size) if weights.exists() else 'not recorded'} | "
        f"`{lm_hash[:12] if lm_hash else 'n/a'}` | experimental | {lfs_status('models/lm-global/model.safetensors')} |"
    )

    lines += [
        "## PatchWork LM (`models/lm-global`)",
        "",
        "### Identity",
        "",
        f"- Global model: round **{status.get('round', 'not recorded')}** of the coordinator "
        f"(`runs/state/status.json`), with {status.get('workers', 'not recorded')} workers and a quorum of "
        f"{status.get('quorum', 'not recorded')} deltas per round.",
        f"- Weights: `model.safetensors`, {st['tensors'] if st else 'n/a'} tensors, "
        f"{st['parameters']:,} parameters, stored as {', '.join(f'{k} ({v})' for k, v in (st or {}).get('dtypes', {}).items()) or 'not recorded'}, "
        f"{mb(weights.stat().st_size) if weights.exists() else 'not recorded'}.",
        f"- SHA-256 of `model.safetensors`: `{lm_hash or 'not recorded'}`.",
        f"- Initial weights were seeded with {serve.get('init_seed', 'not recorded')} (`runs/serve/run.json`). "
        f"The parameter count recorded there matches the weights: "
        f"{'yes' if serve.get('params') == st['parameters'] else 'no'}.",
        "- No licence is assigned to the weights by this repository. The training sources carry their own licences "
        "(see the shard `LICENSES.md` files and `data/corpus/general/attribution.json` below).",
        "",
        "### Architecture",
        "",
        f"- Family: `{cfg.get('model_type', 'not recorded')}` (Gemma-3 text decoder), tied input and output embeddings: "
        f"{cfg.get('tie_word_embeddings', 'not recorded')}.",
        f"- Layers: {cfg.get('num_hidden_layers')}, hidden size {cfg.get('hidden_size')}, feed-forward size "
        f"{cfg.get('intermediate_size')} ({cfg.get('hidden_activation')}).",
        f"- Attention: {cfg.get('num_attention_heads')} query heads, {cfg.get('num_key_value_heads')} key/value head, "
        f"head dimension {cfg.get('head_dim')}; sliding window {cfg.get('sliding_window')} tokens on "
        f"{sum(1 for t in cfg.get('layer_types', []) if t == 'sliding_attention')} layers, full attention on "
        f"{sum(1 for t in cfg.get('layer_types', []) if t == 'full_attention')} layers.",
        f"- Context: {cfg.get('max_position_embeddings')} tokens. Vocabulary: {cfg.get('vocab_size')}. "
        f"RMS norm epsilon {cfg.get('rms_norm_eps')}.",
        f"- Tokenizer: byte-level {tok_type or 'not recorded'}, {vocab_size} entries, {merges} merges. "
        f"Special ids: bos {cfg.get('bos_token_id')}, eos {cfg.get('eos_token_id')}, pad {cfg.get('pad_token_id')}.",
        "",
        "### Training data",
        "",
        f"The LM was trained from random weights on {len(aliases)} shards, one per group. Each shard is the group's "
        "slice of the general code corpus, plus PatchWork rows. `training/shards.py build` used `--pw-share 0.2`.",
        "",
        "| Group | Train examples | Train tokens | Dropped (too long) | PatchWork rows | PatchWork token share | Held-out examples |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in shards:
        note = f" (run.json says `{s['run_alias']}`)" if s["run_alias"] and s["run_alias"] != s["alias"] else ""
        lines.append(
            f"| `{s['alias']}`{note} | {num(s['examples'])} | {num(s['tokens'])} | {s['dropped'] if s['dropped'] is not None else 'n/a'} | "
            f"{s['pw_rows'] if s['pw_rows'] is not None else 'n/a'} | {s['pw_share'] if s['pw_share'] is not None else 'n/a'} | "
            f"{s['heldout_examples'] if s['heldout_examples'] is not None else 'n/a'} |"
        )
    lines += [
        "",
        f"- Totals over the {len(have)} shards with metadata: **{total_examples:,} training examples and "
        f"{total_tokens:,} training tokens**, mean {pooled_len:.1f} tokens per example (prompt, completion, bos and eos "
        "included). Max length per example: "
        f"{have[0]['max_len'] if have else 'not recorded'} tokens; longer examples are dropped and counted above.",
        f"- Distinct training files: {len(set(token_hashes.values()))} different SHA-256 values among "
        f"{len(token_hashes)} token files" + (f"; duplicated across groups: {', '.join(dupes)}" if dupes else "") + ".",
        "- Each shard's held-out split is the same for every group, so held-out losses are comparable across groups.",
        "",
    ]
    general = load_json(ROOT / "data" / "corpus" / "general" / "build.json") or {}
    attribution = load_json(ROOT / "data" / "corpus" / "general" / "attribution.json") or []
    lic = Counter(str(a.get("licence")) for a in attribution if isinstance(a, dict))
    manifest = [json.loads(x) for x in (ROOT / "data" / "corpus" / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] if (ROOT / "data" / "corpus" / "manifest.jsonl").exists() else []
    allowed = sum(1 for m in manifest if m.get("allowed") is True)
    reasons = Counter(str(m.get("reason")) for m in manifest if m.get("allowed") is not True)
    lines += [
        "### General corpus and licences",
        "",
        f"- Source: {len(attribution)} packages from crates.io, npm and PyPI (`data/corpus/general`), "
        f"{num(general.get('rows_train'))} training rows and {num(general.get('rows_heldout'))} held-out rows, "
        f"built into {general.get('shards', 'not recorded')} shards.",
        f"- Candidates: {len(manifest)} source records in `data/corpus/manifest.jsonl`; {allowed} passed the licence "
        f"filter; {len(manifest) - allowed} were excluded. Main exclusion reasons: "
        + ", ".join(f"{k} ({v})" for k, v in reasons.most_common(4)) + ".",
        "- Licences of the packages used (count of packages): "
        + ", ".join(f"{k} ({v})" for k, v in lic.most_common(10)) + ".",
        "",
        "### PatchWork rows",
        "",
        "- Synthetic patch-authoring examples (`task`, `split`, `site`, `kind`, `candidates`, `prompt`, `completion`; "
        "loss on the completion only). The shard metadata does not record which PatchWork folder was used. "
        f"`data/pw-v2/meta.json` describes {pw_v2_sites} sites and `data/patchwork/meta.json` describes {pw_v1_sites} sites.",
        "",
        "### Training run",
        "",
        f"- Method: DiLoCo-style. Each group trains locally, then sends its parameter delta. The coordinator averages a "
        f"quorum of deltas and applies an outer step (learning rate {consts.get('outer_lr', 'not recorded')}, momentum "
        f"{consts.get('outer_momentum', 'not recorded')}, from `crates/pw-train/src/coordinator.rs`). Workers download the global model "
        "quantized to int8 and send deltas in float16.",
        f"- Inner optimizer: AdamW (betas {consts.get('adam_beta1', 'n/a')} and {consts.get('adam_beta2', 'n/a')}, weight decay "
        f"{consts.get('weight_decay', 'n/a')}), gradient clipping at {consts.get('grad_clip', 'n/a')}, learning rate with "
        f"{consts.get('warmup_steps', 'n/a')} warm-up steps and cosine decay (`training/train.py: lr_at`). Mixed precision: "
        f"{start.get('precision', 'not recorded')} autocast on {start.get('device', 'not recorded')}.",
        f"- Per group: {world or 'n/a'} GPU ranks with gradient averaging, micro-batch {micro or 'n/a'} sequences, "
        f"{accum or 'n/a'} gradient-accumulation steps, so **{seqs_per_step or 'n/a'} sequences per optimizer step**. "
        f"Each send covers {inner or 'n/a'} inner steps.",
        f"- Peak learning rate in the logs: {max(lrs):.2e}; last logged: {lrs[-1]:.2e} "
        f"({100 * lrs[-1] / max(lrs):.0f}% of peak)." if lrs else "- Learning rate: not recorded in the logs.",
        f"- Outer rounds merged: {len(rounds)} (versions {min(rounds) if rounds else 'n/a'} to {max(rounds) if rounds else 'n/a'}). "
        f"Deltas merged: **{deltas_total}**, of which {stale_total} were computed from an older version. "
        "Each round's record is in the table below, read from the coordinator logs in `runs/`.",
        "",
        "| Round | Deltas merged | Stale deltas | Log |",
        "|---|---|---|---|",
    ]
    for v, (d, s, name) in rounds.items():
        lines.append(f"| {v} | {d} | {s if s is not None else 'n/a'} | `{name}` |")
    first_t = min(times) if times else None
    last_t = max(times) if times else None
    lines += [
        "",
        f"- Sends logged: {len(sends)}, all of {inner} steps: {'yes' if set(send_steps) <= {inner} else 'no'} "
        f"(counts by steps: {dict(send_steps)}). Start events (worker starts, including restarts): {len(starts)}.",
        f"- Span of the timestamped event log (the coordinator logs carry no timestamps): "
        + (f"{dt.datetime.fromtimestamp(first_t, dt.timezone.utc):%Y-%m-%d %H:%M} to "
           f"{dt.datetime.fromtimestamp(last_t, dt.timezone.utc):%Y-%m-%d %H:%M} UTC "
           f"({(last_t - first_t) / 3600:.1f} hours)." if times else "not recorded."),
        f"- Throughput (median over step events): {statistics.median(tok_s):,.0f} tokens per second per group."
        if tok_s else "- Throughput: not recorded.",
        "",
        "### Tokens",
        "",
        f"- Sequences per optimizer step: {seqs_per_step or 'n/a'}. Mean tokens per sequence (shard average): "
        f"{pooled_len:.1f}." if pooled_len else "- Token counts: not recorded.",
        f"- **Training tokens in the shipped round-{status.get('round', '?')} model:** "
        f"{deltas_total} deltas × {mean_steps:,.0f} steps per delta (mean of the {len(send_steps_list)} logged sends) × "
        f"{seqs_per_step} sequences = {merged_seqs:,.0f} sequences, "
        f"about **{merged_tokens / 1e9:.2f} billion tokens** (estimate: sequences × the shard-average length of "
        f"{pooled_len:.1f} tokens). That is about {merged_seqs / total_examples:.1f} passes over the combined "
        f"{total_examples:,} examples of the {len(have)} shards."
        if merged_seqs and pooled_len and total_examples else "- Tokens in the shipped model: not recorded.",
        f"- Tokens processed by the workers, counted from the last logged step of each group (rough: the logs are "
        f"incomplete, and work after a group's last logged step is not counted): "
        f"about {logged_tokens / 1e9:.2f} billion." if logged_tokens else "- Logged worker tokens: not recorded.",
        "",
        "### Held-out evaluation",
        "",
        "Training-time held-out loss, from the workers' `eval` events (the first 512 held-out examples, loss on the "
        "completion tokens only, in nats per token):",
        "",
        "| Global version | Evaluations | Median loss | Perplexity (exp of median) |",
        "|---|---|---|---|",
    ]
    spike_notes: list[str] = []
    previous = None
    for v in sorted(evals):
        med = statistics.median(evals[v])
        lines.append(f"| {v} | {len(evals[v])} | {med:.4f} | {math.exp(med):.1f} |")
        if previous is not None and med - previous[1] > 1.0:
            d, s, _ = rounds.get(v, (None, None, ""))
            spike_notes.append(
                f"- Version {v} rose from {previous[1]:.2f} (version {previous[0]}) to {med:.2f}: round {v} merged "
                f"{d} deltas, and {s} of them were computed from an older version."
            )
        previous = (v, med)
    lines += [
        "",
        *spike_notes,
        "",
        "Final evaluation of the shipped weights (`training.eval_checkpoint`, run on this checkpoint):",
        "",
    ]
    if lm_eval:
        lines += [
            "| Measure | Value | What it is |",
            "|---|---|---|",
            f"| General held-out loss | {lm_eval['general_heldout_loss']:.4f} nats per token (perplexity {lm_eval['general_heldout_ppl']}) | teacher-forced loss on whole packages never seen in training |",
            f"| PatchWork held-out loss | {lm_eval['patchwork_heldout_loss']:.4f} nats per completion token | the same on held-out synthetic PatchWork sites |",
            f"| Relocation, exact answer | {100 * lm_eval['relocate_exact']:.2f}% of {lm_eval['relocate_n']} held-out sites | greedy answer equals the gold answer |",
            f"| Relocation, element picks | {100 * lm_eval['relocate_pick_exact']:.2f}% exact | sites whose gold answer is an element |",
            f"| Relocation, refusals | {100 * lm_eval['relocate_refusal_exact']:.2f}% exact | sites whose gold answer is no element ({100 * lm_eval['relocate_refusal_share']:.1f}% of sites) |",
            f"| Relocation, wrong picks | {100 * lm_eval['relocate_wrong_pick_rate']:.2f}% | harmful errors: an element picked where the answer was a refusal, or the wrong element |",
            f"| Author, exact operations | {100 * lm_eval['author_exact']:.2f}% of {lm_eval['author_n']} | the patch operations match the gold answer |",
            f"| Author, valid DSL JSON | {100 * lm_eval['author_valid_json']:.2f}% | the answer parses as the patch DSL |",
            "",
            "The PatchWork and relocation figures are on synthetic sites from the same generator as the training data, "
            "held out by seed, not by distribution. They do not show performance on real sites, and the model relocates "
            "nothing on the hand-made redesign in `demo/tracking`.",
        ]
    else:
        lines.append("Not recorded: the evaluation result file is missing.")
    lines += [
        "",
        "### Limitations",
        "",
        "- Experimental. On the redesign demo (`demo/tracking`) it relocates nothing, and it was not measured on the "
        "held-out tracker seeds. DESIGN 6.5 records that it mismatched deleted targets more often than the heuristic.",
        f"- The run was stopped by hand at round {status.get('round', '?')}. "
        + (f"The last logged learning rate was {100 * lrs[-1] / max(lrs):.0f}% of the peak." if lrs else "The learning rate is not recorded."),
        "- Training sources are package sources and PatchWork synthetic rows. The model is not evaluated for other "
        "languages, safety, or general chat.",
        "",
        "### Reproduce",
        "",
        "```",
        "python -m training.shards tokenizer --general data/corpus/general --patchwork data/patchwork \\",
        "    --out runs/tokenizer.json --vocab 32768 --sample-mb 100",
        "python -m training.shards build --general data/corpus/general --patchwork data/patchwork \\",
        "    --tokenizer runs/tokenizer.json --serve runs/serve --out data/accounts --accounts 14 --pw-share 0.2",
        "python -m training.eval_checkpoint --serve runs/serve --weights models/lm-global/model.safetensors \\",
        "    --patchwork data/pw-v2/heldout.jsonl --general data/corpus/general/heldout.jsonl --limit 300",
        "```",
        "",
        "The coordinator and the group workers are described in `docs/KAGGLE.md`.",
        "",
    ]


def matcher_section(lines: list[str], summary: list[str], d: Path) -> None:
    cfg = load_json(d / "config.json") or {}
    meta = load_json(d / "meta.json") or {}
    weights = d / "model.safetensors"
    st = safetensors_summary(weights) if weights.exists() else None
    sel = load_json(d / "selection.json")
    current = sha256(weights) if weights.exists() else None
    valid = (sel or {}).get("matcher_sha256") == current if sel else None
    train_rows = [json.loads(x) for x in (MATCH_DATA / "train.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] if (MATCH_DATA / "train.jsonl").exists() else []
    same_data = len(train_rows) == meta.get("train_rows")
    per_epoch = None
    if same_data and (d / "tokenizer.json").exists() and train_rows:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        encs = tok.encode_batch([(r["a"], r["c"]) for r in train_rows])
        per_epoch = sum(len(e.ids) for e in encs)
    name = d.name
    summary.append(
        f"| {name} | tracker: the matcher in the combined tracker (opt-in `--matcher`) | "
        f"{st['parameters']:,} | {mb(weights.stat().st_size) if weights.exists() else 'not recorded'} | "
        f"`{current[:12] if current else 'n/a'}` | {'selection valid' if valid else ('selection stale' if sel else 'no selection record')} | "
        f"{lfs_status(f'models/{name}/model.safetensors')} |"
    )
    epochs, batch = meta.get("epochs"), meta.get("batch")
    flags = [("--epochs", epochs), ("--lr", meta.get("lr")), ("--batch", batch), ("--dropout", meta.get("dropout")),
             ("--pos-weight", meta.get("pos_weight")), ("--warmup", meta.get("warmup")), ("--max-len", meta.get("max_len")),
             ("--seed", meta.get("seed")), ("--vocab", meta.get("vocab")), ("--hidden", meta.get("hidden")),
             ("--layers", meta.get("layers")), ("--heads", meta.get("heads"))]
    cmd = " ".join(["python -m training.match --data data/match", f"--out models/{name}"]
                   + [f"{k} {v}" for k, v in flags if v is not None])
    lines += [
        f"## Matcher: `models/{name}`",
        "",
        "### Identity",
        "",
        f"- Role: the matcher in the combined tracker. It decides when the heuristic is unsure, and it can veto a pick. "
        "It is opt-in with `--matcher`. The heuristic stays the default (DESIGN 6.10).",
        f"- Weights: `model.safetensors`, {st['parameters']:,} parameters, {mb(weights.stat().st_size) if weights.exists() else 'not recorded'}; "
        f"SHA-256 `{current or 'not recorded'}`.",
        f"- Device and precision in training: {meta.get('device', 'not recorded')}, mixed precision "
        f"{meta.get('amp', 'not recorded')}, {meta.get('seconds', 'not recorded')} seconds.",
        "",
        "### Architecture",
        "",
        f"- BERT-style cross-encoder: {cfg.get('num_hidden_layers')} layers, hidden size {cfg.get('hidden_size')}, "
        f"{cfg.get('num_attention_heads')} heads, feed-forward size {cfg.get('intermediate_size')}, vocabulary "
        f"{cfg.get('vocab_size')}, maximum length {cfg.get('max_position_embeddings')}. A single linear unit on the "
        "`[CLS]` vector gives the score (a logit; sigmoid gives the probability).",
        "- Input: the anchor description and the candidate description as one pair, built by `pw_core::describe`. "
        "The gold marker is stripped (DESIGN 6.7a).",
        "",
        "### Training data and tokens",
        "",
        f"- Synthetic pairs from `pw train match-data`: {num(meta.get('train_rows'))} training rows and "
        f"{num(meta.get('val_rows'))} validation rows. Data folder: `data/match` "
        + ("(row counts agree with the metadata)." if same_data else "(row counts do NOT agree with the metadata; token counts are not computed)."),
    ]
    if per_epoch is not None:
        lines.append(
            f"- Tokens per epoch (pairs as the tokenizer encodes them, truncated at {meta.get('max_len')}): {per_epoch:,}; "
            f"over {epochs} epochs: **{per_epoch * epochs:,} tokens**."
        )
    lines += [
        "",
        "### Training",
        "",
        f"- Command that reproduces it (from the metadata): `{cmd}`",
        f"- Optimizer: AdamW, learning rate {rec(meta.get('lr'))}, batch {rec(batch)}, {rec(epochs)} epochs, warm-up share "
        f"{rec(meta.get('warmup'))}, dropout {rec(meta.get('dropout'))}, positive weight {rec(meta.get('pos_weight'))}, "
        f"seed {rec(meta.get('seed'))}.",
        "",
        "| Epoch | Train loss | Validation top-1 (target present) | Validation refusal (target absent) |",
        "|---|---|---|---|",
    ]
    for h in meta.get("history", []):
        lines.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['top1_present']:.4f} | {h['refused_absent']:.4f} |")
    best = max(meta.get("history", [{"top1_present": None, "epoch": None}]), key=lambda h: h["top1_present"] or 0)
    lines += [
        "",
        f"- Best validation top-1: epoch {best['epoch']} ({best['top1_present']}). The validation split comes from the "
        "same generator as the training split, so these numbers are not held-out seeds.",
        "",
        "### Tracker selection",
        "",
    ]
    if sel:
        s = sel["setting"]
        lines += [
            f"- Setting from `{name}/selection.json` (method `{sel['method']}`, written by `{sel['written_by']}`): "
            f"accept {s['accept']}, margin {s['margin']}, review {s['review']}, veto {s['veto']}.",
            f"- Chosen on the development slices {', '.join(sel['slices'])} ({sel['sites']} sites, {sel['variants']} variants "
            f"per site), out of {sel['settings_searched']} settings, by the summed objective {sel['objective_sum']}.",
            f"- Valid for the current weights: {'yes' if valid else 'NO: the weights changed after the grid ran'}.",
        ]
    else:
        lines.append(
            f"- No selection record. `pw --matcher models/{name}` refuses to run until "
            "`pw train track --grid` writes one."
        )
    lines.append("")


def tracker_section(lines: list[str]) -> None:
    lines += ["## Tracker results on the held-out seeds", ""]
    heur = {s: tracker_result(s, "heur") for s in (404, 505, 606)}
    comb = {s: tracker_result(s, "test") for s in (404, 505, 606)}
    if not all(heur.values()) or not all(comb.values()):
        lines.append("Not recorded: the result files are missing.")
        lines.append("")
        return
    kinds = {404: "mixed", 505: "mixed", 606: "compound"}
    lines += [
        "Objective J = correct present − 5 × (wrong present + wrong deleted). The test seeds are spent (each used once).",
        "",
        "| Seed | Drift | Heuristic J | Heuristic correct | Combined J | Combined correct |",
        "|---|---|---|---|---|---|",
    ]
    for s in (606, 404, 505):
        h, c = heur[s], comb[s]
        lines.append(
            f"| {s} | {kinds[s]} | {objective(h)} | {h['present']['correct']}/{h['present']['cases']} "
            f"({100 * rate(h):.1f}%) | {objective(c)} | {c['present']['correct']}/{c['present']['cases']} ({100 * rate(c):.1f}%) |"
        )
    ok, reasons = adoption(heur, comb)
    lines += [
        "",
        f"Adoption rule (DESIGN 6.10): **{'ADOPTED' if ok else 'NOT ADOPTED'}**. The heuristic stays the default; the combined tracker is opt-in.",
        "",
    ]
    lines += [f"- {r}" for r in reasons]
    lines.append("")


def main() -> None:
    lines: list[str] = []
    summary: list[str] = []
    head = [
        "# PatchWork models",
        "",
        f"Generated by `training/model_card.py` on {dt.datetime.now(dt.timezone(UTC_OFFSET)):%Y-%m-%d %H:%M} IST "
        f"from commit `{git('rev-parse', '--short', 'HEAD') or 'not recorded'}`. Every figure below is read or computed "
        "from the files in the repository; estimates say how they were made. Re-run the script after any change to "
        "the weights, the logs, or the results.",
        "",
        "## Summary",
        "",
        "| Model | Role | Parameters | Size | SHA-256 (first 12) | Status | Git LFS |",
        "|---|---|---|---|---|---|---|",
    ]
    body: list[str] = []
    lm_section(body, summary)
    for d in MATCHER_DIRS:
        if d.exists() and (d / "model.safetensors").exists():
            matcher_section(body, summary, d)
    tracker_section(body)
    head += summary
    lfs = git("lfs", "ls-files", "-n")
    if lfs is None:
        head += ["", "Git LFS listing: not recorded.", ""]
    else:
        tracked = [p for p in lfs.splitlines() if p]
        head += ["", ("Files stored in Git LFS: " + ", ".join(f"`{p}`" for p in tracked)) if tracked
                 else "No file is stored in Git LFS yet.", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
