"""Program post-mortem memo.

The narrator receives a display payload built from ProgramResult (numbers
already formatted as strings) and returns prose. It never sees raw tables.

Provenance guard: after each draft, every numeric token in the memo is
checked against the numeric tokens in the payload. Any unmatched number
rejects the draft, the failure is logged, and the memo is regenerated (up to
MAX_ATTEMPTS). If no draft passes, or no API key is configured, a template
memo built from the same payload is returned instead.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .assemble import STATUS_NO_OVERLAP, STATUS_OK, ProgramResult

MODEL = "claude-opus-5-5"
MAX_ATTEMPTS = 3
LOG_PATH = Path(__file__).resolve().parent / "logs" / "provenance_guard_log.jsonl"   # every draft, accepted or not


# ---------------------------------------------------------------------------
# formatting: the only place numbers become text
# ---------------------------------------------------------------------------
def usd(x: float) -> str:
    x = round(x)
    return f"-${abs(x):,}" if x < 0 else f"${x:,}"


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"


def ci_usd(e) -> str:
    return f"{usd(e.ci_low)} to {usd(e.ci_high)}"


def narration_payload(r: ProgramResult) -> dict:
    s, m, g = r.setup, r.matching, r.engagement
    worst = max(m.balance, key=lambda b: abs(b.smd_after)) if m.balance else None
    payload = {
        "program": {"name": s.name, "id": s.program_id, "window": f"{s.window_start} to {s.window_end}",
                    "months_in_window": str(s.months), "eligibility_rule": s.eligibility_rule,
                    "mechanic": s.mechanic, "budget": usd(s.budget), "target_product": s.target_product},
        "cohorts": {"eligible_reps": str(s.eligible_reps), "treated_reps_analysed": str(s.treated_reps),
                    "control_pool": str(s.control_pool),
                    "excluded_paid_by_another_program": str(s.excluded_contaminated),
                    "excluded_too_little_pre_period_history": str(s.excluded_thin_history),
                    "excluded_new_hires_still_ramping": str(s.excluded_ramping)},
        "confidence_level": "95%",
        "design_status": r.status,
        "design_problems": list(r.status_reasons),
        "matching": {"matched_pairs": str(m.matched_pairs), "treated_dropped_no_match": str(m.dropped_treated),
                     "balance_check": "pass" if m.balance_passed else "fail",
                     "worst_feature_after_matching": worst.feature if worst else None,
                     "worst_smd_after_matching": f"{abs(worst.smd_after):.3f}" if worst else None,
                     "smd_threshold": f"{r.assumptions['smd_threshold']:.2f}"},
        "engagement": {"participation_rate": pct(g.participation_rate), "participants": str(g.participants),
                       "payout_concentration_top_share": pct(g.concentration_top, 0),
                       "payout_concentration": pct(g.payout_concentration),
                       "median_days_to_first_payout": f"{g.median_days_to_first_payout:.0f}",
                       "median_payout_per_participant": usd(g.median_payout_per_participant)},
        "spend": {"payout_cost": usd(g.total_payout), "budget_utilisation": pct(g.budget_utilisation)},
    }
    if r.parallel_trends is not None:
        payload["parallel_trends"] = {"check": "pass" if r.parallel_trends.passed else "fail",
                                      "joint_test_p_value": f"{r.parallel_trends.joint_p:.3f}"}
    if r.placebo is not None:
        payload["placebo"] = {"check": ("pass" if r.placebo.passed else "fail") if r.placebo.available
                              else "not available (not enough history before launch)",
                              "placebo_window": f"{r.placebo.window_start} to {r.placebo.window_end}"}
    if r.status == STATUS_OK:
        e, ec = r.estimates, r.economics
        payload["result"] = {
            "att_total_bookings_per_rep_month": usd(e.att_total.beta),
            "att_total_95ci": ci_usd(e.att_total),
            "att_total_significant": "yes" if e.att_total.significant else "no",
            "att_target_product_per_rep_month": usd(e.att_target.beta),
            "att_target_95ci": ci_usd(e.att_target),
            "cannibalisation_rate": pct(e.cannibalisation_rate.beta),
            "cannibalisation_95ci": f"{pct(e.cannibalisation_rate.ci_low)} to {pct(e.cannibalisation_rate.ci_high)}",
            "incremental_bookings": usd(ec.incremental_bookings.point),
            "incremental_bookings_95ci": f"{usd(ec.incremental_bookings.low)} to {usd(ec.incremental_bookings.high)}",
            "gross_margin_assumption": pct(ec.gross_margin, 0),
            "incremental_gross_profit": usd(ec.incremental_gross_profit.point),
            "net_program_return": usd(ec.net_return.point),
            "net_program_return_95ci": f"{usd(ec.net_return.low)} to {usd(ec.net_return.high)}",
            "program_roi": pct(ec.roi.point),
            "program_roi_95ci": f"{pct(ec.roi.low)} to {pct(ec.roi.high)}",
            "breakeven_att_per_rep_month": usd(ec.breakeven_att),
            "breakeven_verdict": ec.breakeven_verdict,
        }
        payload["naive_reads_for_contrast"] = {
            "treated_minus_untreated_window_bookings": usd(r.naive.treated_vs_untreated),
            "treated_before_vs_after": usd(r.naive.before_after)}
    return payload


# ---------------------------------------------------------------------------
# provenance guard
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?%?")


def _norm(tok: str) -> str:
    return tok.replace("$", "").replace(",", "").lstrip("-").rstrip(".")


def numeric_tokens(text: str) -> set:
    return {_norm(t) for t in _NUM.findall(text.replace("−", "-"))}


def unverified_numbers(memo: str, payload: dict) -> list:
    allowed = numeric_tokens(json.dumps(payload, ensure_ascii=False))
    return sorted({t for t in _NUM.findall(memo.replace("−", "-")) if _norm(t) not in allowed})


# ---------------------------------------------------------------------------
# drafting
# ---------------------------------------------------------------------------
SYSTEM = """You write one-page SPIFF program post-mortems for sales leadership and finance.

You are given PAYLOAD, a JSON object produced by a measurement pipeline. Rules:
- Every number you write must be copied exactly, character for character, from PAYLOAD. Do not calculate, \
round, convert units, derive percentages, add, subtract or restate a number in a different form. If a point \
needs a number that is not in PAYLOAD, make it in words.
- Do not use numbered lists or numbered headings.
- If design_status is not "ok", open with the limitation and why the evidence cannot support an effect \
estimate. Do not state or imply any effect size, return or ROI.
- Make only claims the data supports. You may describe what happened (participation, concentration, cost, \
the estimated effect and its uncertainty). Do not recommend cancelling, expanding or renewing the program; \
recommendations are limited to how the next program should be designed and measured.
- The unit of judgement is the program. Never single out or evaluate individual reps.
- Always give a causal estimate together with its 95% confidence interval.

Structure, in Markdown, with these bold headings: **What the program did**, **What it cost**, \
**How much to trust the evidence**, **What to change next time**. Plain, direct prose. Under 400 words."""


def draft_llm(payload: dict) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=4000, thinking={"type": "adaptive"}, output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default", system=SYSTEM,
        messages=[{"role": "user", "content": "PAYLOAD:\n" + json.dumps(payload, indent=2, ensure_ascii=False)}])
    if resp.stop_reason == "refusal":
        raise RuntimeError("model declined the request")
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def draft_template(payload: dict) -> str:
    p, c, eng = payload["program"], payload["cohorts"], payload["engagement"]
    out = [f"### {p['name']} ({p['id']}), {p['window']}", ""]
    if payload["design_status"] != STATUS_OK:
        out += ["**How much to trust the evidence**",
                "The measurement design did not pass its checks, so no effect estimate is reported:", ""]
        out += [f"- {x}" for x in payload["design_problems"]] + [""]
    out += ["**What the program did**",
            f"A {p['mechanic']} SPIFF on {p['target_product']} for reps matching `{p['eligibility_rule']}`. "
            f"{eng['participants']} of {c['eligible_reps']} eligible reps earned a payout "
            f"({eng['participation_rate']} participation). The top {eng['payout_concentration_top_share']} of "
            f"participants earned {eng['payout_concentration']} of all payouts. Payouts totalled "
            f"{payload['spend']['payout_cost']}, {payload['spend']['budget_utilisation']} of the "
            f"{p['budget']} budget.", ""]
    if "result" in payload:
        r = payload["result"]
        out += [f"Against matched controls, total bookings moved by {r['att_total_bookings_per_rep_month']} per "
                f"treated rep-month (95% CI {r['att_total_95ci']}), and {p['target_product']} bookings by "
                f"{r['att_target_product_per_rep_month']} (95% CI {r['att_target_95ci']}). Estimated "
                f"cannibalisation of other products: {r['cannibalisation_rate']} "
                f"(95% CI {r['cannibalisation_95ci']}).", "",
                "**What it cost**",
                f"Payouts were {payload['spend']['payout_cost']} ({payload['spend']['budget_utilisation']} of the "
                f"{p['budget']} budget). At an "
                f"assumed {r['gross_margin_assumption']} gross margin, net program return is "
                f"{r['net_program_return']} (95% CI {r['net_program_return_95ci']}); ROI {r['program_roi']} "
                f"(95% CI {r['program_roi_95ci']}). Breakeven required {r['breakeven_att_per_rep_month']} per "
                f"treated rep-month: {r['breakeven_verdict']}.", "",
                "**How much to trust the evidence**",
                f"Matched {payload['matching']['matched_pairs']} pairs; balance, parallel trends and placebo "
                "checks passed. The naive comparison of participants against non-participants "
                f"({payload['naive_reads_for_contrast']['treated_minus_untreated_window_bookings']}) is mostly "
                "selection, not program effect.", ""]
    out += ["**What to change next time**",
            "Define the measurement design before launch: hold out an eligible control group, fix the "
            "evaluation window, and set the breakeven effect size in advance."]
    return "\n".join(out)


@dataclass
class Memo:
    markdown: str
    source: str                       # model id or "template"
    attempts: int
    rejected: list = field(default_factory=list)   # unmatched tokens per rejected draft
    error: str | None = None


def _log(entry: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def narrate(result: ProgramResult, draft_fn: Callable[[dict], str] | None = draft_llm,
            max_attempts: int = MAX_ATTEMPTS, log_path: Path | None = LOG_PATH) -> Memo:
    payload = narration_payload(result)
    rejected, err = [], None
    if draft_fn is not None:
        for attempt in range(1, max_attempts + 1):
            try:
                text = draft_fn(payload)
            except Exception as e:          # no key, network, refusal: fall back, never block the report
                err = f"{type(e).__name__}: {e}"
                break
            bad = unverified_numbers(text, payload)
            if log_path is not None:
                _log({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL,
                      "program_id": result.setup.program_id, "design_status": result.status,
                      "gross_margin": result.assumptions["gross_margin"], "attempt": attempt,
                      "outcome": "rejected" if bad else "accepted",
                      "numbers_checked": len(_NUM.findall(text)), "unmatched_numbers": bad,
                      "context": [_context(text, t) for t in bad[:5]]}, log_path)
            if not bad:
                return Memo(text, MODEL, attempt, rejected)
            rejected.append(bad)
    return Memo(draft_template(payload), "template", len(rejected), rejected, err)


def _context(text: str, tok: str, width: int = 60) -> str:
    i = text.find(tok)
    return text[max(0, i - width): i + len(tok) + width].replace("\n", " ") if i >= 0 else tok


__all__ = ["narrate", "narration_payload", "unverified_numbers", "Memo", "STATUS_NO_OVERLAP"]
