#!/usr/bin/env python3
"""
Build the paired-flow subset for experiment E3.

CIC-UNSW-NB15 was labelled by matching CICFlowMeter flows against the
UNSW-NB15 ground truth on the 5-tuple (src IP, src port, dst IP, dst port,
protocol). The same join therefore locates the *same physical flow* in both
feature representations. Running it yields two CSVs describing an identical
set of flows with an identical label but different feature vectors, which is
what makes a genuinely paired comparison -- and McNemar's test on it --
valid.

Order matters: the 5-tuple must be used for the join and only then dropped.
The output files carry no identifier columns, so they can be passed straight
to ids_evaluation.py.

Usage:
    python build_paired_flows.py --cic CIC.csv --unsw UNSW.csv \
        --label-cic Label --label-unsw label --outdir paired/

Then:
    python ids_evaluation.py --cic paired/paired_cic.csv \
        --unsw paired/paired_unsw.csv --label-cic Label --label-unsw label \
        --outdir results_E3/ --skip-matched
"""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path

import pandas as pd

# CHECK -- adjust to your actual headers. Left: canonical name used for the
# join. Right: candidate column names, first match wins.
KEYS = {
    "src_ip":   ["srcip", "src ip", "source ip", "src_ip"],
    "src_port": ["sport", "src port", "source port", "src_port"],
    "dst_ip":   ["dstip", "dst ip", "destination ip", "dst_ip"],
    "dst_port": ["dsport", "dst port", "destination port", "dst_port"],
    "proto":    ["proto", "protocol"],
}

# UNSW-NB15 stores the protocol as a name, CICFlowMeter as an IANA number.
PROTO_NUM = {"tcp": 6, "udp": 17, "icmp": 1, "igmp": 2, "sctp": 132}


def norm(name: str) -> str:
    return str(name).strip().lower()


def find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {norm(c): c for c in df.columns}
    for cand in candidates:
        if norm(cand) in lookup:
            return lookup[norm(cand)]
    return None


def canon_ip(value) -> str:
    try:
        return str(ipaddress.ip_address(str(value).strip()))
    except ValueError:
        return str(value).strip().lower()


def canon_port(value) -> int:
    """UNSW-NB15 stores some ports as hex strings (e.g. '0x000b')."""
    text = str(value).strip()
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(float(text))
    except (ValueError, TypeError):
        return -1


def canon_proto(value) -> int:
    text = str(value).strip().lower()
    if text in PROTO_NUM:
        return PROTO_NUM[text]
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return -1


def build_key(df: pd.DataFrame, name: str) -> pd.Series:
    cols = {}
    for key, candidates in KEYS.items():
        col = find(df, candidates)
        if col is None:
            raise SystemExit(
                f"{name}: no column found for '{key}'. Tried {candidates}. "
                f"Edit KEYS at the top of this script. "
                f"Available: {list(df.columns)[:30]}"
            )
        cols[key] = col
    return (
        df[cols["src_ip"]].map(canon_ip) + "|"
        + df[cols["src_port"]].map(canon_port).astype(str) + "|"
        + df[cols["dst_ip"]].map(canon_ip) + "|"
        + df[cols["dst_port"]].map(canon_port).astype(str) + "|"
        + df[cols["proto"]].map(canon_proto).astype(str)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cic", required=True)
    ap.add_argument("--unsw", required=True)
    ap.add_argument("--label-cic", required=True)
    ap.add_argument("--label-unsw", required=True)
    ap.add_argument("--outdir", default="paired")
    ap.add_argument("--drop-ambiguous", action="store_true", default=True,
                    help="drop keys occurring more than once in either file "
                         "(recommended: a duplicated 5-tuple cannot be paired "
                         "unambiguously without timestamp comparison)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("loading ...")
    cic = pd.read_csv(args.cic, low_memory=False)
    unsw = pd.read_csv(args.unsw, low_memory=False)
    cic.columns = [str(c).strip() for c in cic.columns]
    unsw.columns = [str(c).strip() for c in unsw.columns]

    cic["_key"] = build_key(cic, "CIC-UNSW-NB15")
    unsw["_key"] = build_key(unsw, "UNSW-NB15")

    stats = {"cic_rows": len(cic), "unsw_rows": len(unsw)}

    if args.drop_ambiguous:
        for df, tag in ((cic, "cic"), (unsw, "unsw")):
            dup = df["_key"].duplicated(keep=False)
            stats[f"{tag}_ambiguous_keys_dropped"] = int(dup.sum())
        cic = cic[~cic["_key"].duplicated(keep=False)]
        unsw = unsw[~unsw["_key"].duplicated(keep=False)]

    shared = set(cic["_key"]) & set(unsw["_key"])
    stats["paired_flows"] = len(shared)
    stats["cic_match_rate_pct"] = round(100 * len(shared) / max(stats["cic_rows"], 1), 2)
    stats["unsw_match_rate_pct"] = round(100 * len(shared) / max(stats["unsw_rows"], 1), 2)

    if not shared:
        raise SystemExit(
            "No flows matched. The 5-tuple columns are probably named or "
            "formatted differently than assumed -- inspect the '_key' values "
            "of each file and adjust KEYS / the canon_* functions."
        )

    cic_p = cic[cic["_key"].isin(shared)].sort_values("_key").reset_index(drop=True)
    unsw_p = unsw[unsw["_key"].isin(shared)].sort_values("_key").reset_index(drop=True)

    # Label agreement is a sanity check on the join itself, not on the data.
    lc = cic_p[args.label_cic].astype(str).str.lower().isin(
        {"benign", "normal", "0", "no", "false"})
    lu = pd.to_numeric(unsw_p[args.label_unsw], errors="coerce").fillna(0) == 0
    if unsw_p[args.label_unsw].dtype == object or str(
            unsw_p[args.label_unsw].dtype) == "str":
        lu = unsw_p[args.label_unsw].astype(str).str.lower().isin(
            {"benign", "normal", "0", "no", "false"})
    agree = float((lc.to_numpy() == lu.to_numpy()).mean())
    stats["label_agreement_pct"] = round(100 * agree, 2)

    cic_p.drop(columns=["_key"]).to_csv(outdir / "paired_cic.csv", index=False)
    unsw_p.drop(columns=["_key"]).to_csv(outdir / "paired_unsw.csv", index=False)
    pd.Series(stats).to_csv(outdir / "pairing_report.csv", header=["value"])

    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nwritten -> {outdir.resolve()}")
    if agree < 0.95:
        print("\nWARNING: label agreement below 95%. Either the join is matching "
              "the wrong flows, or the two files disagree on labelling. "
              "Investigate before reporting E3 -- report the agreement rate in "
              "the paper either way.")


if __name__ == "__main__":
    main()
