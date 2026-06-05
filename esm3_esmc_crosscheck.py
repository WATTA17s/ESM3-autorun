# =========================
# ESM3 -> ESMC Cross-check Module
# Use with ESM3 Conservative Consensus-Guided Mutagenesis Pipeline V3
# =========================

from pathlib import Path
from getpass import getpass
from typing import Dict, List, Optional
import csv
import math
import os

import numpy as np
import torch

from esm.sdk import esmc_client
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers


VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA_SET = set(VALID_AA)
MASK_CHAR = "_"


# =========================
# Basic helpers
# =========================

def clean_sequence(seq: str) -> str:
    return "".join(str(seq).split()).upper()


def validate_sequence(seq: str, name: str = "sequence") -> None:
    if not seq:
        raise ValueError(f"{name} is empty.")

    bad = sorted(set(seq) - VALID_AA_SET)
    if bad:
        raise ValueError(f"{name} has invalid amino acid(s): {bad}")


def read_text_sequence(path: str) -> str:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return clean_sequence(text)


def read_final_vote_report(path: str) -> List[Dict[str, str]]:
    path = Path(path)

    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def find_accepted_mutations(ref_seq: str, final_seq: str) -> List[Dict[str, object]]:
    """
    Return accepted ESM3 mutations by comparing WT/reference and final sequence.
    """
    ref_seq = clean_sequence(ref_seq)
    final_seq = clean_sequence(final_seq)

    validate_sequence(ref_seq, "ref_seq")
    validate_sequence(final_seq, "final_seq")

    if len(ref_seq) != len(final_seq):
        raise ValueError(
            f"Length mismatch: ref_seq={len(ref_seq)}, final_seq={len(final_seq)}"
        )

    mutations = []

    for i, (wt, mut) in enumerate(zip(ref_seq, final_seq)):
        if wt != mut:
            mutations.append({
                "position_1based": i + 1,
                "position_0based": i,
                "WT": wt,
                "MUT": mut,
                "mutation": f"{wt}{i + 1}{mut}"
            })

    return mutations


def build_final_report_lookup(final_report_rows: Optional[List[Dict[str, object]]]):
    lookup = {}

    if not final_report_rows:
        return lookup

    for row in final_report_rows:
        try:
            pos = int(row["position_1based"])
            lookup[pos] = row
        except Exception:
            continue

    return lookup


# =========================
# ESMC helpers
# =========================

def get_token(token=None):
    if token:
        return token

    env_token = os.getenv("BIOHUB_TOKEN")
    if env_token:
        print("Using BIOHUB_TOKEN from environment.")
        return env_token

    return getpass("Biohub / ESMC API token: ")


def build_esmc_client(model_name: str, token: str, url: str = "https://biohub.ai"):
    return esmc_client(
        model=model_name,
        url=url,
        token=token
    )


def get_sequence_vocab() -> Dict[str, int]:
    tokenizers = get_esmc_model_tokenizers()

    candidates = [
        tokenizers,
        getattr(tokenizers, "sequence", None),
        getattr(tokenizers, "sequence_tokenizer", None),
    ]

    for tok in candidates:
        if tok is None:
            continue

        if hasattr(tok, "get_vocab"):
            vocab = tok.get_vocab()

            if all(aa in vocab for aa in VALID_AA):
                return vocab

    raise RuntimeError("Could not find ESMC sequence vocabulary.")


def mask_one_position(seq: str, pos_0based: int) -> str:
    return seq[:pos_0based] + MASK_CHAR + seq[pos_0based + 1:]


def run_esmc_logits(client, sequence: str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    if isinstance(protein_tensor, ESMProteinError):
        raise RuntimeError(protein_tensor)

    return client.logits(
        protein_tensor,
        LogitsConfig(sequence=True)
    )


def extract_sequence_logits(logits_output):
    logits = logits_output.logits

    if hasattr(logits, "sequence"):
        x = logits.sequence
    else:
        x = logits

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    if x.ndim == 3 and x.shape[0] == 1:
        x = x.squeeze(0)

    return x.float()


def infer_position_offset(logits_tensor, seq_len: int) -> int:
    """
    Some ESM outputs include BOS/EOS tokens.
    """
    L = logits_tensor.shape[0]

    if L == seq_len:
        return 0

    if L == seq_len + 1:
        return 1

    if L == seq_len + 2:
        return 1

    return 1


def aa_distribution_at_position(
    logits_output,
    pos_0based: int,
    seq_len: int,
    vocab: Dict[str, int]
):
    logits_tensor = extract_sequence_logits(logits_output)
    offset = infer_position_offset(logits_tensor, seq_len)

    logit_vec = logits_tensor[pos_0based + offset]
    logp_vec = torch.log_softmax(logit_vec, dim=-1)

    rows = []

    for aa in VALID_AA:
        token_id = vocab[aa]
        logprob = float(logp_vec[token_id].detach().cpu())
        prob = float(math.exp(logprob))

        rows.append({
            "AA": aa,
            "prob": prob,
            "logprob": logprob
        })

    rows = sorted(rows, key=lambda x: x["prob"], reverse=True)

    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return rows


def summarize_esmc_support(rank: int, llr: float) -> str:
    if rank == 1:
        return "ESMC_TOP1_MATCH"

    if rank <= 3:
        return "ESMC_TOP3_SUPPORT"

    if rank <= 5:
        return "ESMC_TOP5_SUPPORT"

    if llr > 0:
        return "ESMC_POSITIVE_LLR_BUT_LOW_RANK"

    return "LOW_ESMC_SUPPORT"


# =========================
# Cross-check main function
# =========================

def run_esmc_crosscheck_for_esm3_mutations(
    ref_seq: str,
    final_seq: str,
    final_report_rows: Optional[List[Dict[str, object]]] = None,
    token: Optional[str] = None,
    esmc_model_name: str = "esmc-6b-2024-12",
    fallback_model_name: str = "esmc-600m-2024-12",
    output_dir="pipeline_v3_outputs",
    make_heatmap: bool = True
):
    """
    Cross-check accepted ESM3 point mutations using ESMC.

    For each accepted mutation in final_seq:
        1. Mask WT/reference sequence at that position.
        2. Use ESMC to get AA probabilities at the masked position.
        3. Report the ESMC rank/probability of the ESM3 mutation.
        4. Calculate LLR = logP(MUT) - logP(WT).

    Outputs:
        - esm3_esmc_crosscheck.csv
        - esm3_esmc_crosscheck_distribution.csv
        - figures/esm3_esmc_crosscheck_heatmap.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ref_seq = clean_sequence(ref_seq)
    final_seq = clean_sequence(final_seq)

    validate_sequence(ref_seq, "ref_seq")
    validate_sequence(final_seq, "final_seq")

    mutations = find_accepted_mutations(ref_seq, final_seq)
    final_report_lookup = build_final_report_lookup(final_report_rows)

    summary_path = output_dir / "esm3_esmc_crosscheck.csv"
    distribution_path = output_dir / "esm3_esmc_crosscheck_distribution.csv"

    summary_fieldnames = [
        "position_1based",
        "WT",
        "ESM3_final",
        "mutation",
        "ESM3_vote_winner",
        "ESM3_vote_count",
        "ESM3_total_votes",
        "ESM3_vote_frequency",
        "ESM3_decision",
        "ESMC_model",
        "ESMC_rank_of_ESM3_mutation",
        "ESMC_prob_of_ESM3_mutation",
        "ESMC_logprob_of_ESM3_mutation",
        "ESMC_WT_rank",
        "ESMC_WT_prob",
        "ESMC_WT_logprob",
        "ESMC_top1_AA",
        "ESMC_top1_prob",
        "ESMC_top3",
        "ESMC_top5",
        "LLR_mut_vs_WT",
        "crosscheck_status"
    ]

    distribution_fieldnames = [
        "position_1based",
        "mutation",
        "AA",
        "rank",
        "prob",
        "logprob",
        "is_WT",
        "is_ESM3_mutation"
    ]

    if not mutations:
        write_csv(summary_path, [], summary_fieldnames)
        write_csv(distribution_path, [], distribution_fieldnames)

        print("\nESM3 -> ESMC cross-check skipped: no accepted ESM3 mutations.")
        print(f"Saved empty file: {summary_path}")
        return []

    api_token = get_token(token)

    # Try requested ESMC model first, fallback if needed.
    model_used = esmc_model_name

    try:
        client = build_esmc_client(model_used, api_token)
        vocab = get_sequence_vocab()

    except Exception as e:
        print(f"\nCould not initialize {esmc_model_name}: {e}")
        print(f"Trying fallback model: {fallback_model_name}")

        model_used = fallback_model_name
        client = build_esmc_client(model_used, api_token)
        vocab = get_sequence_vocab()

    summary_rows = []
    distribution_rows = []

    print("\nRunning ESMC cross-check for ESM3 accepted mutations...")
    print(f"ESMC model: {model_used}")
    print(f"Accepted ESM3 mutations: {len(mutations)}")

    for m in mutations:
        pos1 = int(m["position_1based"])
        pos0 = int(m["position_0based"])
        wt_aa = m["WT"]
        mut_aa = m["MUT"]
        mutation_name = m["mutation"]

        masked_seq = mask_one_position(ref_seq, pos0)
        logits_output = run_esmc_logits(client, masked_seq)

        aa_rows = aa_distribution_at_position(
            logits_output=logits_output,
            pos_0based=pos0,
            seq_len=len(ref_seq),
            vocab=vocab
        )

        aa_lookup = {row["AA"]: row for row in aa_rows}

        mut_row = aa_lookup[mut_aa]
        wt_row = aa_lookup[wt_aa]

        llr = mut_row["logprob"] - wt_row["logprob"]
        status = summarize_esmc_support(mut_row["rank"], llr)

        top1 = aa_rows[0]
        top3 = ";".join([f"{r['AA']}:{r['prob']:.5f}" for r in aa_rows[:3]])
        top5 = ";".join([f"{r['AA']}:{r['prob']:.5f}" for r in aa_rows[:5]])

        report = final_report_lookup.get(pos1, {})

        summary_rows.append({
            "position_1based": pos1,
            "WT": wt_aa,
            "ESM3_final": mut_aa,
            "mutation": mutation_name,
            "ESM3_vote_winner": report.get("vote_winner", ""),
            "ESM3_vote_count": report.get("vote_count", ""),
            "ESM3_total_votes": report.get("total_votes", ""),
            "ESM3_vote_frequency": report.get("vote_frequency", ""),
            "ESM3_decision": report.get("decision", ""),
            "ESMC_model": model_used,
            "ESMC_rank_of_ESM3_mutation": mut_row["rank"],
            "ESMC_prob_of_ESM3_mutation": mut_row["prob"],
            "ESMC_logprob_of_ESM3_mutation": mut_row["logprob"],
            "ESMC_WT_rank": wt_row["rank"],
            "ESMC_WT_prob": wt_row["prob"],
            "ESMC_WT_logprob": wt_row["logprob"],
            "ESMC_top1_AA": top1["AA"],
            "ESMC_top1_prob": top1["prob"],
            "ESMC_top3": top3,
            "ESMC_top5": top5,
            "LLR_mut_vs_WT": llr,
            "crosscheck_status": status
        })

        for row in aa_rows:
            distribution_rows.append({
                "position_1based": pos1,
                "mutation": mutation_name,
                "AA": row["AA"],
                "rank": row["rank"],
                "prob": row["prob"],
                "logprob": row["logprob"],
                "is_WT": row["AA"] == wt_aa,
                "is_ESM3_mutation": row["AA"] == mut_aa
            })

    write_csv(summary_path, summary_rows, summary_fieldnames)
    write_csv(distribution_path, distribution_rows, distribution_fieldnames)

    print(f"Saved: {summary_path}")
    print(f"Saved: {distribution_path}")

    if make_heatmap:
        make_esm3_esmc_crosscheck_heatmap(
            summary_rows=summary_rows,
            output_dir=output_dir
        )

    return summary_rows


# =========================
# Visualization
# =========================

def make_esm3_esmc_crosscheck_heatmap(summary_rows, output_dir="pipeline_v3_outputs"):
    """
    Make a compact heatmap only for ESM3 accepted mutation positions.

    Rows:
        WT
        ESM3_final
        ESMC_top1
        ESMC_rank
        LLR
    """
    if not summary_rows:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; heatmap skipped.")
        return

    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    positions = [str(row["position_1based"]) for row in summary_rows]

    row_names = [
        "WT",
        "ESM3_final",
        "ESMC_top1",
        "ESMC_rank",
        "LLR"
    ]

    label_matrix = []
    score_matrix = []

    wt_labels = []
    wt_scores = []

    esm3_labels = []
    esm3_scores = []

    top1_labels = []
    top1_scores = []

    rank_labels = []
    rank_scores = []

    llr_labels = []
    llr_scores = []

    for row in summary_rows:
        wt = row["WT"]
        mut = row["ESM3_final"]
        top1 = row["ESMC_top1_AA"]

        rank = int(row["ESMC_rank_of_ESM3_mutation"])
        llr = float(row["LLR_mut_vs_WT"])

        wt_labels.append(wt)
        wt_scores.append(0)

        if rank == 1:
            esm3_labels.append(mut + "★")
            esm3_scores.append(3)
        elif rank <= 3:
            esm3_labels.append(mut + "✓")
            esm3_scores.append(2)
        elif rank <= 5:
            esm3_labels.append(mut + "✓")
            esm3_scores.append(1)
        else:
            esm3_labels.append(mut + "×")
            esm3_scores.append(-1)

        top1_labels.append(top1)
        if top1 == mut:
            top1_scores.append(3)
        elif top1 == wt:
            top1_scores.append(-1)
        else:
            top1_scores.append(1)

        rank_labels.append(f"R{rank}")
        if rank == 1:
            rank_scores.append(3)
        elif rank <= 3:
            rank_scores.append(2)
        elif rank <= 5:
            rank_scores.append(1)
        else:
            rank_scores.append(-1)

        llr_labels.append(f"{llr:.2f}")
        if llr > 0:
            llr_scores.append(2)
        else:
            llr_scores.append(-1)

    label_matrix.extend([
        wt_labels,
        esm3_labels,
        top1_labels,
        rank_labels,
        llr_labels
    ])

    score_matrix.extend([
        wt_scores,
        esm3_scores,
        top1_scores,
        rank_scores,
        llr_scores
    ])

    # Save label/score CSVs
    labels_path = output_dir / "esm3_esmc_crosscheck_heatmap_labels.csv"
    scores_path = output_dir / "esm3_esmc_crosscheck_heatmap_scores.csv"

    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + positions)

        for name, labels in zip(row_names, label_matrix):
            writer.writerow([name] + labels)

    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["layer"] + positions)

        for name, scores in zip(row_names, score_matrix):
            writer.writerow([name] + scores)

    # Plot
    n_rows = len(row_names)
    n_cols = len(positions)

    fig_w = max(8, n_cols * 0.75)
    fig_h = max(4, n_rows * 0.55)

    plt.figure(figsize=(fig_w, fig_h))

    plt.imshow(
        score_matrix,
        aspect="auto",
        vmin=-1,
        vmax=3
    )

    plt.colorbar(label="Cross-check score")

    plt.xticks(
        ticks=range(n_cols),
        labels=positions,
        rotation=90
    )

    plt.yticks(
        ticks=range(n_rows),
        labels=row_names
    )

    for i in range(n_rows):
        for j in range(n_cols):
            plt.text(
                j,
                i,
                str(label_matrix[i][j]),
                ha="center",
                va="center",
                fontsize=8
            )

    plt.xlabel("Position")
    plt.ylabel("Cross-check layer")
    plt.title("ESM3 Final Mutations Cross-checked by ESMC")
    plt.tight_layout()

    fig_path = fig_dir / "esm3_esmc_crosscheck_heatmap.png"
    plt.savefig(fig_path, dpi=400)
    plt.show()

    print(f"Saved: {labels_path}")
    print(f"Saved: {scores_path}")
    print(f"Saved: {fig_path}")


# =========================
# Optional standalone mode
# =========================

def run_standalone_from_outputs(
    output_dir="pipeline_v3_outputs",
    token=None,
    esmc_model_name="esmc-6b-2024-12",
    fallback_model_name="esmc-600m-2024-12"
):
    """
    Optional mode after pipeline has already run.

    Because the original pipeline does not always save WT/reference as a separate txt,
    this standalone mode asks for WT/reference sequence, then reads:
        - step5_final.txt
        - step5_final_vote_report.csv
    """
    output_dir = Path(output_dir)

    print("\nPaste the same WT/reference sequence used in the ESM3 pipeline.")
    ref_seq = input("Reference sequence: ")
    ref_seq = clean_sequence(ref_seq)
    validate_sequence(ref_seq, "ref_seq")

    final_path = output_dir / "step5_final.txt"
    report_path = output_dir / "step5_final_vote_report.csv"

    if not final_path.exists():
        raise FileNotFoundError(f"Cannot find {final_path}")

    final_seq = read_text_sequence(final_path)
    final_report_rows = read_final_vote_report(report_path)

    return run_esmc_crosscheck_for_esm3_mutations(
        ref_seq=ref_seq,
        final_seq=final_seq,
        final_report_rows=final_report_rows,
        token=token,
        esmc_model_name=esmc_model_name,
        fallback_model_name=fallback_model_name,
        output_dir=output_dir,
        make_heatmap=True
    )


if __name__ == "__main__":
    run_standalone_from_outputs()
