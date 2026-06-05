# ============================================================
# ESMC
#
# Mode 1: WT-only position scan
# Mode 2: Compare candidate sequences against WT
#         + full sequence-level match heatmap
#
# This script asks for:
# - Biohub API token
# - Model name
# - WT sequence
# - Candidate sequences, if using compare mode
# ============================================================

import os
from getpass import getpass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from esm.sdk import esmc_client
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
from esm.tokenization import get_esmc_model_tokenizers


VALID_AA = list("ACDEFGHIKLMNPQRSTVWY")
VALID_AA_SET = set(VALID_AA)


# ============================================================
# Basic input helpers
# ============================================================

def clean_sequence(seq: str) -> str:
    seq = seq.strip().upper()
    seq = seq.replace(" ", "")
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    return seq


def validate_sequence(seq: str, name: str = "sequence") -> None:
    bad = sorted(set(seq) - VALID_AA_SET)

    if bad:
        raise ValueError(f"{name} has invalid amino acid(s): {bad}")


def ask_sequence(title: str, name: str = "sequence") -> str:
    print(f"\n{title}")
    seq = input("> ")
    seq = clean_sequence(seq)
    validate_sequence(seq, name)
    return seq


def ask_token() -> str:
    token = os.getenv("BIOHUB_TOKEN")

    if token:
        print("Using BIOHUB_TOKEN from environment.")
        return token

    return getpass("Biohub API token: ")


def ask_model_name() -> str:
    print("\nChoose ESMC model.")
    print("Try: esmc-6b-2024-12")
    print("Fallback/test: esmc-600m-2024-12")

    model = input("Model name [Enter = esmc-6b-2024-12]: ").strip()

    if not model:
        model = "esmc-6b-2024-12"

    return model


def parse_fasta_text(text: str) -> Dict[str, str]:
    records = {}
    name = None
    seq_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(">"):
            if name is not None:
                records[name] = clean_sequence("".join(seq_parts))

            name = line[1:].strip()

            if not name:
                name = f"seq_{len(records) + 1}"

            seq_parts = []

        else:
            seq_parts.append(line)

    if name is not None:
        records[name] = clean_sequence("".join(seq_parts))

    return records


def ask_variants_fasta() -> Dict[str, str]:
    print("\nPaste candidate sequences in FASTA format.")
    print("Example:")
    print(">candidate_1")
    print("MVLSEGE...")
    print(">candidate_2")
    print("MVLSEGE...")
    print("\nWhen finished, type END on a new line.\n")

    lines = []

    while True:
        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    text = "\n".join(lines)
    records = parse_fasta_text(text)

    if not records:
        raise ValueError("No candidate sequence found.")

    for name, seq in records.items():
        validate_sequence(seq, name)

    return records


# ============================================================
# ESMC helpers
# ============================================================

def build_client(model_name: str, token: str):
    client = esmc_client(
        model=model_name,
        url="https://biohub.ai",
        token=token
    )

    return client


def get_sequence_vocab() -> Dict[str, int]:
    tokenizers = get_esmc_model_tokenizers()

    # Different esm versions may expose the sequence tokenizer differently.
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
    return seq[:pos_0based] + "_" + seq[pos_0based + 1:]


def run_logits(client, sequence: str):
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    if isinstance(protein_tensor, ESMProteinError):
        raise RuntimeError(protein_tensor)

    output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True)
    )

    return output


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
    Some model outputs include BOS/EOS tokens.
    If output length = seq length + 2, residue i maps to i + 1.
    If output length = seq length, residue i maps to i.
    """

    L = logits_tensor.shape[0]

    if L == seq_len:
        return 0

    if L == seq_len + 1:
        return 1

    if L == seq_len + 2:
        return 1

    # safe fallback for ESM-like tokenization
    return 1


def aa_logprob_and_prob(
    logits_output,
    pos_0based: int,
    seq_len: int,
    vocab: Dict[str, int]
) -> Tuple[Dict[str, float], Dict[str, float]]:

    logits_tensor = extract_sequence_logits(logits_output)
    offset = infer_position_offset(logits_tensor, seq_len)

    logit_vec = logits_tensor[pos_0based + offset]
    logp_vec = torch.log_softmax(logit_vec, dim=-1)

    aa_logprob = {}
    aa_prob = {}

    for aa in VALID_AA:
        token_id = vocab[aa]
        lp = float(logp_vec[token_id].detach().cpu())
        aa_logprob[aa] = lp
        aa_prob[aa] = float(np.exp(lp))

    return aa_logprob, aa_prob


def calc_entropy(prob_dict: Dict[str, float]) -> float:
    probs = np.array(list(prob_dict.values()), dtype=float)
    probs = probs / probs.sum()

    return float(-(probs * np.log(probs + 1e-12)).sum())


# ============================================================
# Main analysis
# ============================================================

def wt_position_scan(
    client,
    wt_seq: str,
    vocab: Dict[str, int],
    top_k: int = 5
) -> Tuple[pd.DataFrame, List[Dict[str, float]], List[Dict[str, float]]]:

    rows = []
    saved_logprobs = []
    saved_probs = []

    for pos, wt_aa in tqdm(
        list(enumerate(wt_seq)),
        desc="WT leave-one-out scan"
    ):
        masked_seq = mask_one_position(wt_seq, pos)
        logits_output = run_logits(client, masked_seq)

        aa_logprob, aa_prob = aa_logprob_and_prob(
            logits_output=logits_output,
            pos_0based=pos,
            seq_len=len(wt_seq),
            vocab=vocab
        )

        saved_logprobs.append(aa_logprob)
        saved_probs.append(aa_prob)

        sorted_probs = sorted(
            aa_prob.items(),
            key=lambda x: x[1],
            reverse=True
        )

        ranks = {
            aa: rank + 1
            for rank, (aa, prob) in enumerate(sorted_probs)
        }

        top_items = sorted_probs[:top_k]

        rows.append({
            "position_1based": pos + 1,
            "WT": wt_aa,
            "WT_prob": aa_prob[wt_aa],
            "WT_logprob": aa_logprob[wt_aa],
            "WT_rank": ranks[wt_aa],
            "top1_AA": top_items[0][0],
            "top1_prob": top_items[0][1],
            "entropy": calc_entropy(aa_prob),
            "top_k": ";".join([f"{aa}:{prob:.5f}" for aa, prob in top_items])
        })

    df = pd.DataFrame(rows)

    return df, saved_logprobs, saved_probs


def compare_variants_to_wt(
    wt_seq: str,
    variants: Dict[str, str],
    saved_logprobs: List[Dict[str, float]],
    saved_probs: List[Dict[str, float]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    summary_rows = []
    detail_rows = []

    for variant_name, variant_seq in variants.items():

        if len(variant_seq) != len(wt_seq):
            raise ValueError(
                f"{variant_name} length mismatch: "
                f"WT={len(wt_seq)}, variant={len(variant_seq)}"
            )

        n_mutations = 0
        total_llr = 0.0
        mutation_names = []

        for pos, (wt_aa, mut_aa) in enumerate(zip(wt_seq, variant_seq)):

            if wt_aa == mut_aa:
                continue

            aa_logprob = saved_logprobs[pos]
            aa_prob = saved_probs[pos]

            llr = aa_logprob[mut_aa] - aa_logprob[wt_aa]

            n_mutations += 1
            total_llr += llr

            mutation_name = f"{wt_aa}{pos + 1}{mut_aa}"
            mutation_names.append(mutation_name)

            if llr > 0:
                interpretation = "ESMC_prefers_MUT_over_WT"
            elif llr < 0:
                interpretation = "ESMC_prefers_WT_over_MUT"
            else:
                interpretation = "neutral"

            detail_rows.append({
                "variant_name": variant_name,
                "position_1based": pos + 1,
                "WT": wt_aa,
                "MUT": mut_aa,
                "mutation": mutation_name,
                "WT_prob": aa_prob[wt_aa],
                "MUT_prob": aa_prob[mut_aa],
                "WT_logprob": aa_logprob[wt_aa],
                "MUT_logprob": aa_logprob[mut_aa],
                "LLR": llr,
                "interpretation": interpretation
            })

        average_llr = total_llr / n_mutations if n_mutations > 0 else 0.0

        summary_rows.append({
            "variant_name": variant_name,
            "n_mutations": n_mutations,
            "total_LLR": total_llr,
            "average_LLR": average_llr,
            "mutations": ";".join(mutation_names)
        })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["average_LLR", "total_LLR"],
            ascending=False
        )

    return summary_df, detail_df


# ============================================================
# Full sequence-level ESMC match heatmap
# ============================================================

def parse_topk_cell(top_k_cell, n=3):
    """
    Parse top_k string like:
    A:0.50000;V:0.20000;L:0.10000
    """
    if pd.isna(top_k_cell):
        return []

    items = []

    for part in str(top_k_cell).split(";"):
        part = part.strip()

        if not part or ":" not in part:
            continue

        aa, prob = part.split(":", 1)

        try:
            items.append((aa.strip(), float(prob)))
        except ValueError:
            continue

    return items[:n]


def make_full_sequence_match_heatmap(
    wt_seq: str,
    variants: Dict[str, str],
    wt_scan_df: pd.DataFrame,
    detail_df: pd.DataFrame
) -> None:
    """
    Create one full-sequence heatmap.

    Rows:
    - WT
    - ESMC_top1
    - candidate sequences

    Columns:
    - protein positions from 1 to final residue

    Symbols:
    ★ = candidate matches ESMC top1
    ✓ = candidate matches ESMC top3
    × = candidate mutation does not match ESMC suggestion
    · = candidate keeps WT while ESMC suggests mutation
    """

    import matplotlib.pyplot as plt

    outdir = "outputs"
    figdir = os.path.join(outdir, "figures")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    wt_scan_df = wt_scan_df.copy()
    wt_scan_df["position_1based"] = wt_scan_df["position_1based"].astype(int)

    # ------------------------------------------------------------
    # Build ESMC suggestion table
    # ------------------------------------------------------------

    suggestion_rows = []

    for _, row in wt_scan_df.iterrows():
        pos = int(row["position_1based"])
        wt_aa = row["WT"]

        top_items = parse_topk_cell(row["top_k"], n=3)

        top1_aa = top_items[0][0] if len(top_items) >= 1 else row["top1_AA"]
        top1_prob = top_items[0][1] if len(top_items) >= 1 else row["top1_prob"]

        top2_aa = top_items[1][0] if len(top_items) >= 2 else ""
        top2_prob = top_items[1][1] if len(top_items) >= 2 else np.nan

        top3_aa = top_items[2][0] if len(top_items) >= 3 else ""
        top3_prob = top_items[2][1] if len(top_items) >= 3 else np.nan

        top3_set = "".join([aa for aa, prob in top_items])

        if top1_aa == wt_aa:
            suggestion_status = "WT_best"
            suggested_AA = wt_aa
        else:
            suggestion_status = "ESMC_suggests_mutation"
            suggested_AA = top1_aa

        suggestion_rows.append({
            "position_1based": pos,
            "WT": wt_aa,
            "WT_prob": row["WT_prob"],
            "WT_rank": row["WT_rank"],
            "entropy": row["entropy"],
            "top1_AA": top1_aa,
            "top1_prob": top1_prob,
            "top2_AA": top2_aa,
            "top2_prob": top2_prob,
            "top3_AA": top3_aa,
            "top3_prob": top3_prob,
            "top3_set": top3_set,
            "suggested_AA": suggested_AA,
            "suggestion_status": suggestion_status
        })

    suggestion_df = pd.DataFrame(suggestion_rows)

    suggestion_path = os.path.join(outdir, "esmc_position_suggestion_table.csv")
    suggestion_df.to_csv(suggestion_path, index=False)

    suggestion_map = suggestion_df.set_index("position_1based")

    # ------------------------------------------------------------
    # Build detail lookup for LLR
    # ------------------------------------------------------------

    detail_lookup = {}

    if detail_df is not None and not detail_df.empty:
        for _, row in detail_df.iterrows():
            key = (row["variant_name"], int(row["position_1based"]))
            detail_lookup[key] = row

    # ------------------------------------------------------------
    # Candidate match table
    # ------------------------------------------------------------

    candidate_rows = []

    for variant_name, variant_seq in variants.items():

        if len(variant_seq) != len(wt_seq):
            raise ValueError(
                f"{variant_name} length mismatch: "
                f"WT={len(wt_seq)}, variant={len(variant_seq)}"
            )

        for pos in range(1, len(wt_seq) + 1):
            wt_aa = wt_seq[pos - 1]
            cand_aa = variant_seq[pos - 1]

            srow = suggestion_map.loc[pos]
            top1_aa = srow["top1_AA"]
            top3_set = str(srow["top3_set"])

            llr = np.nan
            mut_prob = np.nan

            if cand_aa != wt_aa:
                drow = detail_lookup.get((variant_name, pos), None)

                if drow is not None:
                    llr = drow["LLR"]
                    mut_prob = drow["MUT_prob"]

            # Score and label logic
            if cand_aa == wt_aa and top1_aa == wt_aa:
                match_status = "KEEP_WT_AND_WT_IS_BEST"
                score = 1
                label = cand_aa

            elif cand_aa == wt_aa and top1_aa != wt_aa:
                match_status = "KEEP_WT_BUT_ESMC_SUGGESTS_MUTATION"
                score = 0
                label = cand_aa + "·"

            elif cand_aa != wt_aa and cand_aa == top1_aa:
                match_status = "MATCH_ESMC_TOP1"
                score = 3
                label = cand_aa + "★"

            elif cand_aa != wt_aa and cand_aa in top3_set:
                match_status = "MATCH_ESMC_TOP3"
                score = 2
                label = cand_aa + "✓"

            elif cand_aa != wt_aa and top1_aa == wt_aa:
                match_status = "MUTATION_WHILE_WT_IS_BEST"
                score = -2
                label = cand_aa + "×"

            else:
                match_status = "MUTATION_NOT_IN_ESMC_TOP3"
                score = -1
                label = cand_aa + "×"

            candidate_rows.append({
                "variant_name": variant_name,
                "position_1based": pos,
                "WT": wt_aa,
                "candidate_AA": cand_aa,
                "ESMC_top1_AA": top1_aa,
                "ESMC_top3_set": top3_set,
                "WT_rank": srow["WT_rank"],
                "WT_prob": srow["WT_prob"],
                "MUT_prob": mut_prob,
                "LLR": llr,
                "match_status": match_status,
                "match_score": score,
                "heatmap_label": label
            })

    candidate_match_df = pd.DataFrame(candidate_rows)

    candidate_match_path = os.path.join(outdir, "esmc_candidate_sequence_match_table.csv")
    candidate_match_df.to_csv(candidate_match_path, index=False)

    # ------------------------------------------------------------
    # Build full alignment-like matrix
    # ------------------------------------------------------------

    positions = list(range(1, len(wt_seq) + 1))
    row_names = ["WT", "ESMC_top1"] + list(variants.keys())

    label_matrix = []
    score_matrix = []

    # WT row
    wt_labels = []
    wt_scores = []

    for pos in positions:
        wt_aa = wt_seq[pos - 1]
        top1_aa = suggestion_map.loc[pos, "top1_AA"]

        wt_labels.append(wt_aa)

        if top1_aa == wt_aa:
            wt_scores.append(1)
        else:
            wt_scores.append(0)

    label_matrix.append(wt_labels)
    score_matrix.append(wt_scores)

    # ESMC top1 row
    esmc_labels = []
    esmc_scores = []

    for pos in positions:
        wt_aa = wt_seq[pos - 1]
        top1_aa = suggestion_map.loc[pos, "top1_AA"]

        if top1_aa == wt_aa:
            esmc_labels.append(top1_aa)
            esmc_scores.append(1)
        else:
            esmc_labels.append(top1_aa + "★")
            esmc_scores.append(3)

    label_matrix.append(esmc_labels)
    score_matrix.append(esmc_scores)

    # Candidate rows
    for variant_name, variant_seq in variants.items():
        sub = candidate_match_df[
            candidate_match_df["variant_name"] == variant_name
        ].set_index("position_1based")

        labels = []
        scores = []

        for pos in positions:
            labels.append(sub.loc[pos, "heatmap_label"])
            scores.append(sub.loc[pos, "match_score"])

        label_matrix.append(labels)
        score_matrix.append(scores)

    label_df = pd.DataFrame(
        label_matrix,
        index=row_names,
        columns=[str(p) for p in positions]
    )

    score_df = pd.DataFrame(
        score_matrix,
        index=row_names,
        columns=[str(p) for p in positions]
    )

    label_path = os.path.join(outdir, "esmc_full_sequence_alignment_labels.csv")
    score_path = os.path.join(outdir, "esmc_full_sequence_alignment_scores.csv")

    label_df.to_csv(label_path)
    score_df.to_csv(score_path)

    # ------------------------------------------------------------
    # Plot full heatmap
    # ------------------------------------------------------------

    n_rows, n_cols = score_df.shape

    fig_w = max(18, n_cols * 0.24)
    fig_h = max(4, n_rows * 0.48)

    plt.figure(figsize=(fig_w, fig_h))

    plt.imshow(
        score_df.values.astype(float),
        aspect="auto",
        vmin=-2,
        vmax=3
    )

    plt.colorbar(label="Match score")

    plt.xticks(
        ticks=range(n_cols),
        labels=score_df.columns,
        rotation=90
    )

    plt.yticks(
        ticks=range(n_rows),
        labels=score_df.index
    )

    if n_cols > 180:
        font_size = 3
    elif n_cols > 120:
        font_size = 4
    elif n_cols > 80:
        font_size = 5
    elif n_cols > 50:
        font_size = 6
    else:
        font_size = 8

    for i in range(n_rows):
        for j in range(n_cols):
            plt.text(
                j,
                i,
                str(label_df.iloc[i, j]),
                ha="center",
                va="center",
                fontsize=font_size
            )

    plt.xlabel("Position")
    plt.ylabel("Sequence")
    plt.title("ESMC Full Sequence Match Heatmap")
    plt.tight_layout()

    fig_path = os.path.join(figdir, "esmc_full_sequence_match_heatmap.png")
    plt.savefig(fig_path, dpi=400)
    plt.show()

    print("\nSaved full sequence match heatmap outputs:")
    print(suggestion_path)
    print(candidate_match_path)
    print(label_path)
    print(score_path)
    print(fig_path)


# ============================================================
# Output and display
# ============================================================

def save_outputs(
    wt_scan_df: pd.DataFrame,
    summary_df: pd.DataFrame = None,
    detail_df: pd.DataFrame = None
) -> None:

    outdir = "outputs"
    os.makedirs(outdir, exist_ok=True)

    wt_path = os.path.join(outdir, "esmc_wt_position_scan.csv")
    wt_scan_df.to_csv(wt_path, index=False)

    print(f"\nSaved: {wt_path}")

    if summary_df is not None:
        summary_path = os.path.join(outdir, "esmc_variant_score_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

    if detail_df is not None:
        detail_path = os.path.join(outdir, "esmc_variant_mutation_details.csv")
        detail_df.to_csv(detail_path, index=False)
        print(f"Saved: {detail_path}")


def print_wt_summary(wt_scan_df: pd.DataFrame) -> None:
    print("\n============================================================")
    print("WT scan summary")
    print("============================================================")

    not_top1 = wt_scan_df[wt_scan_df["WT_rank"] != 1].copy()

    print("\nPositions where WT is not top-1:")

    if not_top1.empty:
        print("All WT residues are top-1 by ESMC.")
    else:
        cols = [
            "position_1based",
            "WT",
            "WT_rank",
            "WT_prob",
            "top1_AA",
            "top1_prob",
            "entropy"
        ]
        print(not_top1[cols].to_string(index=False))

    print("\nMost constrained positions by low entropy:")
    cols = [
        "position_1based",
        "WT",
        "WT_rank",
        "WT_prob",
        "top1_AA",
        "top1_prob",
        "entropy"
    ]
    print(
        wt_scan_df
        .sort_values("entropy", ascending=True)
        .head(10)[cols]
        .to_string(index=False)
    )

    print("\nMost tolerant positions by high entropy:")
    print(
        wt_scan_df
        .sort_values("entropy", ascending=False)
        .head(10)[cols]
        .to_string(index=False)
    )


def print_variant_summary(summary_df: pd.DataFrame, detail_df: pd.DataFrame) -> None:
    print("\n============================================================")
    print("Variant ranking")
    print("============================================================")

    if summary_df.empty:
        print("No variants to compare.")
        return

    print(summary_df.to_string(index=False))

    if not detail_df.empty:
        cols = [
            "variant_name",
            "mutation",
            "LLR",
            "WT_prob",
            "MUT_prob",
            "interpretation"
        ]

        print("\nTop positive mutations:")
        print(
            detail_df
            .sort_values("LLR", ascending=False)
            .head(20)[cols]
            .to_string(index=False)
        )

        print("\nMost negative mutations:")
        print(
            detail_df
            .sort_values("LLR", ascending=True)
            .head(20)[cols]
            .to_string(index=False)
        )


# ============================================================
# Main program
# ============================================================

def main():
    print("============================================================")
    print("ESMC Analyzer")
    print("============================================================")
    print("1) WT-only position scan")
    print("2) Compare candidate sequences against WT")
    print("============================================================")

    mode = input("Choose mode [1/2]: ").strip()

    if mode not in {"1", "2"}:
        raise ValueError("Please choose 1 or 2.")

    token = ask_token()
    model_name = ask_model_name()

    wt_seq = ask_sequence("Paste WT protein sequence:", name="WT")

    print("\nConnecting to Biohub...")
    client = build_client(model_name, token)
    vocab = get_sequence_vocab()

    print(f"\nModel: {model_name}")
    print(f"WT length: {len(wt_seq)}")

    wt_scan_df, saved_logprobs, saved_probs = wt_position_scan(
        client=client,
        wt_seq=wt_seq,
        vocab=vocab,
        top_k=5
    )

    print_wt_summary(wt_scan_df)

    if mode == "1":
        save_outputs(wt_scan_df=wt_scan_df)

    elif mode == "2":
        variants = ask_variants_fasta()

        summary_df, detail_df = compare_variants_to_wt(
            wt_seq=wt_seq,
            variants=variants,
            saved_logprobs=saved_logprobs,
            saved_probs=saved_probs
        )

        print_variant_summary(summary_df, detail_df)

        make_full_sequence_match_heatmap(
            wt_seq=wt_seq,
            variants=variants,
            wt_scan_df=wt_scan_df,
            detail_df=detail_df
        )

        save_outputs(
            wt_scan_df=wt_scan_df,
            summary_df=summary_df,
            detail_df=detail_df
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
