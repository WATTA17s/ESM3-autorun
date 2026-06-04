# =========================
# ESM3 Conservative Consensus-Guided Mutagenesis Pipeline V3
# Model: esm3-medium-2024-08
# =========================

from pathlib import Path
from collections import Counter
from getpass import getpass
import csv
import json

import esm
from esm.sdk.api import ESMProtein, GenerationConfig


# =========================
# CONFIG
# =========================
MODEL_NAME = "esm3-medium-2024-08"

MASK_CHAR = "_"

WINDOW = 10
STEP = 10

N_SAMPLES = 20
TEMPERATURE = 0.5
NUM_STEPS = 8

FINALIST_THRESHOLD = 0.6

OUTPUT_DIR = Path("pipeline_v3_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_LOG = OUTPUT_DIR / "full_pipeline_log.txt"


# =========================
# MODEL
# =========================
from esm.sdk.forge import ESM3ForgeInferenceClient

def create_model(model_name=MODEL_NAME, token=None):

    if token is None:
        token = getpass("Enter ESM API key: ")

    client = ESM3ForgeInferenceClient(
        model=model_name,
        url="https://biohub.ai",
        token=token,
    )

    return client

# =========================
# FILE HELPERS
# =========================
def save_text(path, text):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_lines(path, lines):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def append_log(text):
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(text)


def save_csv(path, rows, fieldnames):
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_json(path, obj):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================
# SEQUENCE HELPERS
# =========================
def clean_sequence(seq):
    """
    Remove whitespace/newlines and uppercase the reference sequence.
    """
    return "".join(str(seq).split()).upper()


def strip_fasta(raw_text):
    """
    Accept plain amino-acid sequence or FASTA-like text.
    Header lines beginning with '>' are ignored.
    """
    lines = str(raw_text).splitlines()
    seq_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            continue
        seq_lines.append(line)

    return "".join(seq_lines)


def prompt_reference_sequence():
    """
    Ask user for the full WT/reference amino-acid sequence at runtime.

    Best for Colab/terminal use:
        - Paste the full sequence as one line.
        - Do not leave '...' in the sequence.
        - FASTA header is optional if pasted as text.
    """
    print("\nPaste the full WT/reference amino-acid sequence.")
    print("Use one line if possible. Do not include '...'.")

    raw_seq = input("Reference sequence: ")
    ref_seq = clean_sequence(strip_fasta(raw_seq))
    validate_reference_sequence(ref_seq)

    print(f"Reference length: {len(ref_seq)} aa")
    return ref_seq


def validate_reference_sequence(ref_seq):
    if not ref_seq:
        raise ValueError("Reference sequence is empty.")

    if MASK_CHAR in ref_seq:
        raise ValueError(
            f"Reference sequence must not contain mask character '{MASK_CHAR}'."
        )

    if "..." in ref_seq:
        raise ValueError(
            "Reference sequence still contains '...'. Replace it with the full amino-acid sequence."
        )


def build_masked_sequences(ref_seq, window, step):
    """
    Build sliding-window masked sequences.

    Returns:
        list of dict:
        {
            "mask_id": int,
            "start": 0-based start,
            "end": 0-based exclusive end,
            "sequence": masked sequence
        }
    """
    masked_items = []
    n = len(ref_seq)

    mask_id = 1
    for start in range(0, n, step):
        end = min(start + window, n)

        seq = list(ref_seq)

        for i in range(start, end):
            seq[i] = MASK_CHAR

        masked_items.append({
            "mask_id": mask_id,
            "start": start,
            "end": end,
            "sequence": "".join(seq)
        })

        mask_id += 1

    return masked_items


def masked_positions(seq):
    return [i for i, c in enumerate(seq) if c == MASK_CHAR]


def mutation_string(ref_seq, final_seq):
    muts = []

    for i, (wt, mut) in enumerate(zip(ref_seq, final_seq), start=1):
        if wt != mut:
            muts.append(f"{wt}{i}{mut}")

    return muts


# =========================
# ESM GENERATION
# =========================
def generate_predictions(model, masked_seq):
    """
    Generate N_SAMPLES predictions from ESM3 for a masked sequence.
    """
    preds = []

    for sample_idx in range(N_SAMPLES):
        protein = ESMProtein(sequence=masked_seq)

        out = model.generate(
            protein,
            GenerationConfig(
                track="sequence",
                temperature=TEMPERATURE,
                num_steps=NUM_STEPS
            )
        )

        if not hasattr(out, "sequence") or out.sequence is None:
            raise RuntimeError(
                f"ESM generation failed at sample {sample_idx + 1}: {out}"
            )

        seq = out.sequence

        if len(seq) != len(masked_seq):
            raise RuntimeError(
                "ESM output length mismatch.\n"
                f"Input length:  {len(masked_seq)}\n"
                f"Output length: {len(seq)}\n"
                f"Sample: {sample_idx + 1}"
            )

        preds.append(seq)

    return preds


# =========================
# STEP 2: STRICT CONSENSUS + MEMORY
# =========================
def consensus_line(
    masked_seq,
    preds,
    ref_seq,
    strict_mutations,
    mask_id=None,
    start=None,
    end=None
):
    """
    Latest workflow logic:

    For each masked residue position:

    1) 20/20 consensus == WT
       -> write WT
       -> do NOT save to memory

    2) 20/20 consensus != WT
       -> save mutation to strict_mutations memory
       -> write MASK_CHAR "_"

    3) not 20/20 consensus
       -> write WT
       -> do NOT save to memory

    strict_mutations:
        dict[int, set[str]]
        0-based position -> set of strict mutation candidates
    """
    result = list(masked_seq)
    positions = masked_positions(masked_seq)
    audit_rows = []

    for pos in positions:
        aa_list = [p[pos] for p in preds]
        count = Counter(aa_list)

        top_aa, top_count = count.most_common(1)[0]
        frac = top_count / len(preds)

        wt_aa = ref_seq[pos]

        # Rule 1 + 2: strict 20/20 consensus
        if top_count == len(preds):

            # Rule 1: 20/20 == WT
            if top_aa == wt_aa:
                result[pos] = wt_aa
                decision = "WT_CONFIRMED_20_OF_20"

            # Rule 2: 20/20 != WT
            else:
                strict_mutations.setdefault(pos, set()).add(top_aa)
                result[pos] = MASK_CHAR
                decision = "STRICT_MUTATION_MEMORY_AND_MASK"

        # Rule 3: not 20/20
        else:
            result[pos] = wt_aa
            decision = "NOT_20_OF_20_REVERT_TO_WT"

        audit_rows.append({
            "mask_id": mask_id,
            "mask_start_1based": None if start is None else start + 1,
            "mask_end_1based": end,
            "position_1based": pos + 1,
            "wt": wt_aa,
            "top_aa": top_aa,
            "top_count": top_count,
            "n_samples": len(preds),
            "frequency": round(frac, 4),
            "decision": decision,
            "all_counts": ";".join(
                [f"{aa}:{c}" for aa, c in count.most_common()]
            )
        })

    return "".join(result), audit_rows


# =========================
# STEP 3: MERGE CONSENSUS LINES
# =========================
def merge_consensus_lines(lines, ref_seq):
    """
    Merge all consensus lines.

    If any line has "_" at a position, that position is kept as "_"
    in the merged sequence.

    Otherwise, the position is WT.
    """
    merged = []

    for i in range(len(ref_seq)):
        if any(line[i] == MASK_CHAR for line in lines):
            merged.append(MASK_CHAR)
        else:
            merged.append(ref_seq[i])

    return "".join(merged)


# =========================
# STEP 4: THREE-STAGE FINALIST INPUTS
# =========================
def build_three_stage_inputs(final_merged, ref_seq):
    """
    Build 3 final-stage inputs:

    1) full:
       all strict mutation candidates remain masked

    2) front_half:
       front half uses final_merged, back half uses WT

    3) back_half:
       front half uses WT, back half uses final_merged

    Each stage generates 20 predictions.
    Total final-stage predictions = 3 x 20 = 60.
    """
    n = len(final_merged)
    mid = n // 2

    return {
        "full": final_merged,
        "front_half": final_merged[:mid] + ref_seq[mid:],
        "back_half": ref_seq[:mid] + final_merged[mid:]
    }


# =========================
# STEP 5: FINALIST VOTING + MEMORY VALIDATION
# =========================
def build_finalist_seq(
    final_merged,
    ref_seq,
    all_stage_preds,
    strict_mutations
):
    """
    Final mutation acceptance rule:

    A mutation is accepted only if:

    1) position was marked as "_" in final_merged
       meaning it passed strict 20/20 != WT stage

    AND

    2) final-stage voting frequency >= FINALIST_THRESHOLD

    AND

    3) voting winner is the same residue stored in strict_mutations memory

    Otherwise:
        revert to WT
    """
    result = list(ref_seq)
    report_rows = []

    append_log("\n" + "=" * 70 + "\n")
    append_log("STEP 5: FINAL DECISION / MEMORY VALIDATION\n")
    append_log("=" * 70 + "\n")

    for pos, ch in enumerate(final_merged):
        if ch != MASK_CHAR:
            continue

        wt_aa = ref_seq[pos]
        strict_aas = strict_mutations.get(pos, set())

        aa_list = [seq[pos] for seq in all_stage_preds]
        count = Counter(aa_list)

        top_aa, top_count = count.most_common(1)[0]
        frac = top_count / len(all_stage_preds)

        memory_string = ",".join(sorted(strict_aas))

        append_log(
            f"\nPosition {pos + 1}\n"
            f"WT: {wt_aa}\n"
            f"Strict memory: {memory_string}\n"
            f"Voting winner: {top_aa}\n"
            f"Voting count: {top_count}/{len(all_stage_preds)}\n"
            f"Voting frequency: {frac:.4f}\n"
            f"All counts: {dict(count)}\n"
        )

        # Rule A: threshold + memory match
        if (
            frac >= FINALIST_THRESHOLD
            and top_aa in strict_aas
        ):
            result[pos] = top_aa
            decision = "ACCEPT"
            append_log("Decision: ACCEPT\n")

        else:
            result[pos] = wt_aa
            decision = "REVERT_TO_WT"
            append_log("Decision: REVERT TO WT\n")

        report_rows.append({
            "position_1based": pos + 1,
            "wt": wt_aa,
            "strict_memory": memory_string,
            "vote_winner": top_aa,
            "vote_count": top_count,
            "total_votes": len(all_stage_preds),
            "vote_frequency": round(frac, 4),
            "threshold": FINALIST_THRESHOLD,
            "final_residue": result[pos],
            "decision": decision,
            "all_counts": ";".join(
                [f"{aa}:{c}" for aa, c in count.most_common()]
            )
        })

    return "".join(result), report_rows


# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(model, ref_seq):
    ref_seq = clean_sequence(ref_seq)
    validate_reference_sequence(ref_seq)

    save_text(MASTER_LOG, "")

    append_log("===== PIPELINE V3 START =====\n\n")

    append_log("CONFIG\n")
    append_log(f"MODEL_NAME = {MODEL_NAME}\n")
    append_log(f"WINDOW = {WINDOW}\n")
    append_log(f"STEP = {STEP}\n")
    append_log(f"N_SAMPLES = {N_SAMPLES}\n")
    append_log(f"TEMPERATURE = {TEMPERATURE}\n")
    append_log(f"NUM_STEPS = {NUM_STEPS}\n")
    append_log(f"FINALIST_THRESHOLD = {FINALIST_THRESHOLD}\n\n")

    append_log("REFERENCE SEQUENCE:\n")
    append_log(ref_seq + "\n\n")

    # =========================
    # STEP 1: Build masked sequences
    # =========================
    masked_items = build_masked_sequences(ref_seq, WINDOW, STEP)

    save_lines(
        OUTPUT_DIR / "step1_masked.txt",
        [item["sequence"] for item in masked_items]
    )

    save_csv(
        OUTPUT_DIR / "step1_masked_index.csv",
        [
            {
                "mask_id": item["mask_id"],
                "start_1based": item["start"] + 1,
                "end_1based": item["end"],
                "masked_sequence": item["sequence"]
            }
            for item in masked_items
        ],
        ["mask_id", "start_1based", "end_1based", "masked_sequence"]
    )

    append_log("=" * 70 + "\n")
    append_log("STEP 1: MASKED SEQUENCES\n")
    append_log("=" * 70 + "\n")

    for item in masked_items:
        append_log(
            f"\nMASK {item['mask_id']:03d} | "
            f"positions {item['start'] + 1}-{item['end']}\n"
        )
        append_log(item["sequence"] + "\n")

    # =========================
    # STEP 2: Per-mask predictions + strict consensus
    # =========================
    all_consensus_lines = []
    all_consensus_audit_rows = []
    strict_mutations = {}

    append_log("\n" + "=" * 70 + "\n")
    append_log("STEP 2: MASK PREDICTIONS + STRICT CONSENSUS MEMORY\n")
    append_log("=" * 70 + "\n")

    for item in masked_items:
        mask_id = item["mask_id"]
        start = item["start"]
        end = item["end"]
        mseq = item["sequence"]

        print(
            f"[Mask {mask_id}/{len(masked_items)}] "
            f"positions {start + 1}-{end}"
        )

        preds = generate_predictions(model, mseq)

        line, audit_rows = consensus_line(
            masked_seq=mseq,
            preds=preds,
            ref_seq=ref_seq,
            strict_mutations=strict_mutations,
            mask_id=mask_id,
            start=start,
            end=end
        )

        all_consensus_lines.append(line)
        all_consensus_audit_rows.extend(audit_rows)

        # Save predictions for this mask
        save_lines(
            OUTPUT_DIR / f"step2_mask_{mask_id:03d}_predictions.txt",
            preds
        )

        append_log("\n" + "-" * 70 + "\n")
        append_log(f"MASK {mask_id:03d} | positions {start + 1}-{end}\n")
        append_log("-" * 70 + "\n")

        append_log("INPUT MASKED SEQUENCE:\n")
        append_log(mseq + "\n\n")

        append_log(f"{N_SAMPLES} PREDICTIONS:\n")
        for j, seq in enumerate(preds):
            append_log(f"{j + 1:02d}: {seq}\n")

        append_log("\nCONSENSUS LINE:\n")
        append_log(line + "\n")

    save_lines(
        OUTPUT_DIR / "step2_consensus_lines.txt",
        all_consensus_lines
    )

    save_csv(
        OUTPUT_DIR / "step2_consensus_audit.csv",
        all_consensus_audit_rows,
        [
            "mask_id",
            "mask_start_1based",
            "mask_end_1based",
            "position_1based",
            "wt",
            "top_aa",
            "top_count",
            "n_samples",
            "frequency",
            "decision",
            "all_counts"
        ]
    )

    # Save strict mutation memory
    strict_memory_rows = []
    strict_memory_json = {}

    for pos in sorted(strict_mutations):
        aa_set = strict_mutations[pos]
        memory_string = ",".join(sorted(aa_set))

        strict_memory_rows.append({
            "position_1based": pos + 1,
            "wt": ref_seq[pos],
            "strict_memory": memory_string
        })

        strict_memory_json[str(pos + 1)] = {
            "wt": ref_seq[pos],
            "strict_memory": sorted(list(aa_set))
        }

    save_csv(
        OUTPUT_DIR / "step2_strict_mutation_memory.csv",
        strict_memory_rows,
        ["position_1based", "wt", "strict_memory"]
    )

    save_json(
        OUTPUT_DIR / "step2_strict_mutation_memory.json",
        strict_memory_json
    )

    append_log("\n" + "=" * 70 + "\n")
    append_log("STRICT MUTATION MEMORY\n")
    append_log("=" * 70 + "\n")

    if strict_memory_rows:
        for row in strict_memory_rows:
            append_log(
                f"Position {row['position_1based']}: "
                f"{row['wt']} -> {row['strict_memory']}\n"
            )
    else:
        append_log("No strict 20/20 non-WT mutations found.\n")

    # =========================
    # STEP 3: Merge consensus lines
    # =========================
    final_merged = merge_consensus_lines(
        all_consensus_lines,
        ref_seq
    )

    save_text(
        OUTPUT_DIR / "step3_merged.txt",
        final_merged
    )

    append_log("\n" + "=" * 70 + "\n")
    append_log("STEP 3: FINAL MERGED SEQUENCE\n")
    append_log("=" * 70 + "\n")
    append_log(final_merged + "\n")

    print("\nMerged:", final_merged)

    # =========================
    # STEP 4: Three-stage predictions
    # =========================
    stages = build_three_stage_inputs(
        final_merged,
        ref_seq
    )

    save_csv(
        OUTPUT_DIR / "step4_stage_inputs.csv",
        [
            {
                "stage": name,
                "sequence": seq
            }
            for name, seq in stages.items()
        ],
        ["stage", "sequence"]
    )

    all_stage_preds = []

    append_log("\n" + "=" * 70 + "\n")
    append_log("STEP 4: THREE-STAGE FINALIST PREDICTIONS\n")
    append_log("=" * 70 + "\n")

    for name, seq in stages.items():
        print("Stage:", name)

        preds = generate_predictions(model, seq)
        all_stage_preds.extend(preds)

        save_lines(
            OUTPUT_DIR / f"step4_stage_{name}_predictions.txt",
            preds
        )

        append_log("\n" + "-" * 70 + "\n")
        append_log(f"STAGE: {name}\n")
        append_log("-" * 70 + "\n")

        append_log("STAGE INPUT SEQUENCE:\n")
        append_log(seq + "\n\n")

        append_log(f"{N_SAMPLES} STAGE PREDICTIONS:\n")
        for j, p in enumerate(preds):
            append_log(f"{j + 1:02d}: {p}\n")

    save_lines(
        OUTPUT_DIR / "step4_all_60_stage_predictions.txt",
        all_stage_preds
    )

    # =========================
    # STEP 5: Final sequence by voting + memory validation
    # =========================
    final, final_report_rows = build_finalist_seq(
        final_merged=final_merged,
        ref_seq=ref_seq,
        all_stage_preds=all_stage_preds,
        strict_mutations=strict_mutations
    )

    save_text(
        OUTPUT_DIR / "step5_final.txt",
        final
    )

    save_csv(
        OUTPUT_DIR / "step5_final_vote_report.csv",
        final_report_rows,
        [
            "position_1based",
            "wt",
            "strict_memory",
            "vote_winner",
            "vote_count",
            "total_votes",
            "vote_frequency",
            "threshold",
            "final_residue",
            "decision",
            "all_counts"
        ]
    )

    muts = mutation_string(ref_seq, final)

    save_lines(
        OUTPUT_DIR / "step5_mutations.txt",
        muts
    )

    append_log("\n" + "=" * 70 + "\n")
    append_log("STEP 5: FINAL RESULT\n")
    append_log("=" * 70 + "\n")

    append_log("FINAL MERGED:\n")
    append_log(final_merged + "\n\n")

    append_log("FINAL SEQUENCE:\n")
    append_log(final + "\n\n")

    append_log("ACCEPTED MUTATIONS:\n")
    if muts:
        for m in muts:
            append_log(m + "\n")
    else:
        append_log("No accepted mutations.\n")

    print("\nFinal:", final)
    print("\nAccepted mutations:")
    if muts:
        for m in muts:
            print(m)
    else:
        print("No accepted mutations.")

    print(f"\nOutput directory: {OUTPUT_DIR.resolve()}")

    return final_merged, final


# =========================
# API WRAPPER
# =========================
def run_api_pipeline(ref_seq, token=None):
    model = create_model(token=token)
    return run_pipeline(model, ref_seq)


# =========================
# RUN / INTERACTIVE MODE
# =========================
if __name__ == "__main__":

    # When you run this file, it will ask for:
    # 1) ESM API key
    # 2) WT/reference amino-acid sequence
    #
    # No hard-coded REF_SEQ is needed.

    REF_SEQ = prompt_reference_sequence()
    merged, final = run_api_pipeline(REF_SEQ)
