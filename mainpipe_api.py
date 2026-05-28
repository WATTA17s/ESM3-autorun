
# =========================
# IMPORTS
# =========================
from pathlib import Path
from collections import Counter
from getpass import getpass

import esm
from esm.sdk.api import ESMProtein, GenerationConfig

MODEL_NAME = "esm3-medium-2024-08"

MASK_CHAR = "_"

WINDOW = 10
STEP = 10

N_SAMPLES = 20
TEMPERATURE = 0.5
NUM_STEPS = 8

FINALIST_THRESHOLD = 0.6

# =========================
# MODEL
# =========================
def create_model(model_name=MODEL_NAME, token=None):
    if token is None:
        token = getpass("Enter ESM API key: ")

    print("esm version:", esm.__version__)

    return esm.sdk.client(model_name, token=token)

# =========================
# OUTPUT
# =========================
OUTPUT_DIR = Path("pipeline_v1_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

MASTER_LOG = OUTPUT_DIR / "full_pipeline_log.txt"

# =========================
# FILE HELPERS
# =========================
def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)

def save_lines(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")

def append_log(text):
    with open(MASTER_LOG, "a") as f:
        f.write(text)

# =========================
# SEQ HELPERS
# =========================
def build_masked_sequences(ref_seq, window, step):
    seqs = []

    n = len(ref_seq)

    for start in range(0, n, step):
        end = min(start + window, n)

        seq = list(ref_seq)

        for i in range(start, end):
            seq[i] = MASK_CHAR

        seqs.append("".join(seq))

    return seqs

def masked_positions(seq):
    return [i for i, c in enumerate(seq) if c == MASK_CHAR]

# =========================
# GENERATE
# =========================
def generate_predictions(model, masked_seq):
    preds = []

    for i in range(N_SAMPLES):
        protein = ESMProtein(sequence=masked_seq)

        out = model.generate(
            protein,
            GenerationConfig(
                track="sequence",
                temperature=TEMPERATURE,
                num_steps=NUM_STEPS
            )
        )

        if not hasattr(out, "sequence"):
            raise RuntimeError(f"ESM generation failed at sample {i+1}: {out}")

        preds.append(out.sequence)

    return preds

# =========================
# STRICT CONSENSUS
# =========================
def consensus_line(
    masked_seq,
    preds,
    ref_seq,
    strict_mutations
):
    result = list(masked_seq)

    positions = masked_positions(masked_seq)

    for pos in positions:

        aa_list = [p[pos] for p in preds]

        count = Counter(aa_list)

        top_aa, top_count = count.most_common(1)[0]

        # strict 100%
        if top_count == len(preds):

            # mutation passed strict stage
            if top_aa != ref_seq[pos]:

                result[pos] = MASK_CHAR

                # save strict mutation memory
                strict_mutations[pos] = top_aa

            # identical to WT
            else:
                result[pos] = ref_seq[pos]

        else:
            result[pos] = ref_seq[pos]

    return "".join(result)

# =========================
# MERGE
# =========================
def merge_consensus_lines(lines, ref_seq):

    merged = []

    for i in range(len(ref_seq)):

        if any(line[i] == MASK_CHAR for line in lines):
            merged.append(MASK_CHAR)

        else:
            merged.append(ref_seq[i])

    return "".join(merged)

# =========================
# 3-STAGE INPUTS
# =========================
def build_three_stage_inputs(final_merged, ref_seq):

    n = len(final_merged)

    mid = n // 2

    return {
        "full": final_merged,

        "front_half":
            final_merged[:mid] + ref_seq[mid:],

        "back_half":
            ref_seq[:mid] + final_merged[mid:]
    }

# =========================
# FINALIST
# =========================
def build_finalist_seq(
    final_merged,
    ref_seq,
    all_preds,
    strict_mutations
):

    result = list(ref_seq)

    append_log("\n" + "="*70 + "\n")
    append_log("FINAL DECISION\n")
    append_log("="*70 + "\n")

    for pos, ch in enumerate(final_merged):

        # only positions that passed strict stage
        if ch == MASK_CHAR:

            aa_list = [seq[pos] for seq in all_preds]

            count = Counter(aa_list)

            top_aa, top_count = count.most_common(1)[0]

            frac = top_count / len(all_preds)

            strict_aa = strict_mutations[pos]

            append_log(
                f"\nPos {pos+1} | "
                f"strict={strict_aa} | "
                f"final={top_aa} | "
                f"freq={frac:.2f}\n"
            )

            # FINAL RULE:
            # must pass threshold
            # AND match strict memory

            if (
                frac >= FINALIST_THRESHOLD
                and top_aa == strict_aa
            ):

                result[pos] = top_aa

                append_log("→ ACCEPT\n")

            else:

                result[pos] = ref_seq[pos]

                append_log("→ REVERT TO WT\n")

    return "".join(result)

# =========================
# MAIN PIPELINE
# =========================
def run_pipeline(model, ref_seq):

    save_text(MASTER_LOG, "")

    append_log("===== PIPELINE V1 START =====\n\n")

    append_log("REFERENCE SEQUENCE:\n")
    append_log(ref_seq + "\n\n")

    # =========================
    # STEP 1
    # =========================
    masked_seqs = build_masked_sequences(
        ref_seq,
        WINDOW,
        STEP
    )

    save_lines(
        OUTPUT_DIR / "step1_masked.txt",
        masked_seqs
    )

    append_log("="*70 + "\n")
    append_log("STEP 1: MASKED SEQUENCES\n")
    append_log("="*70 + "\n")

    for i, mseq in enumerate(masked_seqs):

        append_log(f"\nMASK {i+1:03d}\n")

        append_log(mseq + "\n")

    all_lines = []

    # strict mutation memory
    strict_mutations = {}

    # =========================
    # STEP 2
    # =========================
    append_log("\n" + "="*70 + "\n")
    append_log("STEP 2: MASK PREDICTIONS + CONSENSUS\n")
    append_log("="*70 + "\n")

    for i, mseq in enumerate(masked_seqs):

        start = i * STEP

        end = min(start + WINDOW, len(ref_seq))

        print(
            f"[Mask {i+1}/{len(masked_seqs)}] "
            f"positions {start+1}-{end}"
        )

        preds = generate_predictions(model, mseq)

        line = consensus_line(
            mseq,
            preds,
            ref_seq,
            strict_mutations
        )

        append_log("\n" + "-"*70 + "\n")

        append_log(
            f"MASK {i+1:03d} | "
            f"positions {start+1}-{end}\n"
        )

        append_log("-"*70 + "\n")

        append_log("INPUT MASKED SEQUENCE:\n")

        append_log(mseq + "\n\n")

        append_log("20 PREDICTIONS:\n")

        for j, seq in enumerate(preds):

            append_log(f"{j+1:02d}: {seq}\n")

        append_log("\nCONSENSUS LINE:\n")

        append_log(line + "\n")

        all_lines.append(line)

    # =========================
    # STRICT MEMORY LOG
    # =========================
    append_log("\n" + "="*70 + "\n")
    append_log("STRICT MUTATION MEMORY\n")
    append_log("="*70 + "\n")

    for pos, aa in strict_mutations.items():

        append_log(
            f"Position {pos+1}: {ref_seq[pos]} -> {aa}\n"
        )

    # =========================
    # STEP 3
    # =========================
    final_merged = merge_consensus_lines(
        all_lines,
        ref_seq
    )

    save_text(
        OUTPUT_DIR / "step3_merged.txt",
        final_merged
    )

    append_log("\n" + "="*70 + "\n")
    append_log("STEP 3: FINAL MERGED SEQUENCE\n")
    append_log("="*70 + "\n")

    append_log(final_merged + "\n")

    print("\nMerged:", final_merged)

    # =========================
    # STEP 4
    # =========================
    stages = build_three_stage_inputs(
        final_merged,
        ref_seq
    )

    all_stage_preds = []

    append_log("\n" + "="*70 + "\n")
    append_log("STEP 4: STAGE PREDICTIONS\n")
    append_log("="*70 + "\n")

    for name, seq in stages.items():

        print("Stage:", name)

        preds = generate_predictions(model, seq)

        all_stage_preds.extend(preds)

        append_log("\n" + "-"*70 + "\n")

        append_log(f"STAGE: {name}\n")

        append_log("-"*70 + "\n")

        append_log("STAGE INPUT SEQUENCE:\n")

        append_log(seq + "\n\n")

        append_log("20 STAGE PREDICTIONS:\n")

        for j, p in enumerate(preds):

            append_log(f"{j+1:02d}: {p}\n")

    # =========================
    # STEP 5
    # =========================
    final = build_finalist_seq(
        final_merged,
        ref_seq,
        all_stage_preds,
        strict_mutations
    )

    save_text(
        OUTPUT_DIR / "step5_final.txt",
        final
    )

    append_log("\n" + "="*70 + "\n")
    append_log("STEP 5: FINAL RESULT\n")
    append_log("="*70 + "\n")

    append_log("FINAL MERGED:\n")
    append_log(final_merged + "\n\n")

    append_log("FINAL SEQUENCE:\n")
    append_log(final + "\n")

    print("\nFinal:", final)

    return final_merged, final

# =========================
# API WRAPPER
# =========================
def run_api_pipeline(ref_seq, token=None):

    model = create_model(token=token)

    return run_pipeline(model, ref_seq)

# =========================
# RUN
# =========================
if __name__ == "__main__":

    merged, final = run_api_pipeline(
        "MVLSEGEWQLVLHVWAKVE..."
    )

