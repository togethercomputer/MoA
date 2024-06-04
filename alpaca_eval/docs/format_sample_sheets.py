from pathlib import Path

import pandas as pd

F_OUTPUTS = "model_outputs.json"
F_ANNOTATIONS = "annotations.json"
CURRENT_DIR = Path(__file__).parents[1]
RESULTS_DIR = CURRENT_DIR / "results"


df_references = {
    r: pd.read_json(RESULTS_DIR / r / F_OUTPUTS, orient="records")
    for r in ["text_davinci_003", "gpt4_1106_preview", "gpt4"]
}
df_reference = df_references["text_davinci_003"]


min_output_columns = {"instruction", "output", "generator"}
min_annotation_columns = {
    "instruction",
    "annotator",
    "preference",
    "generator_1",
    "generator_2",
    "output_1",
    "output_2",
}

# Create a dict mapping each instruction in df_reference to its index => will keep that order for the other files
order = {tuple(pair): i for i, pair in enumerate(zip(df_reference["dataset"], df_reference["instruction"]))}
max_missing_examples = 5

for f in RESULTS_DIR.glob(f"*/*/{F_OUTPUTS}"):
    df = pd.read_json(f, orient="records")
    n_diff = len(df_reference) - len(df)
    if n_diff - max_missing_examples:
        raise ValueError(f"There are more than 5 examples missing in {f}. {len(df_reference)}!={len(df)}.")

    if (df["output"].str.len() == 0).any():
        raise ValueError(f"Empty output in {f}.")

    # Sort the df using the reference df
    df["order"] = df.apply(lambda row: order[(row["dataset"], row["instruction"])], axis=1)
    df = df.sort_values("order").drop("order", axis=1)

    df.to_json(f, orient="records", indent=2)

    missing_columns = min_output_columns - set(df.columns)
    if len(missing_columns) > 0:
        raise ValueError(f"Missing columns in {f}: {missing_columns}.")

for f in RESULTS_DIR.glob(f"*/*/{F_ANNOTATIONS}"):
    df = pd.read_json(f, orient="records")
    n_diff = len(df_reference) - len(df)
    if n_diff - max_missing_examples > 0:
        raise ValueError(f"There are more than 5 examples missing in {f}. {len(df_reference)}!={len(df)}.")

    # can't sort because you don't have the dataset
    # df["order"] = df.apply(lambda row: order[(row["dataset"], row["instruction"])], axis=1)
    # df = df.sort_values("order").drop("order", axis=1)

    df.to_json(f, orient="records", indent=2)

    missing_columns = min_annotation_columns - set(df.columns)
    if len(missing_columns) > 0:
        raise ValueError(f"Missing columns in {f}: {missing_columns}.")

    # make sure that the model was always compared to the right output
    for baseline in df["generator_1"].unique():
        if baseline == "text_davinci_003":
            continue  # for historical reasons the reference text_davinci_003 is diff than the model output
        df_subset = df.query(f"generator_1 == '{baseline}'").sort_values("instruction")
        df_baseline = df_references[baseline].sort_values("instruction")
        # now make sure that df_subset["output_1"] is the same as df_baseline["output"] when you sort by instruction
        if n_diff == 0:  # hard to check otherwise
            if not df_baseline["output"].equals(df_subset["output_1"]):
                raise ValueError(f"Output_1 in {f} is not the same as the reference file {baseline}.")
