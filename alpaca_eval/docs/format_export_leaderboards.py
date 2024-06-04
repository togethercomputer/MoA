import logging
from pathlib import Path

from alpaca_eval.constants import MODELS_CONFIG_DIR, PRECOMPUTED_LEADERBOARDS
from alpaca_eval.metrics.glm_winrate import get_is_extreme_changes
from alpaca_eval.utils import load_configs, load_or_convert_to_dataframe

CURRENT_DIR = Path(__file__).parents[1]
RESULTS_DIR = CURRENT_DIR / "results"

for leaderboard_file in PRECOMPUTED_LEADERBOARDS.values():
    df = load_or_convert_to_dataframe(leaderboard_file)
    df["link"] = ""
    df["samples"] = ""
    cols_to_keep = ["win_rate", "avg_length", "link", "samples", "mode"]
    if "length_controlled_winrate" in df.columns:
        cols_to_keep = ["length_controlled_winrate"] + cols_to_keep
    df = df[cols_to_keep]

    # drop mode == 'dev'
    df = df[df["mode"] != "dev"]

    df = df.rename(columns={"mode": "filter"})
    df = df.reset_index(names="name")
    for idx in range(len(df)):
        informal_name = df.loc[idx, "name"]
        try:
            model_config = load_configs(df.loc[idx, "name"], relative_to=MODELS_CONFIG_DIR)[informal_name]
        except KeyError as e:
            logging.exception(
                f"Could not find model config for {informal_name}. This is likely because the name of "
                f"the annotator does not match the name of the model's directory."
            )
            raise e

        if "pretty_name" in model_config:
            df.loc[idx, "name"] = model_config["pretty_name"]

        if "link" in model_config:
            df.loc[idx, "link"] = model_config["link"]

        file_outputs = RESULTS_DIR / informal_name / "model_outputs.json"
        if file_outputs.is_file():
            df.loc[
                idx, "samples"
            ] = f"https://github.com/tatsu-lab/alpaca_eval/blob/main/results/{informal_name}/model_outputs.json"

    # if "length_controlled_winrate" never nan then we can use it as the main metric
    if "length_controlled_winrate" in cols_to_keep and df["length_controlled_winrate"].notna().all():
        df = df.sort_values(by=["length_controlled_winrate"], ascending=False)
    else:
        df = df.sort_values(by=["win_rate"], ascending=False)

    # run get_is_extreme_changes on each row where length_controlled_winrate is not nan to avoid merging PRs
    # where the length controlled results seem very suspicious
    if "length_controlled_winrate" in cols_to_keep:
        idx_notna = df["length_controlled_winrate"].notna()
        arr_is_extreme = df[idx_notna].apply(
            lambda row: get_is_extreme_changes(row["win_rate"], row["length_controlled_winrate"], min_warn=False),
            axis=1,
        )
        if arr_is_extreme.any():
            raise ValueError(
                f"Found extreme changes in the length controlled winrate. Please check the following rows: "
                f"{df[idx_notna][arr_is_extreme][['name', 'win_rate','length_controlled_winrate']]}"
            )

    save_dir = Path("docs") / leaderboard_file.parent.name
    save_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(save_dir / leaderboard_file.name, index=False)
