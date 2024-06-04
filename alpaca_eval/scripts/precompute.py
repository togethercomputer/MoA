import logging
from typing import Optional

import fire
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from alpaca_eval import analyze, annotators, constants
from alpaca_eval import main as alpaca_main
from alpaca_eval import utils
from alpaca_eval.types import AnyPath


def precompute_on_all_human_leaderboard(
    annotators_config="gpt4",
    Annotator=annotators.PairwiseAnnotator,
    all_data=constants.ALPACAFARM_GOLD_ANNOTATIONS,
    analyzer_kwargs=None,
    **annotator_kwargs,
):
    """Precompute all instructions on the eval leaderboard that has been annotated by humans."""
    analyzer_kwargs = analyzer_kwargs or {}
    analyzer = analyze.Analyzer(gold_annotations=all_data, **analyzer_kwargs)
    df_annotations = analyze.get_annotations(
        analyzer, Annotator=Annotator, annotators_config=annotators_config, **annotator_kwargs
    )


def precompute_evaluator_leaderboard(
    annotators_configs_to_analyze="MINIMAL_EVALUATORS",
    annotators_configs_to_benchmark="VERIFIED_EVALUATORS",
    max_instances=None,
    **kwargs,
):
    """Precompute evaluator's leaderboard for important API models."""
    if isinstance(annotators_configs_to_analyze, str):
        annotators_configs_to_analyze = getattr(constants, annotators_configs_to_analyze)

    if isinstance(annotators_configs_to_benchmark, str):
        annotators_configs_to_benchmark = getattr(constants, annotators_configs_to_benchmark)

    for annotators_config in annotators_configs_to_analyze:
        # saving is done automatically
        _ = alpaca_main.analyze_evaluators(
            annotators_config=annotators_config,
            max_instances=max_instances,
            is_save_leaderboard=max_instances is None,
            is_return_instead_of_print=True,  # don't print
            current_leaderboard_mode="minimal",
            **kwargs,
        )

    for annotators_config in annotators_configs_to_benchmark:
        # saving is done automatically
        _ = alpaca_main.analyze_evaluators(
            annotators_config=annotators_config,
            max_instances=max_instances,
            is_save_leaderboard=max_instances is None,
            is_return_instead_of_print=True,  # don't print
            is_single_annotator=True,
            current_leaderboard_mode="verified",
            **kwargs,
        )


def update_leaderboard(leaderboard_path, model_outputs="results/{model_name}/model_outputs.json", **kwargs):
    """Rerun evaluate on each model in the leaderboard. Useful to add a column suc as avg_length."""
    df_leaderboard = utils.load_or_convert_to_dataframe(leaderboard_path)
    for model_name in df_leaderboard.index:
        alpaca_main.evaluate(model_outputs=model_outputs.format(model_name=model_name), **kwargs)


def compare_leaderboards(leaderboard_path_1, leaderboard_path_2):
    df_lb_1 = utils.load_or_convert_to_dataframe(leaderboard_path_1)
    df_lb_2 = utils.load_or_convert_to_dataframe(leaderboard_path_2)

    # keep only intersection of models and in the same order
    intersected_models = df_lb_1.index.intersection(df_lb_2.index)
    df_lb_1 = df_lb_1.loc[intersected_models]
    df_lb_2 = df_lb_2.loc[intersected_models]

    metrics = {}
    metrics["Spearman corr."] = spearmanr(df_lb_1["win_rate"], df_lb_2["win_rate"]).statistic
    metrics["Pearson corr."] = pearsonr(df_lb_1["avg_length"], df_lb_2["avg_length"]).statistic

    print(pd.Series(metrics).to_string(float_format="%.2f"))


def make_leaderboard_like(leaderboard_to_copy: Optional[AnyPath], **kwargs):
    """Make a leaderboard on all the models that have been evaluated in another leaderboard."""
    df_lb_old = pd.read_csv(leaderboard_to_copy, index_col=0)

    kwargs["is_cache_leaderboard"] = True
    kwargs["is_return_instead_of_print"] = True
    for m, r in df_lb_old.iterrows():
        kwargs["current_leaderboard_mode"] = r["mode"]
        leaderboard_new, _ = alpaca_main.evaluate(model_outputs=f"results/{m}/model_outputs.json", **kwargs)

    print("Comparison between the leaderboards:")
    compare_leaderboards(leaderboard_to_copy, leaderboard_new)


def run_all_length_corrected_winrates(**kwargs):
    # load the leaderboard for AlpacaEval
    for is_alpaca_eval_2 in [True, False]:
        for annotator in [
            "mistral-large-2402_ranking",
            "claude_3_opus_ranking",
            constants.ANNOTATOR_CONFIG_AE2,
            constants.ANNOTATOR_CONFIG_AE1,
        ]:
            if is_alpaca_eval_2:
                lb_path = constants.ALPACAEVAL_2_LEADERBOARD_PATHS / f"{annotator}_leaderboard.csv"
                ref_outputs = constants.ALPACAEVAL_REFERENCE_OUTPUTS_2
                if not lb_path.exists():
                    break

            else:
                if annotator != constants.ANNOTATOR_CONFIG_AE1:
                    break
                lb_path = constants.ALPACAEVAL_1_LEADERBOARD_PATHS / f"{annotator}_leaderboard.csv"
                ref_outputs = constants.ALPACAEVAL_REFERENCE_OUTPUTS_1

            lb = utils.load_or_convert_to_dataframe(lb_path)
            for m in lb.index:
                try:
                    alpaca_main.evaluate(
                        model_outputs=f"results/{m}/model_outputs.json",
                        reference_outputs=ref_outputs,
                        annotators_config=annotator,
                        is_recompute_metrics_only=True,
                        name=m,
                        **kwargs,
                    )
                except:
                    logging.exception(f"Error while computing the length corrected winrate for {m}")


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
