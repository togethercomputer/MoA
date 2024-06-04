import re
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from alpaca_eval import constants, main, utils
from alpaca_eval.annotators import SinglePairwiseAnnotator


@pytest.fixture
def model_outputs():
    return [
        {
            "instruction": "1+1",
            "output": "3",
            "dataset": "test",
        },
        {
            "instruction": "2+2",
            "output": "5",
            "dataset": "test",
        },
        {
            "instruction": "2+3",
            "output": "6",
            "dataset": "test",
        },
    ]


@pytest.fixture
def reference_outputs():
    return [
        {
            "instruction": "1+1",
            "output": "2",
            "dataset": "test",
        },
        {
            "instruction": "2+2",
            "output": "4",
            "dataset": "test",
        },
        {
            "instruction": "2+3",
            "output": "5",
            "dataset": "test",
        },
    ]


@pytest.fixture
def expected_annotations():
    return [
        {
            "instruction": "1+1",
            "output_1": "2",
            "dataset": "test",
            "output_2": "3",
            "annotator": "test",
            "preference": 1,
        },
        {
            "instruction": "2+2",
            "output_1": "4",
            "dataset": "test",
            "output_2": "5",
            "annotator": "test",
            "preference": 1,
        },
        {
            "instruction": "2+3",
            "output_1": "5",
            "dataset": "test",
            "output_2": "6",
            "annotator": "test",
            "preference": 1,
        },
    ]


def _get_mock_annotate(preference=None):
    def annotate(df):
        df = df.copy()
        if preference is None:
            df["preference"] = [1] * len(df)
        else:
            df["preference"] = preference
        return df

    mock_function = MagicMock(side_effect=annotate)
    return mock_function


def clean_up():
    # remove the cache
    Path(constants.EVALUATORS_CONFIG_DIR / "test/annotations_seed0_configs").unlink(missing_ok=True)


def test_evaluate_print(model_outputs, reference_outputs, capsys, expected_annotations):
    mock_function = _get_mock_annotate()
    with patch.object(SinglePairwiseAnnotator, "__call__", mock_function):
        main.evaluate(model_outputs, reference_outputs, annotators_config="test", is_avoid_reannotations=False)

        # Capture the stdout
        captured = capsys.readouterr()
        printed_string = captured.out.strip()
        printed_string = re.sub(r"\s+", " ", printed_string)

        assert printed_string == "win_rate standard_error n_total avg_length Current model 0.00 0.00 3 1"

    clean_up()


def test_evaluate_basic(model_outputs, reference_outputs, expected_annotations):
    mock_function = _get_mock_annotate()
    name = "Current model"
    with patch.object(SinglePairwiseAnnotator, "__call__", mock_function):
        df_leaderboard, annotations = main.evaluate(
            model_outputs,
            reference_outputs,
            annotators_config="test",
            is_return_instead_of_print=True,
            is_avoid_reannotations=False,
        )
        assert annotations == expected_annotations
        assert isinstance(df_leaderboard, pd.DataFrame)
        # win rate is 1 because we always give preference to second which is model output
        assert df_leaderboard.loc[name, "win_rate"] == 0
        assert df_leaderboard.loc[name, "standard_error"] == 0
        assert df_leaderboard.loc[name, "n_total"] == 3
        assert df_leaderboard.loc[name, "mode"] == "community"

    clean_up()


def test_evaluate_advanced(model_outputs, reference_outputs, expected_annotations):
    mock_function = _get_mock_annotate()
    current_leaderboard_mode = "verified"
    name = "current"
    with patch.object(SinglePairwiseAnnotator, "__call__", mock_function):
        df_leaderboard, annotations = main.evaluate(
            model_outputs,
            reference_outputs,
            annotators_config="test",
            is_return_instead_of_print=True,
            is_avoid_reannotations=False,
            current_leaderboard_mode=current_leaderboard_mode,
            name=name,
            max_instances=2,
            precomputed_leaderboard=constants.ALPACAEVAL_LEADERBOARD_PATHS / "alpaca_eval_gpt4_leaderboard.csv",
        )
        assert annotations == expected_annotations[:2]
        assert isinstance(df_leaderboard, pd.DataFrame)
        # win rate is 0 because we always give preference to first which is reference
        assert df_leaderboard.loc[name, "win_rate"] == 0
        assert df_leaderboard.loc[name, "standard_error"] == 0
        assert df_leaderboard.loc[name, "n_total"] == 2
        assert df_leaderboard.loc[name, "mode"] == current_leaderboard_mode
        assert len(df_leaderboard) > 5  # entire leaderboard, 5 is arbitrary here

    clean_up()


def test_analyze_evaluators():
    # make a temporary config file called test
    configs = utils.load_configs(constants.EVALUATORS_CONFIG_DIR / "claude")
    evaluator_name = "tmp_test"
    tmp_path = constants.EVALUATORS_CONFIG_DIR / evaluator_name
    try:
        tmp_path.mkdir(parents=True, exist_ok=False)
        with open(tmp_path / "configs.yaml", "w") as f:
            yaml.dump(configs, f)

        # mock by always returning preference 1
        mock_function = _get_mock_annotate()
        with patch.object(SinglePairwiseAnnotator, "__call__", mock_function):
            df_leaderboard, all_crossannotations = main.analyze_evaluators(
                annotators_config=evaluator_name,
                is_return_instead_of_print=True,
                current_leaderboard_mode="community",
                is_single_annotator=True,
                max_instances=100,
            )
            assert evaluator_name in df_leaderboard.index
            assert df_leaderboard.loc[evaluator_name, "mode"] == "community"
            assert df_leaderboard.loc[evaluator_name, "Human agreement"] < 60
            assert isinstance(all_crossannotations, dict)
            assert isinstance(all_crossannotations[evaluator_name], pd.DataFrame)
            assert len(all_crossannotations[evaluator_name]) == 100
    finally:
        shutil.rmtree(tmp_path)
