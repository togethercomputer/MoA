from unittest.mock import MagicMock

import pandas as pd
import pytest

from alpaca_eval.annotators import PairwiseAnnotator, SinglePairwiseAnnotator


@pytest.fixture
def df_to_annotate():
    return pd.DataFrame(
        {
            "instruction": ["2+2", "1+1", "2+3"],
            "output_1": ["4", "2", "5"],
            "output_2": ["5", "1", "4"],
        }
    )


@pytest.fixture
def expected_annotations():
    return [
        {
            "instruction": "2+2",
            "output_1": "4",
            "output_2": "5",
            "annotator": "test",
            "preference": 1,
        },
        {
            "instruction": "1+1",
            "output_1": "2",
            "output_2": "1",
            "annotator": "test",
            "preference": 2,
        },
        {
            "instruction": "2+3",
            "output_1": "5",
            "output_2": "4",
            "annotator": "test",
            "preference": 1,
        },
    ]


@pytest.fixture
def single_annotator():
    return SinglePairwiseAnnotator(
        prompt_template="text_davinci_003/basic_prompt.txt",
        completion_parser_kwargs=dict(outputs_to_match={1: r"(?:^|\n) ?Output \(a\)", 2: "(?:^|\n) ?Output \(b\)"}),
        is_randomize_output_order=False,
        is_shuffle=False,
        is_store_raw_completions=False,
    )


def test_single_annotator(single_annotator, df_to_annotate):
    # Create a sample DataFrame for testing
    parsable_completions = ["Output (a)", "Output (b)"]
    completions = parsable_completions + ["not parsable"]  # add an example that can't be parsed
    single_annotator.fn_completions = MagicMock(return_value={"completions": completions})
    # set a completion_column => store it
    single_annotator.completion_column = "completions"

    # Call the preprocess method
    df_annotated = single_annotator(df_to_annotate)

    assert df_annotated["preference"].tolist() == [1, 2]
    assert df_annotated["instruction"].tolist() == ["2+2", "1+1"]
    assert set(df_annotated.columns.tolist()) == {
        "instruction",
        "output_1",
        "output_2",
        "preference",
        "completions",
    }
    # check that you also save the completions.
    assert df_annotated["completions"].tolist() == parsable_completions


@pytest.fixture
def pairwise_annotator(tmp_path):
    return PairwiseAnnotator(
        annotators_config="test",
        caching_path=tmp_path / "cache_{seed}.json",
    )


def _get_mock_annotator(annotator, preference=None):
    def annotate(df):
        df = df.copy()
        if preference is None:
            df["preference"] = [1] * len(df)
        else:
            df["preference"] = preference
        return df

    mock_function = MagicMock(side_effect=annotate)
    for a in annotator.annotators.keys():
        annotator.annotators[a] = mock_function

    return annotator, mock_function


def test_annotate_pairs(pairwise_annotator, df_to_annotate, expected_annotations):
    pairwise_annotator, mock_function = _get_mock_annotator(pairwise_annotator, preference=[1, 2, 1])

    # Call the annotate_pairs method and assert output
    annotated = pairwise_annotator.annotate_pairs(df_to_annotate)
    assert annotated == expected_annotations
    args, kwargs = mock_function.call_args
    assert len(args[0]) == 3

    # now try again with caching => should not be annotating anything
    annotated = pairwise_annotator.annotate_pairs(df_to_annotate)
    assert annotated == expected_annotations
    args, kwargs = mock_function.call_args
    assert len(args[0]) == 0

    # now try again without caching => should be annotating again
    pairwise_annotator.is_avoid_reannotations = False
    annotated = pairwise_annotator.annotate_pairs(df_to_annotate)
    assert annotated == expected_annotations
    args, kwargs = mock_function.call_args
    assert len(args[0]) == 3


def test_annotate_samples(pairwise_annotator):
    # this has a list of outputs and you will samples pairs from it
    potential_outputs = ["5", "1", "4"]
    outputs_to_annotate = [
        {
            "instruction": "2+2",
            "output": potential_outputs,
        }
    ]

    pairwise_annotator, mock_function = _get_mock_annotator(pairwise_annotator)

    def run_all_tests(annotated_list_of_dict):
        # run tests for each row => use dataframe
        assert isinstance(annotated_list_of_dict, list)
        assert len(annotated_list_of_dict[0]), dict
        df_annotated = pd.DataFrame(annotated_list_of_dict)
        assert (df_annotated["instruction"] == "2+2").all()
        assert df_annotated["output_1"].isin(potential_outputs).all()
        assert df_annotated["output_2"].isin(potential_outputs).all()
        assert (df_annotated["output_1"] != df_annotated["output_2"]).all()
        assert (df_annotated["preference"] == 1).all()

    # let's sample one pair from the list of outputs (is_unique_instructions=True)
    annotated = pairwise_annotator.annotate_samples(outputs_to_annotate, is_unique_instructions=True)
    assert len(annotated) == 1
    run_all_tests(annotated)

    # same but one pair per output (is_unique_instructions=False)
    annotated = pairwise_annotator.annotate_samples(outputs_to_annotate, is_unique_instructions=False)
    assert len(annotated) == 3
    run_all_tests(annotated)


def test_annotate_head2head(pairwise_annotator, df_to_annotate, expected_annotations):
    pairwise_annotator, mock_function = _get_mock_annotator(pairwise_annotator, preference=[1, 2, 1])

    df_1 = df_to_annotate.drop(columns="output_2").rename(columns={"output_1": "output"})
    df_2 = df_to_annotate.drop(columns="output_1").rename(columns={"output_2": "output"})
    annotated = pairwise_annotator.annotate_head2head(df_1, df_2, is_ordered=True)

    assert annotated == expected_annotations
    args, kwargs = mock_function.call_args
    assert len(args[0]) == 3

    # let's change the order of one dataframe and see if results are the same
    df_2 = df_2.iloc[::-1]
    annotated = pairwise_annotator.annotate_head2head(df_1, df_2, is_ordered=False)

    assert annotated == expected_annotations
