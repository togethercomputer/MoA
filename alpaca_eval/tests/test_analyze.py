import pandas as pd
import pytest

from alpaca_eval import analyze
from alpaca_eval.analyze import SCORING_RULES


@pytest.fixture
def analyzer():
    return analyze.Analyzer(
        n_annotators=4, gold_crossannotations=RECORDS, gold_annotations=None, scoring_rule="zero_one"
    )


def test_agreement_of_annotations(analyzer):
    df_crossannotations = analyzer.df_gold_crossannotations.head(8).copy()
    # index is 0,1,2,3,0,1,2,3
    df_crossannotations["preference"] = [1] * 4 + [2, 2, 2, 1]
    agreement = analyzer.agreement_of_annotations(
        df_crossannotations,
        annotations_2=None,
        n_majority_vote_1=1,
        n_majority_vote_2=1,
    )
    # for first example is always correct => 1. for second example, agreement is 2/3, 2/3, 2/3, 0 => 6/12
    # in total mean(1,1/2) = 0.75
    assert agreement["score"] == pytest.approx(0.75)
    assert agreement["sem_samples"] == pytest.approx(0.25)
    assert agreement["counts"] == pytest.approx(2)
    assert agreement["sem_annotators"] == pytest.approx(0.07537783614444091)

    # now with 3 annotators
    kwargs_3_annotators = dict(
        annotations_1=df_crossannotations,
        annotations_2=None,
        n_majority_vote_1=1,
        n_majority_vote_2=3,
    )
    agreement = analyzer.agreement_of_annotations(**kwargs_3_annotators)
    # for first example is always correct => 1. for second example, agreement is 1, 1, 1, 0 => 3/4
    # in total mean(1,3/4) = 0.875. Where the 1 come from the mode
    assert agreement["score"] == pytest.approx(0.875)
    assert agreement["sem_samples"] == pytest.approx(0.125)
    assert agreement["counts"] == pytest.approx(2)
    assert agreement["sem_annotators"] == pytest.approx(0.125)

    # now with scoring rule "absolute", which should give the same result as predictions are discrete
    analyzer.scoring_rule = SCORING_RULES["absolute"]()
    agreement = analyzer.agreement_of_annotations(**kwargs_3_annotators)
    assert agreement["score"] == pytest.approx(0.875)

    # now change a little the preferences
    df_crossannotations["preference"] = [1] * 4 + [2, 2, 2, 1.5]
    agreement = analyzer.agreement_of_annotations(**kwargs_3_annotators)
    # for first example is always correct => 1. for second example, agreement is 1,1,1,0.5 => 0.875
    # in total mean(1,0.875) = 0.9375.
    assert agreement["score"] == pytest.approx(0.9375)

    df_crossannotations["preference"] = [1] * 4 + [1.9, 1.8, 1.7, 1.6]
    agreement = analyzer.agreement_of_annotations(**kwargs_3_annotators)
    # for first example is always correct => 1. for second example, agreement is 0.8,0.9,0.9,0.8 => 0.85
    # in total mean(1,0.85) = 0.925.
    assert agreement["score"] == pytest.approx(0.925)


def test_get_length_biases(analyzer):
    # Create a sample DataFrame for testing
    df = pd.DataFrame(
        {
            "instruction": ["A", "B", "C"],
            "output_1": ["long output", "short", "much longer output" * 30],
            "output_2": ["short", "somewhat short", "shorter output"],
            "preference": [2, 1, 1],
        }
    )

    # Test the get_length_biases method
    result = analyzer.get_length_biases(df)
    assert result["probability_prefer_longer"] == pytest.approx(1.0)
    assert result["percentage_longer"] == pytest.approx(12.127705627705629)


def test_get_list_biases(analyzer):
    # Create a sample DataFrame for testing
    df = pd.DataFrame(
        {
            "instruction": ["A", "B", "C"],
            "output_1": ["apple", "- apple\n - banana", "apple"],
            "output_2": ["1. apple\n 2. banana", "apple", "a. apple\n b. banana"],
            "preference": [1, 1, 2],
        }
    )

    # Test the get_list_biases method
    result = analyzer.get_list_biases(df)
    assert result["probability_prefer_list"] == pytest.approx(0.6666666667)


RECORDS = [
    {
        "instruction": "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the "
        "sentence and make your writing clearer by keeping it concise. Whenever possible, "
        "break complex sentences into multiple sentences and eliminate unnecessary words.\n\nIf "
        "you have any questions about my rate or if you find it necessary to increase or "
        "decrease the scope for this project, please let me know.",
        "output_1": "If you have questions about my rate or need to modify the scope of this project, "
        "please let me know.",
        "output_2": "If you have any questions about my rate or need to adjust the scope for this project, "
        "please let me know.",
        "preference": 1,
        "annotator_index": 15,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 0,
        "n_annotated": 4,
    },
    {
        "instruction": "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the "
        "sentence and make your writing clearer by keeping it concise. Whenever possible, "
        "break complex sentences into multiple sentences and eliminate unnecessary words.\n\nIf "
        "you have any questions about my rate or if you find it necessary to increase or "
        "decrease the scope for this project, please let me know.",
        "output_1": "If you have questions about my rate or need to modify the scope of this project, "
        "please let me know.",
        "output_2": "If you have any questions about my rate or need to adjust the scope for this project, "
        "please let me know.",
        "preference": 1,
        "annotator_index": 0,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 1,
        "n_annotated": 4,
    },
    {
        "instruction": "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the "
        "sentence and make your writing clearer by keeping it concise. Whenever possible, "
        "break complex sentences into multiple sentences and eliminate unnecessary words.\n\nIf "
        "you have any questions about my rate or if you find it necessary to increase or "
        "decrease the scope for this project, please let me know.",
        "output_1": "If you have questions about my rate or need to modify the scope of this project, "
        "please let me know.",
        "output_2": "If you have any questions about my rate or need to adjust the scope for this project, "
        "please let me know.",
        "preference": 1,
        "annotator_index": 9,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 2,
        "n_annotated": 4,
    },
    {
        "instruction": "The sentence you are given might be too wordy, complicated, or unclear. Rewrite the "
        "sentence and make your writing clearer by keeping it concise. Whenever possible, "
        "break complex sentences into multiple sentences and eliminate unnecessary words.\n\nIf "
        "you have any questions about my rate or if you find it necessary to increase or "
        "decrease the scope for this project, please let me know.",
        "output_1": "If you have questions about my rate or need to modify the scope of this project, "
        "please let me know.",
        "output_2": "If you have any questions about my rate or need to adjust the scope for this project, "
        "please let me know.",
        "preference": 2,
        "annotator_index": 7,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 3,
        "n_annotated": 4,
    },
    {
        "instruction": "Analyze the word choice, phrasing, punctuation, and capitalization in the given email. "
        "How may the writer of this email sound to the reader? These tones include "
        "Disheartening, Accusatory, Worried, Curious, Surprised, Disapproving, Unassuming, "
        "Formal, Assertive, Confident, Appreciative, Concerned, Sad, Informal, Regretful, "
        "Encouraging, Egocentric, Joyful, Optimistic, and Excited.\n\nHi Jen, \nI hope you're "
        "well. Can we catch up today? I'd appreciate your input on my presentation for "
        "tomorrow's meeting. I'd especially love it if you could double-check the sales numbers "
        "with me. There's a coffee in it for you!",
        "output_1": "The writer of this email likely sounds Appreciative, Encouraging, and Optimistic.",
        "output_2": "The tone of the email is mostly informal, with a hint of laughter and enthusiasm.",
        "preference": 1,
        "annotator_index": 10,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 0,
        "n_annotated": 4,
    },
    {
        "instruction": "Analyze the word choice, phrasing, punctuation, and capitalization in the given email. "
        "How may the writer of this email sound to the reader? These tones include "
        "Disheartening, Accusatory, Worried, Curious, Surprised, Disapproving, Unassuming, "
        "Formal, Assertive, Confident, Appreciative, Concerned, Sad, Informal, Regretful, "
        "Encouraging, Egocentric, Joyful, Optimistic, and Excited.\n\nHi Jen, \nI hope you're "
        "well. Can we catch up today? I'd appreciate your input on my presentation for "
        "tomorrow's meeting. I'd especially love it if you could double-check the sales numbers "
        "with me. There's a coffee in it for you!",
        "output_1": "The writer of this email likely sounds Appreciative, Encouraging, and Optimistic.",
        "output_2": "The tone of the email is mostly informal, with a hint of laughter and enthusiasm.",
        "preference": 2,
        "annotator_index": 15,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 1,
        "n_annotated": 4,
    },
    {
        "instruction": "Analyze the word choice, phrasing, punctuation, and capitalization in the given email. "
        "How may the writer of this email sound to the reader? These tones include "
        "Disheartening, Accusatory, Worried, Curious, Surprised, Disapproving, Unassuming, "
        "Formal, Assertive, Confident, Appreciative, Concerned, Sad, Informal, Regretful, "
        "Encouraging, Egocentric, Joyful, Optimistic, and Excited.\n\nHi Jen, \nI hope you're "
        "well. Can we catch up today? I'd appreciate your input on my presentation for "
        "tomorrow's meeting. I'd especially love it if you could double-check the sales numbers "
        "with me. There's a coffee in it for you!",
        "output_1": "The writer of this email likely sounds Appreciative, Encouraging, and Optimistic.",
        "output_2": "The tone of the email is mostly informal, with a hint of laughter and enthusiasm.",
        "preference": 1,
        "annotator_index": 0,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 2,
        "n_annotated": 4,
    },
    {
        "instruction": "Analyze the word choice, phrasing, punctuation, and capitalization in the given email. "
        "How may the writer of this email sound to the reader? These tones include "
        "Disheartening, Accusatory, Worried, Curious, Surprised, Disapproving, Unassuming, "
        "Formal, Assertive, Confident, Appreciative, Concerned, Sad, Informal, Regretful, "
        "Encouraging, Egocentric, Joyful, Optimistic, and Excited.\n\nHi Jen, \nI hope you're "
        "well. Can we catch up today? I'd appreciate your input on my presentation for "
        "tomorrow's meeting. I'd especially love it if you could double-check the sales numbers "
        "with me. There's a coffee in it for you!",
        "output_1": "The writer of this email likely sounds Appreciative, Encouraging, and Optimistic.",
        "output_2": "The tone of the email is mostly informal, with a hint of laughter and enthusiasm.",
        "preference": 1,
        "annotator_index": 4,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 3,
        "n_annotated": 4,
    },
    {
        "instruction": "Rewrite the given text and correct grammar, spelling, and punctuation errors.\n\nIf "
        "you'd told me year ago that today I would finish a marathon, I would of laughed. Your "
        "support had a huge affect on me!",
        "output_1": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge effect on me!",
        "output_2": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge affect on me!",
        "preference": 1,
        "annotator_index": 7,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 0,
        "n_annotated": 4,
    },
    {
        "instruction": "Rewrite the given text and correct grammar, spelling, and punctuation errors.\n\nIf "
        "you'd told me year ago that today I would finish a marathon, I would of laughed. Your "
        "support had a huge affect on me!",
        "output_1": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge effect on me!",
        "output_2": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge affect on me!",
        "preference": 1,
        "annotator_index": 8,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 1,
        "n_annotated": 4,
    },
    {
        "instruction": "Rewrite the given text and correct grammar, spelling, and punctuation errors.\n\nIf "
        "you'd told me year ago that today I would finish a marathon, I would of laughed. Your "
        "support had a huge affect on me!",
        "output_1": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge effect on me!",
        "output_2": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge affect on me!",
        "preference": 1,
        "annotator_index": 11,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 2,
        "n_annotated": 4,
    },
    {
        "instruction": "Rewrite the given text and correct grammar, spelling, and punctuation errors.\n\nIf "
        "you'd told me year ago that today I would finish a marathon, I would of laughed. Your "
        "support had a huge affect on me!",
        "output_1": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge effect on me!",
        "output_2": "If you had told me a year ago that today I would finish a marathon, I would have laughed. "
        "Your support had a huge affect on me!",
        "preference": 2,
        "annotator_index": 1,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 3,
        "n_annotated": 4,
    },
    {
        "instruction": "You are given a paper citation, convert it to the requested citation "
        "style.\n\nChicago: Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, "
        'Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, and Illia Polosukhin. "Attention Is All '
        'You Need." arXiv, (2017). https://doi.org/10.48550/arXiv.1706.03762.\nMLA:',
        "output_1": 'Vaswani, Ashish, et al. "Attention Is All You Need." arXiv, '
        "2017. https://doi.org/10.48550/arXiv.1706.03762.",
        "output_2": 'Vaswani, Ashish et al. "Attention Is All You Need." arXiv, 2017, '
        "https://doi.org/10.48550/arXiv.1706.03762. MLA.",
        "preference": 1,
        "annotator_index": 2,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 0,
        "n_annotated": 4,
    },
    {
        "instruction": "You are given a paper citation, convert it to the requested citation "
        "style.\n\nChicago: Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, "
        'Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, and Illia Polosukhin. "Attention Is All '
        'You Need." arXiv, (2017). https://doi.org/10.48550/arXiv.1706.03762.\nMLA:',
        "output_1": 'Vaswani, Ashish, et al. "Attention Is All You Need." arXiv, '
        "2017. https://doi.org/10.48550/arXiv.1706.03762.",
        "output_2": 'Vaswani, Ashish et al. "Attention Is All You Need." arXiv, 2017, '
        "https://doi.org/10.48550/arXiv.1706.03762. MLA.",
        "preference": 1,
        "annotator_index": 0,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 1,
        "n_annotated": 4,
    },
    {
        "instruction": "You are given a paper citation, convert it to the requested citation "
        "style.\n\nChicago: Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, "
        'Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, and Illia Polosukhin. "Attention Is All '
        'You Need." arXiv, (2017). https://doi.org/10.48550/arXiv.1706.03762.\nMLA:',
        "output_1": 'Vaswani, Ashish, et al. "Attention Is All You Need." arXiv, '
        "2017. https://doi.org/10.48550/arXiv.1706.03762.",
        "output_2": 'Vaswani, Ashish et al. "Attention Is All You Need." arXiv, 2017, '
        "https://doi.org/10.48550/arXiv.1706.03762. MLA.",
        "preference": 1,
        "annotator_index": 15,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 2,
        "n_annotated": 4,
    },
    {
        "instruction": "You are given a paper citation, convert it to the requested citation "
        "style.\n\nChicago: Vaswani, Ashish, Shazeer, Noam, Parmar, Niki, Uszkoreit, Jakob, "
        'Jones, Llion, Gomez, Aidan N., Kaiser, Lukasz, and Illia Polosukhin. "Attention Is All '
        'You Need." arXiv, (2017). https://doi.org/10.48550/arXiv.1706.03762.\nMLA:',
        "output_1": 'Vaswani, Ashish, et al. "Attention Is All You Need." arXiv, '
        "2017. https://doi.org/10.48550/arXiv.1706.03762.",
        "output_2": 'Vaswani, Ashish et al. "Attention Is All You Need." arXiv, 2017, '
        "https://doi.org/10.48550/arXiv.1706.03762. MLA.",
        "preference": 1,
        "annotator_index": 4,
        "dataset": "selfinstruct",
        "datasplit": "eval",
        "time_per_example": 36.8,
        "price_per_example": 0.3,
        "index": 3,
        "n_annotated": 4,
    },
]
