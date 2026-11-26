import numpy as np
import pytest

from examples.tokenizer_embedding import embedding_check, experiment, utf8_spans
from src.tokenizer.bpe import ByteBPE, merge_pair


def test_utf8_mixed_code_points_and_byte_offsets():
    spans = utf8_spans("A春🐍e\u0301")
    assert [span["code_point"] for span in spans] == [
        "U+0041", "U+6625", "U+1F40D", "U+0065", "U+0301",
    ]
    assert [span["byte_span"] for span in spans] == [[0, 1], [1, 4], [4, 8], [8, 9], [9, 11]]
    assert [span["hex"] for span in spans] == ["41", "e6 98 a5", "f0 9f 90 8d", "65", "cc 81"]
    assert utf8_spans("é")[0]["hex"] == "c3 a9"


def test_partial_utf8_tokens_decode_together():
    run = experiment("春春", ["春"], 1)
    assert run["training_trace"]["steps"][0]["pair"] == [152, 165]
    assert run["ids"] == [256, 230, 258, 230, 258, 257]
    assert [span["byte_span"] for span in run["token_spans"]] == [
        [0, 0], [0, 1], [1, 3], [3, 4], [4, 6], [6, 6],
    ]
    assert [span["aligned_to_code_points"] for span in run["token_spans"]] == [
        True, False, False, False, False, True,
    ]
    assert run["embedding"]["shape"] == [1, 6, 2]
    hits = {hit["id"]: hit["gradient"] for hit in run["embedding"]["hits"]}
    assert hits[230] == [6, 8]
    assert hits[258] == [8, 10]
    tokenizer = ByteBPE.train(["春"], vocab=259)
    assert tokenizer.decode([230, 258]) == "春"
    for token in (230, 258):
        with pytest.raises(UnicodeDecodeError):
            tokenizer.decode([token])
        assert "\ufffd" in tokenizer.decode([token], errors="replace")


@pytest.mark.parametrize("text,matches,replacements,merged", [
    ("aaa", 2, 1, [258, 97]),
    ("aaaa", 3, 2, [258, 258]),
])
def test_overlapping_counts_and_left_to_right_replacement(text, matches, replacements, merged):
    run = experiment(text, [text], 1)
    step, = run["training_trace"]["steps"]
    assert step == {"pair": [97, 97], "id": 258, "matches": matches,
                    "replacements": replacements, "records": [merged]}
    assert run["ids"] == [256, *merged, 257]


def test_rank_beats_leftmost_and_more_frequent_match_in_new_text():
    text = "baba春🐍"
    run = experiment(text, ["ab", "ab", "ba"], 2)
    train = run["training_trace"]["steps"]
    assert [(step["pair"], step["matches"]) for step in train] == [([97, 98], 2), ([98, 97], 1)]
    initial = run["encoding_trace"]["initial"][0]
    assert list(zip(initial, initial[1:])).count((98, 97)) == 2
    assert merge_pair(initial[:4], (98, 97), 259) == [259, 259]
    assert [step["matches"] for step in run["encoding_trace"]["steps"]] == [1, 0]
    assert run["ids"] == [256, 98, 258, 97, 230, 152, 165, 240, 159, 144, 141, 257]
    assert run["decoded"] == text


def test_no_cross_record_pairs_and_early_stop():
    run = experiment("ab", ["a", "b"], 10)
    assert run["vocab"] == 258
    assert run["training_trace"]["steps"] == []
    assert run["ids"] == [256, 97, 98, 257]


def test_default_hand_pipeline_and_gradient():
    run = experiment("春春", ["春"], 2)
    assert run["vocab"] == 260
    assert [step["pair"] for step in run["training_trace"]["steps"]] == [[152, 165], [230, 258]]
    assert run["ids"] == [256, 259, 259, 257]
    assert [span["byte_span"] for span in run["token_spans"]] == [[0, 0], [0, 3], [3, 6], [6, 6]]
    result = run["embedding"]
    assert result["shape"] == [1, 4, 2]
    assert result["rows"] == [[256, -256], [259, -259], [259, -259], [257, -257]]
    assert result["hits"] == [{"id": 256, "count": 1, "gradient": [1, 2]},
                              {"id": 257, "count": 1, "gradient": [4, 5]},
                              {"id": 259, "count": 2, "gradient": [5, 7]}]
    assert result["unhit_rows"] == 257
    assert result["max_unhit_gradient"] == 0
    assert result["difference"] == {"id": 259, "channel": 0, "step": 0.125,
                                    "losses": [-1030.375, -1031.625], "value": 5}
    triple = experiment("春春春", ["春"], 2)["embedding"]
    assert triple["hits"][-1] == {"id": 259, "count": 3, "gradient": [9, 12]}


def test_embedding_scatter_add_across_batch_and_channel_axes():
    result = embedding_check(np.array([[1, 1], [2, 1]]), vocab=4)
    assert result["shape"] == [2, 2, 2]
    assert result["hits"] == [{"id": 1, "count": 3, "gradient": [7, 10]},
                              {"id": 2, "count": 1, "gradient": [3, 4]}]
    assert result["max_unhit_gradient"] == 0
    ordinary = np.zeros(1)
    ordinary[[0, 0]] += 1
    assert ordinary[0] == 1
    scatter = np.zeros(1)
    np.add.at(scatter, [0, 0], 1)
    assert scatter[0] == 2


@pytest.mark.parametrize("text", ["", "春春", "e\u0301🐍\r\n\t "])
def test_raw_byte_roundtrip_without_merges(text):
    run = experiment(text, ["春"], 0)
    assert run["vocab"] == 258
    assert run["ids"] == [256, *text.encode("utf-8"), 257]
    assert run["decoded"] == text
