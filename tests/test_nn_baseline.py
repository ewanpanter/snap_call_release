"""data/nn_baseline.py on tiny hand-made splits."""

from data import nn_baseline as NN


def _row(text, label, flag):
    return {"dataset": "d", "state": text, "gold": {"label": label, "flag": flag}}


def test_nn_copies_the_closest_train_answer():
    train = [
        _row("my mortgage payment was lost by the servicer", "mortgage", True),
        _row("a debt collector keeps calling about a debt I do not owe", "debt", False),
        _row("my credit card was charged twice for one purchase", "card", False),
    ]
    test = [
        _row("the servicer lost my mortgage payment again", "mortgage", True),
        _row("collector calling about debt that is not mine", "debt", True),
    ]
    r = NN.nn_baseline(train, test, max_df=1.0)
    assert r["questions"]["label"]["nn_accuracy"] == 1.0
    assert r["questions"]["flag"]["nn_accuracy"] == 0.5
    assert r["nn_all_questions_right"] == 0.5
    assert r["questions"]["flag"]["majority_answer"] is False
    assert r["nn_cosine"]["share_ge_0.9"] == 0.0
    assert "| label | 1.000 |" in NN.render(r)


def test_redactions_are_ignored():
    assert NN.tokens("On XX/XX/XXXX I paid {$100.00} to XXXX Bank") == ["on", "paid", "to", "bank"]


def test_exact_copy_is_a_near_copy():
    train = [_row("identical complaint text here", "a", True), _row("other words", "b", False)]
    r = NN.nn_baseline(train, [_row("Identical complaint text here", "a", True)], max_df=1.0)
    assert r["nn_cosine"]["share_ge_0.9"] == 1.0


def test_keep_numbers_matches_tables():
    assert NN.tokens("PCI|-85.48|11.59|600", keep_numbers=True) == ["pci", "-85.48", "11.59", "600"]
    # Same words everywhere: without numbers it's a tie, which goes to the first train row.
    train = [_row("rsrp -99.2 speed 12", "rbs", True), _row("rsrp -85.1 speed 60", "speed", True)]
    test = [_row("rsrp -85.1 speed 61", "speed", True)]
    assert NN.nn_baseline(train, test, max_df=1.0)["questions"]["label"]["nn_accuracy"] == 0.0
    r = NN.nn_baseline(train, test, max_df=1.0, keep_numbers=True)
    assert r["questions"]["label"]["nn_accuracy"] == 1.0
