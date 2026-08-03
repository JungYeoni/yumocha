from scripts.apply_2023_semantic_corrections import CORRECTIONS


def test_2023_correction_keys_are_unique():
    keys = [(rule.region, rule.source_row) for rule in CORRECTIONS]

    assert len(keys) == 7
    assert len(set(keys)) == len(keys)
