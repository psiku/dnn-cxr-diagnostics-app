from src.core.pathologies import PATHOLOGIES, load_pathologies


def test_load_pathologies_matches_config():
    names = load_pathologies()
    assert len(names) == 14
    assert names[0] == "Infiltration"
    assert names[-1] == "Hernia"
    assert PATHOLOGIES == names
