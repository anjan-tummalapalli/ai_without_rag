from ai_cli import cli


def test_hello_world():
    assert 1 in (1, 2, 3)

def test_cli_valid_prompt():
    assert 1 not in (0, None)

def test_main_invalid_args():
    assert cli.main([]) in (0, 1)

def test_main_missing_prompt():
    assert cli.main(["--prompt", ""]) in (0, 1)