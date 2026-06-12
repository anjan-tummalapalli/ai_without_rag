from ai_cli.cli import build_parser

def test_cli_build_parser():
    parser = build_parser()
    assert parser is not None

def test_cli_help():
    parser = build_parser()
    try:
        parser.parse_args(["--help"])

    except SystemExit as exc:
        assert exc.code == 0
