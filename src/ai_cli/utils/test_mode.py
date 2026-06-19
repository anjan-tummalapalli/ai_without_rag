import os


def is_test_mode() -> bool:
    return os.getenv("AI_CLI_TEST_MODE", "0") == "1"