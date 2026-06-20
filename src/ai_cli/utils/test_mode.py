import os


def is_test_mode() -> bool:
    """
    Return True when running under tests (controlled via AI_CLI_TEST_MODE env var).
    """
    return os.getenv("AI_CLI_TEST_MODE", "0") == "1"