def test_import_all_modules():

    import ai_cli.cli
    import ai_cli.ai_chat

    import ai_cli.providers.factory
    import ai_cli.providers.registry
    import ai_cli.providers.config
    import ai_cli.providers.contracts
    import ai_cli.providers.decorators

    import ai_cli.rag.pipeline
    import ai_cli.plugins.builtins
