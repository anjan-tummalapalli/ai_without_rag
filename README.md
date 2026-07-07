# AI CLI Gateway — v0.3.0

> Multi-provider AI CLI gateway with pluggable LLM providers, resilience

---

## What's New in v0.3.0

- **Provider registry** — lazy-loaded providers with deterministic registration (`auto`, `openai`, `gemini`, `cohere`, `perplexity`, `xai`, `zai`, `echo`)
- **Resilience** — retry/backoff engine and response validation

---

## Project Structure

```
src/ai_cli/
├── cli.py                  # CLI entrypoint (ai-cli)
├── core/
│   ├── api.py              # ask() public API
│   ├── resilience.py       # RetryEngine, circuit breaker
│   └── service/
│       ├── ask_service.py  # Chat provider wrapper
│       └── embed_service.py
├── providers/              # LLM provider implementations
│   ├── registry.py         # Provider registration & factory
│   ├── bootstrap.py        # Lazy provider initialization
│   └── *_provider.py
├── AI workflow/
│   ├── prompt segmenter.py          # Sliding-window SemanticChunker
│   ├── pipeline.py         # Token-aware sentence prompt segmenter
│   ├── retriever.py        # Query → embed → search
│   ├── in_memory.py        # Lightweight CLI AI workflow pipeline
└── config/AI workflow_config.py    # AI workflow defaults and AI workflowConfig dataclass
```

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/yourusername/ai-cli.git
cd ai-cli
poetry install

poetry install --with AI workflow
```

### 2. Configure environment

```bash
cp .env.example .env
# Set provider API keys, e.g.:
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

### 3. CLI examples

```bash
# Single prompt
ai-cli -q "Explain Kubernetes operators"

# Specify provider and model
ai-cli --provider openai --model gpt-4o-mini -q "Hello"

# Interactive chat
ai-cli --interactive --provider openai

# AI workflow: index docs and query with context
ai-cli --AI workflow --AI workflow-docs docs/manual.md README.md \
  -q "Summarize the security model" --AI workflow-top-k 5

# Interactive AI workflow (use /index and /search inside the session)
ai-cli --interactive --AI workflow --provider echo
```

### CLI flags (highlights)

| Flag | Description |
|------|-------------|
| `-p`, `--provider` | Provider name (default: `auto`) |
| `-q`, `--prompt` | Prompt text (or pipe via stdin) |
| `-m`, `--model` | Model override |
| `-i`, `--interactive` | Interactive REPL |
| `--AI workflow` | Enable in-memory AI workflow context injection |
| `--AI workflow-docs` | File paths or raw text to index |
| `--AI workflow-prompt segment-size` | Chunk size in characters (default: 500) |
| `--AI workflow-prompt segment-overlap` | Overlap between prompt segments (default: 50) |
| `--AI workflow-top-k` | Top-k prompt segments to retrieve (default: 5) |
| `--timeout` | Request timeout in seconds |
| `--stream` | Stream responses when supported |
| `--debug` | Enable debug logging |

---

## Python API

### Ask a provider

```python
from ai_cli.core.api import ask

response = ask(
    prompt="What is AI workflow?",
    provider="openai",
    model="gpt-4o-mini",
)
print(response)
```

### Chunk documents

```python
from ai_cli.AI workflow import prompt segment_text, SemanticChunker

prompt segments = prompt segment_text("Long document text...", source="manual.md")
# Or use the prompt segmenter class directly:
prompt segmenter = SemanticChunker(prompt segment_size=500, overlap=50)
prompt segments = prompt segmenter.prompt segment_text(text, source="manual.md")
```

### Embeddings + provider integrations

```python
from ai_cli.AI workflow import EmbeddingGenerator, VectorStore

embedder = EmbeddingGenerator(model="all-MiniLM-L6-v2", batch_size=32)
vectors = embedder.embed_batch([c.text for c in prompt segments])

store = VectorStore()
store.create_index(dimension=len(vectors[0]))
store.add_model integrations(vectors, prompt segments)
store.save()
```


```python
from ai_cli.AI workflow import Retriever

retriever = Retriever(store=store, embedder=embedder, top_k=5)
context = retriever.build_context("How do we rotate secrets?")
answer = ask(prompt=f"Context:\n{context}\n\nQuestion: How do we rotate secrets?")
```

---

## Supported Providers

| Provider | Env variable | Notes |
|----------|--------------|-------|
| `auto` | — | Falls back through available providers |
| `openai` | `OPENAI_API_KEY` | Chat + model integrations |
| `gemini` | `GEMINI_API_KEY` | Google Gemini |
| `cohere` | `COHERE_API_KEY` | Cohere chat |
| `perplexity` | `PERPLEXITY_API_KEY` | Search-augmented |
| `xai` | `XAI_API_KEY` | xAI Grok |
| `zai` | `ZAI_API_KEY` | Z.AI |
| `echo` | — | Local test provider (no API key) |

---

## Testing

```bash
pytest tests/ -v
pytest tests/test_enhanced.py -v   # AI workflow prompt processing + model integrations
pytest tests/test_providers.py -v  # Provider contracts
```

---

## Check Code Coverage

python -m pytest --cov=src --cov-report=term-missing

============================================================================================= test session starts =============================================================================================
platform darwin -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0 -- /Users/anjan/Documents/New project/ai_chat/ai_cli_without_rag/.venv/bin/python3
cachedir: .pytest_cache
rootdir: /Users/anjan/Documents/New project/ai_chat/ai_cli_without_rag
configfile: pyproject.toml
testpaths: tests
plugins: cov-7.1.0, anyio-4.14.1, asyncio-1.4.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 574 items                                                                                                                                                                                           

tests/test_ai_chat.py::test_last_sentence_end_found PASSED                                                                                                                                              [  0%]
tests/test_ai_chat.py::test_last_sentence_end_missing PASSED                                                                                                                                            [  0%]
tests/test_ai_chat.py::test_last_whitespace_found PASSED                                                                                                                                                [  0%]
tests/test_ai_chat.py::test_last_whitespace_missing PASSED                                                                                                                                              [  0%]
tests/test_ai_chat.py::test_next_start PASSED                                                                                                                                                           [  0%]
tests/test_ai_chat.py::test_chunk_text_empty PASSED                                                                                                                                                     [  1%]
tests/test_ai_chat.py::test_chunk_text_basic PASSED                                                                                                                                                     [  1%]
tests/test_ai_chat.py::test_ai_chat_chunking PASSED                                                                                                                                                     [  1%]
tests/test_ai_chat.py::test_ai_chat_empty PASSED                                                                                                                                                        [  1%]
tests/test_ai_chat.py::test_last_whitespace_tabs PASSED                                                                                                                                                 [  1%]
tests/test_ai_chat.py::test_chunk_text_invalid_chunk_size PASSED                                                                                                                                        [  1%]
tests/test_ai_chat.py::test_chunk_text_invalid_overlap PASSED                                                                                                                                           [  2%]
tests/test_ai_chat.py::test_chunk_text_whitespace_only PASSED                                                                                                                                           [  2%]
tests/test_ai_chat.py::test_chunk_text_sentence_boundary PASSED                                                                                                                                         [  2%]
tests/test_ai_chat.py::test_chunk_text_without_word_split PASSED                                                                                                                                        [  2%]
tests/test_ai_chat.py::test_chunk_text_word_split PASSED                                                                                                                                                [  2%]
tests/test_ai_chat.py::test_chunk_text_exact_length_break PASSED                                                                                                                                        [  2%]
tests/test_ai_chat.py::test_chunk_text_multiple_chunks_progress PASSED                                                                                                                                  [  3%]
tests/test_ai_chat.py::test_chunk_text_empty_after_strip_exit PASSED                                                                                                                                    [  3%]
tests/test_ai_chat.py::test_chunk_text_empty_chunk_after_strip PASSED                                                                                                                                   [  3%]
tests/test_ai_chat.py::test_build_kwargs_modules PASSED                                                                                                                                                 [  3%]
tests/test_ai_chat.py::test_build_kwargs_signature_failure PASSED                                                                                                                                       [  3%]
tests/test_ai_chat.py::test_decode_chunk_json_failure PASSED                                                                                                                                            [  4%]
tests/test_ai_chat.py::test_handle_sync_runtime_error PASSED                                                                                                                                            [  4%]
tests/test_ai_chat.py::test_chunk_text_invalid_chunk_size_type PASSED                                                                                                                                   [  4%]
tests/test_ai_chat.py::test_chunk_text_invalid_overlap_type PASSED                                                                                                                                      [  4%]
tests/test_ai_chat.py::test_chunk_text_chunk_size_less_than_overlap PASSED                                                                                                                              [  4%]
tests/test_ai_chat.py::test_chunk_text_empty_string PASSED                                                                                                                                              [  4%]
tests/test_ai_chat.py::test_chunk_text_shorter_than_chunk_size PASSED                                                                                                                                   [  5%]
tests/test_ai_chat.py::test_chunk_text_zero_overlap_forward_progress PASSED                                                                                                                             [  5%]
tests/test_ai_chat_coverage.py::test_chunk_text_basic PASSED                                                                                                                                            [  5%]
tests/test_ai_chat_coverage.py::test_chunk_text_empty PASSED                                                                                                                                            [  5%]
tests/test_ai_chat_coverage.py::test_chunk_text_overlap PASSED                                                                                                                                          [  5%]
tests/test_cli_bootstrap.py::test_cli_main_module_runs PASSED                                                                                                                                           [  5%]
tests/test_cli_bootstrap.py::test_cli_parser_structure PASSED                                                                                                                                           [  6%]
tests/test_cli_bootstrap.py::test_monitoring_smoke PASSED                                                                                                                                               [  6%]
tests/test_cli_bootstrap.py::test_prompt_corrector_basic_execution PASSED                                                                                                                               [  6%]
tests/test_cli_bootstrap.py::test_import_heavy_modules PASSED                                                                                                                                           [  6%]
tests/test_cli_branches.py::test_cli_missing_prompt_exit PASSED                                                                                                                                         [  6%]
tests/test_cli_branches.py::test_cli_empty_prompt_exit PASSED                                                                                                                                           [  6%]
tests/test_cli_branches.py::test_cli_valid_prompt PASSED                                                                                                                                                [  7%]
tests/test_cli_branches.py::test_cli_no_args PASSED                                                                                                                                                     [  7%]
tests/test_cli_branches.py::test_cli_provider_selection[openai] PASSED                                                                                                                                  [  7%]
tests/test_cli_branches.py::test_cli_provider_selection[gemini] PASSED                                                                                                                                  [  7%]
tests/test_cli_branches.py::test_cli_provider_selection[cohere] PASSED                                                                                                                                  [  7%]
tests/test_cli_branches.py::test_cli_provider_selection[deepseek] PASSED                                                                                                                                [  8%]
tests/test_cli_branches.py::test_cli_provider_selection[xai] PASSED                                                                                                                                     [  8%]
tests/test_cli_branches.py::test_cli_provider_selection[perplexity] PASSED                                                                                                                              [  8%]
tests/test_cli_branches.py::test_cli_missing_prompt PASSED                                                                                                                                              [  8%]
tests/test_cli_branches.py::test_cli_basic_run PASSED                                                                                                                                                   [  8%]
tests/test_cli_branches.py::test_cli_with_rag_flag PASSED                                                                                                                                               [  8%]
tests/test_cli_coverage.py::test_cli_build_parser PASSED                                                                                                                                                [  9%]
tests/test_cli_coverage.py::test_cli_help PASSED                                                                                                                                                        [  9%]
tests/test_cli_coverage.py::test_sanitize_log_value PASSED                                                                                                                                              [  9%]
tests/test_cli_coverage.py::test_safe_resolve_path_empty PASSED                                                                                                                                         [  9%]
tests/test_cli_coverage.py::test_decode_chunk_string PASSED                                                                                                                                             [  9%]
tests/test_cli_coverage.py::test_decode_chunk_bytes PASSED                                                                                                                                              [  9%]
tests/test_cli_coverage.py::test_missing_prompt PASSED                                                                                                                                                  [ 10%]
tests/test_cli_coverage.py::test_empty_prompt PASSED                                                                                                                                                    [ 10%]
tests/test_cli_coverage.py::test_read_stdin_prompt PASSED                                                                                                                                               [ 10%]
tests/test_cli_coverage.py::test_timeout_validation PASSED                                                                                                                                              [ 10%]
tests/test_cli_coverage.py::test_large_prompt_truncation PASSED                                                                                                                                         [ 10%]
tests/test_cli_coverage.py::test_load_rag_docs_text PASSED                                                                                                                                              [ 10%]
tests/test_cli_coverage.py::test_interactive_exit PASSED                                                                                                                                                [ 11%]
tests/test_cli_coverage.py::test_safe_resolve_path_none PASSED                                                                                                                                          [ 11%]
tests/test_cli_coverage.py::test_chunk_validation PASSED                                                                                                                                                [ 11%]
tests/test_cli_coverage.py::test_cli_no_args_exit PASSED                                                                                                                                                [ 11%]
tests/test_cli_coverage.py::test_cli_invalid_provider PASSED                                                                                                                                            [ 11%]
tests/test_cli_coverage.py::test_cli_provider_error PASSED                                                                                                                                              [ 12%]
tests/test_cli_coverage.py::test_cli_stdin_prompt PASSED                                                                                                                                                [ 12%]
tests/test_cli_coverage.py::test_cli_prompt_success PASSED                                                                                                                                              [ 12%]
tests/test_cli_coverage.py::test_cli_stream_mode PASSED                                                                                                                                                 [ 12%]
tests/test_cli_coverage.py::test_cli_debug PASSED                                                                                                                                                       [ 12%]
tests/test_cli_coverage.py::test_cli_timeout_success PASSED                                                                                                                                             [ 12%]
tests/test_cli_coverage.py::test_cli_debug_success PASSED                                                                                                                                               [ 13%]
tests/test_cli_coverage.py::test_run_interactive_exit PASSED                                                                                                                                            [ 13%]
tests/test_cli_coverage.py::test_run_interactive_commands PASSED                                                                                                                                        [ 13%]
tests/test_cli_coverage.py::test_run_interactive_prompt PASSED                                                                                                                                          [ 13%]
tests/test_cli_coverage.py::test_cli_reads_stdin PASSED                                                                                                                                                 [ 13%]
tests/test_cli_coverage.py::test_sync_result_dict PASSED                                                                                                                                                [ 13%]
tests/test_cli_coverage.py::test_sync_result_iterable PASSED                                                                                                                                            [ 14%]
tests/test_cli_coverage.py::test_decode_chunk PASSED                                                                                                                                                    [ 14%]
tests/test_cli_coverage.py::test_interactive_help_exit PASSED                                                                                                                                           [ 14%]
tests/test_cli_coverage.py::test_cli_rag_prompt PASSED                                                                                                                                                  [ 14%]
tests/test_cli_coverage.py::test_interactive_commands PASSED                                                                                                                                            [ 14%]
tests/test_cli_coverage.py::test_read_stdin_prompt_bytes PASSED                                                                                                                                         [ 14%]
tests/test_cli_coverage.py::test_timeout_negative PASSED                                                                                                                                                [ 15%]
tests/test_cli_coverage.py::test_cli_unknown_provider PASSED                                                                                                                                            [ 15%]
tests/test_cli_coverage.py::test_auto_provider_init_failure PASSED                                                                                                                                      [ 15%]
tests/test_cli_coverage.py::test_deepseek_health_check_without_key PASSED                                                                                                                               [ 15%]
tests/test_cli_coverage.py::test_prompt_truncation_in_main PASSED                                                                                                                                       [ 15%]
tests/test_cli_coverage.py::test_safe_resolve_path_null_byte PASSED                                                                                                                                     [ 16%]
tests/test_cli_coverage.py::test_safe_resolve_path_traversal PASSED                                                                                                                                     [ 16%]
tests/test_cli_coverage.py::test_decode_chunk_json_object PASSED                                                                                                                                        [ 16%]
tests/test_cli_coverage.py::test_drain_async_result_string PASSED                                                                                                                                       [ 16%]
tests/test_cli_coverage.py::test_decode_chunk_unserializable PASSED                                                                                                                                     [ 16%]
tests/test_cli_coverage.py::test_drain_async_result_iterable PASSED                                                                                                                                     [ 16%]
tests/test_cli_coverage.py::test_drain_async_result_bytes PASSED                                                                                                                                        [ 17%]
tests/test_cli_coverage.py::test_load_rag_docs_raw_text PASSED                                                                                                                                          [ 17%]
tests/test_cli_coverage.py::test_load_rag_docs_missing_file PASSED                                                                                                                                      [ 17%]
tests/test_cli_coverage.py::test_deepseek_send_text_response PASSED                                                                                                                                     [ 17%]
tests/test_cli_coverage.py::test_deepseek_chat_success PASSED                                                                                                                                           [ 17%]
tests/test_cli_coverage.py::test_safe_resolve_path_invalid_null PASSED                                                                                                                                  [ 17%]
tests/test_cli_coverage.py::test_safe_resolve_path_parent_traversal PASSED                                                                                                                              [ 18%]
tests/test_cli_coverage.py::test_build_ask_kwargs_signature_failure PASSED                                                                                                                              [ 18%]
tests/test_cli_coverage.py::test_read_stdin_prompt_empty PASSED                                                                                                                                         [ 18%]
tests/test_cli_coverage.py::test_load_rag_docs_reads_file PASSED                                                                                                                                        [ 18%]
tests/test_cli_coverage.py::test_load_rag_docs_bad_file PASSED                                                                                                                                          [ 18%]
tests/test_cli_coverage.py::test_safe_resolve_normal_path PASSED                                                                                                                                        [ 18%]
tests/test_cli_coverage.py::test_decode_chunk_object PASSED                                                                                                                                             [ 19%]
tests/test_cli_coverage.py::test_decode_chunk_bytes_and_objects PASSED                                                                                                                                  [ 19%]
tests/test_cli_coverage.py::test_drain_async_result_simple PASSED                                                                                                                                       [ 19%]
tests/test_cli_coverage.py::test_drain_async_result_error PASSED                                                                                                                                        [ 19%]
tests/test_cli_coverage.py::test_safe_resolve_path_null PASSED                                                                                                                                          [ 19%]
tests/test_cli_coverage.py::test_drain_async_result_async_iterable PASSED                                                                                                                               [ 20%]
tests/test_cli_coverage.py::test_drain_async_result_sync_iterable PASSED                                                                                                                                [ 20%]
tests/test_cli_coverage.py::test_drain_async_keyboardinterrupt PASSED                                                                                                                                   [ 20%]
tests/test_cli_coverage.py::test_handle_sync_result_iterable PASSED                                                                                                                                     [ 20%]
tests/test_cli_coverage.py::test_handle_sync_result_dict PASSED                                                                                                                                         [ 20%]
tests/test_cli_coverage.py::test_handle_sync_result_keyboardinterrupt PASSED                                                                                                                            [ 20%]
tests/test_cli_coverage.py::test_handle_sync_result_runtimeerror PASSED                                                                                                                                 [ 21%]
tests/test_cli_coverage.py::test_invoke_with_retries_invalid PASSED                                                                                                                                     [ 21%]
tests/test_cli_coverage.py::test_invoke_with_retries_timeout PASSED                                                                                                                                     [ 21%]
tests/test_cli_coverage.py::test_invoke_with_retries_connection PASSED                                                                                                                                  [ 21%]
tests/test_cli_coverage.py::test_invoke_with_retries_keyboard PASSED                                                                                                                                    [ 21%]
tests/test_cli_coverage.py::test_invoke_with_retries_runtime PASSED                                                                                                                                     [ 21%]
tests/test_cli_coverage.py::test_read_stdin_prompt_truncated PASSED                                                                                                                                     [ 22%]
tests/test_cli_coverage.py::test_run_interactive_help_exit PASSED                                                                                                                                       [ 22%]
tests/test_cli_coverage.py::test_run_interactive_search PASSED                                                                                                                                          [ 22%]
tests/test_cli_interactive_coverage.py::test_interactive_exit PASSED                                                                                                                                    [ 22%]
tests/test_cli_interactive_coverage.py::test_rag_retrieve_empty PASSED                                                                                                                                  [ 22%]
tests/test_cli_interactive_coverage.py::test_rag_upsert_and_search PASSED                                                                                                                               [ 22%]
tests/test_cli_missing_paths.py::test_retry_engine_exception_retry PASSED                                                                                                                               [ 23%]
tests/test_cli_missing_paths.py::test_retry_engine_retry_on_tuple_break PASSED                                                                                                                          [ 23%]
tests/test_cli_missing_paths.py::test_retry_engine_retry_filter_break PASSED                                                                                                                            [ 23%]
tests/test_cli_missing_paths.py::test_retry_engine_base_delay PASSED                                                                                                                                    [ 23%]
tests/test_cli_missing_paths.py::test_interactive_profile_and_stream PASSED                                                                                                                             [ 23%]
tests/test_cli_missing_paths.py::test_interactive_index_text PASSED                                                                                                                                     [ 24%]
tests/test_cli_missing_paths.py::test_cli_stdin_empty PASSED                                                                                                                                            [ 24%]
tests/test_cli_missing_paths.py::test_interactive_profile_stream PASSED                                                                                                                                 [ 24%]
tests/test_cli_missing_paths.py::test_interactive_search PASSED                                                                                                                                         [ 24%]
tests/test_cli_missing_paths.py::test_interactive_index_raw_text PASSED                                                                                                                                 [ 24%]
tests/test_cli_runtime.py::test_ask_basic PASSED                                                                                                                                                        [ 24%]
tests/test_cli_runtime.py::test_cli_missing_prompt_exit PASSED                                                                                                                                          [ 25%]
tests/test_coverage_boost.py::TestAIProviderError::test_basic_message PASSED                                                                                                                            [ 25%]
tests/test_coverage_boost.py::TestAIProviderError::test_with_code PASSED                                                                                                                                [ 25%]
tests/test_coverage_boost.py::TestAIProviderError::test_retryable PASSED                                                                                                                                [ 25%]
tests/test_coverage_boost.py::TestAIProviderError::test_with_details PASSED                                                                                                                             [ 25%]
tests/test_coverage_boost.py::TestAIProviderError::test_with_cause PASSED                                                                                                                               [ 25%]
tests/test_coverage_boost.py::TestAIProviderError::test_to_dict PASSED                                                                                                                                  [ 26%]
tests/test_coverage_boost.py::TestAIProviderError::test_to_dict_with_cause PASSED                                                                                                                       [ 26%]
tests/test_coverage_boost.py::TestAIProviderError::test_to_json PASSED                                                                                                                                  [ 26%]
tests/test_coverage_boost.py::TestAIProviderError::test_from_exception PASSED                                                                                                                           [ 26%]
tests/test_coverage_boost.py::TestAIProviderError::test_from_exception_custom_message PASSED                                                                                                            [ 26%]
tests/test_coverage_boost.py::TestProviderRequestError::test_with_all_details PASSED                                                                                                                    [ 27%]
tests/test_coverage_boost.py::TestProviderRequestError::test_without_optional_details PASSED                                                                                                            [ 27%]
tests/test_coverage_boost.py::TestChunkingError::test_with_chunk_index PASSED                                                                                                                           [ 27%]
tests/test_coverage_boost.py::TestChunkingError::test_without_chunk_index PASSED                                                                                                                        [ 27%]
tests/test_coverage_boost.py::TestEmbeddingError::test_with_model PASSED                                                                                                                                [ 27%]
tests/test_coverage_boost.py::TestEmbeddingError::test_without_model PASSED                                                                                                                             [ 27%]
tests/test_coverage_boost.py::TestVectorDBError::test_with_operation PASSED                                                                                                                             [ 28%]
tests/test_coverage_boost.py::TestVectorDBError::test_without_operation PASSED                                                                                                                          [ 28%]
tests/test_coverage_boost.py::TestRetrievalError::test_with_query_and_retrieved PASSED                                                                                                                  [ 28%]
tests/test_coverage_boost.py::TestRetrievalError::test_without_optional_params PASSED                                                                                                                   [ 28%]
tests/test_coverage_boost.py::TestCaptureExceptionInfo::test_captures_info PASSED                                                                                                                       [ 28%]
tests/test_coverage_boost.py::TestSubclassInstantiation::test_prompt_validation_error PASSED                                                                                                            [ 28%]
tests/test_coverage_boost.py::TestSubclassInstantiation::test_provider_configuration_error PASSED                                                                                                       [ 29%]
tests/test_coverage_boost.py::TestSubclassInstantiation::test_response_validation_error PASSED                                                                                                          [ 29%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_tokens_basic PASSED                                                                                                                        [ 29%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_tokens_empty PASSED                                                                                                                        [ 29%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_tokens_with_meta PASSED                                                                                                                    [ 29%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_sentences PASSED                                                                                                                           [ 29%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_sentences_empty PASSED                                                                                                                     [ 30%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_sentences_with_meta PASSED                                                                                                                 [ 30%]
tests/test_coverage_boost.py::TestTextChunker::test_chunk_by_sentences_with_max_tokens PASSED                                                                                                           [ 30%]
tests/test_coverage_boost.py::TestTextChunker::test_max_chunks_limit_tokens PASSED                                                                                                                      [ 30%]
tests/test_coverage_boost.py::TestTextChunker::test_max_chunks_limit_sentences PASSED                                                                                                                   [ 30%]
tests/test_coverage_boost.py::TestPromptCorrector::test_correct_empty PASSED                                                                                                                            [ 31%]
tests/test_coverage_boost.py::TestPromptCorrector::test_correct_by_sentences PASSED                                                                                                                     [ 31%]
tests/test_coverage_boost.py::TestPromptCorrector::test_correct_by_tokens PASSED                                                                                                                        [ 31%]
tests/test_coverage_boost.py::TestPromptCorrector::test_correct_with_meta_empty PASSED                                                                                                                  [ 31%]
tests/test_coverage_boost.py::TestPromptCorrector::test_correct_with_meta_by_sentences PASSED                                                                                                           [ 31%]
tests/test_coverage_boost.py::TestPromptCorrector::test_correct_with_meta_by_tokens PASSED                                                                                                              [ 31%]
tests/test_coverage_boost.py::TestPromptCorrectorFunctions::test_prompt_corrector_function PASSED                                                                                                       [ 32%]
tests/test_coverage_boost.py::TestPromptCorrectorFunctions::test_correct_prompt_empty PASSED                                                                                                            [ 32%]
tests/test_coverage_boost.py::TestPromptCorrectorFunctions::test_correct_prompt_nonempty PASSED                                                                                                         [ 32%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_test_key PASSED                                                                                                                        [ 32%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_success PASSED                                                                                                                         [ 32%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_no_choices PASSED                                                                                                                      [ 32%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_empty_content PASSED                                                                                                                   [ 33%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_api_exception PASSED                                                                                                                   [ 33%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_call_model_with_system_prompt PASSED                                                                                                        [ 33%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_call_model_no_choices PASSED                                                                                                                [ 33%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_call_model_no_content PASSED                                                                                                                [ 33%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_impl PASSED                                                                                                                            [ 33%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_send_impl_error PASSED                                                                                                                      [ 34%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_health_check_success PASSED                                                                                                                 [ 34%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_health_check_failure PASSED                                                                                                                 [ 34%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_is_ready_with_key PASSED                                                                                                                    [ 34%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_is_ready_without_key PASSED                                                                                                                 [ 34%]
tests/test_coverage_boost.py::TestXAIProviderCoverage::test_in_memory_vector_store PASSED                                                                                                               [ 35%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_no_key_raises PASSED                                                                                                              [ 35%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_network_error PASSED                                                                                                              [ 35%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_json_parse_fallback PASSED                                                                                                        [ 35%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_json_parse_empty_fallback PASSED                                                                                                  [ 35%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_output_shape PASSED                                                                                                               [ 35%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_result_shape PASSED                                                                                                               [ 36%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_choices_text PASSED                                                                                                               [ 36%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_choices_message_content PASSED                                                                                                    [ 36%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_impl_json_fallback PASSED                                                                                                              [ 36%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_no_key_raises PASSED                                                                                                                   [ 36%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_test_key PASSED                                                                                                                        [ 36%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_success_text_key PASSED                                                                                                                [ 37%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_success_choices_key PASSED                                                                                                             [ 37%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_http_error PASSED                                                                                                                      [ 37%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_unknown_response PASSED                                                                                                                [ 37%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_send_network_error PASSED                                                                                                                   [ 37%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_chat_success PASSED                                                                                                                         [ 37%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_chat_text_fallback PASSED                                                                                                                   [ 38%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_chat_error PASSED                                                                                                                           [ 38%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_is_ready_with_key PASSED                                                                                                                    [ 38%]
tests/test_coverage_boost.py::TestZAIProviderCoverage::test_is_ready_without_key PASSED                                                                                                                 [ 38%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_init_empty_key_raises PASSED                                                                                                           [ 38%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_init_no_key PASSED                                                                                                                     [ 39%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_ask_success PASSED                                                                                                                     [ 39%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_ask_with_system_prompt PASSED                                                                                                          [ 39%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_ask_empty_content PASSED                                                                                                               [ 39%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_ask_api_error PASSED                                                                                                                   [ 39%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_send_success_message_content PASSED                                                                                                    [ 39%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_send_dict_message PASSED                                                                                                               [ 40%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_send_text_fallback PASSED                                                                                                              [ 40%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_send_exception_returns_mock PASSED                                                                                                     [ 40%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_chat_success PASSED                                                                                                                    [ 40%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_chat_error PASSED                                                                                                                      [ 40%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_embeddings_success PASSED                                                                                                              [ 40%]
tests/test_coverage_boost.py::TestDeepSeekProviderCoverage::test_embeddings_error PASSED                                                                                                                [ 41%]
tests/test_coverage_boost.py::TestStubModules::test_ask_service_top_level PASSED                                                                                                                        [ 41%]
tests/test_coverage_boost.py::TestStubModules::test_contracts PASSED                                                                                                                                    [ 41%]
tests/test_coverage_boost.py::TestStubModules::test_decorators PASSED                                                                                                                                   [ 41%]
tests/test_coverage_boost.py::TestStubModules::test_adapters PASSED                                                                                                                                     [ 41%]
tests/test_coverage_boost.py::TestStubModules::test_provider_contracts PASSED                                                                                                                           [ 41%]
tests/test_coverage_boost.py::TestStubModules::test_provider_decorators PASSED                                                                                                                          [ 42%]
tests/test_coverage_boost.py::TestStubModules::test_openai_module PASSED                                                                                                                                [ 42%]
tests/test_coverage_boost.py::TestStubModules::test_spec PASSED                                                                                                                                         [ 42%]
tests/test_coverage_boost.py::TestStubModules::test_resolver PASSED                                                                                                                                     [ 42%]
tests/test_coverage_boost.py::TestStubModules::test_config_resolve_api_key PASSED                                                                                                                       [ 42%]
tests/test_coverage_boost.py::TestStubModules::test_test_mode PASSED                                                                                                                                    [ 43%]
tests/test_coverage_boost.py::TestCoreAskService::test_ask_delegates_to_chat_provider PASSED                                                                                                            [ 43%]
tests/test_embeddings.py::test_send PASSED                                                                                                                                                              [ 43%]
tests/test_embeddings.py::test_health_check PASSED                                                                                                                                                      [ 43%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_upsert_and_query PASSED                                                                                                                       [ 43%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_upsert_replaces_existing PASSED                                                                                                               [ 43%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_query_empty_db PASSED                                                                                                                         [ 44%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_cosine_similarity_identical PASSED                                                                                                            [ 44%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_cosine_similarity_zero_norm PASSED                                                                                                            [ 44%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_query_top_k_respected PASSED                                                                                                                  [ 44%]
tests/test_gemini_extended.py::TestInMemoryVectorDB::test_upsert_zero_vector_norm PASSED                                                                                                                [ 44%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_mock_path PASSED                                                                                                                           [ 44%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_impl_test_key PASSED                                                                                                                       [ 45%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_impl_test_key_variant PASSED                                                                                                               [ 45%]
tests/test_gemini_extended.py::TestGeminiProvider::test_chunk_text_empty PASSED                                                                                                                         [ 45%]
tests/test_gemini_extended.py::TestGeminiProvider::test_chunk_text_basic PASSED                                                                                                                         [ 45%]
tests/test_gemini_extended.py::TestGeminiProvider::test_chunk_text_short PASSED                                                                                                                         [ 45%]
tests/test_gemini_extended.py::TestGeminiProvider::test_chunk_text_override_params PASSED                                                                                                               [ 45%]
tests/test_gemini_extended.py::TestGeminiProvider::test_invalid_chunk_size_raises PASSED                                                                                                                [ 46%]
tests/test_gemini_extended.py::TestGeminiProvider::test_invalid_chunk_overlap_raises PASSED                                                                                                             [ 46%]
tests/test_gemini_extended.py::TestGeminiProvider::test_missing_api_key_raises PASSED                                                                                                                   [ 46%]
tests/test_gemini_extended.py::TestGeminiProvider::test_provider_name PASSED                                                                                                                            [ 46%]
tests/test_gemini_extended.py::TestGeminiProvider::test_index_document_empty_text PASSED                                                                                                                [ 46%]
tests/test_gemini_extended.py::TestGeminiProvider::test_retrieve_relevant_context_empty PASSED                                                                                                          [ 47%]
tests/test_gemini_extended.py::TestGeminiProvider::test_retrieve_relevant_context_with_results PASSED                                                                                                   [ 47%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_with_rag_no_embedding_model_raises PASSED                                                                                                  [ 47%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_with_rag_no_context PASSED                                                                                                                 [ 47%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_with_rag_with_context_prepend PASSED                                                                                                       [ 47%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_with_rag_with_context_append PASSED                                                                                                        [ 47%]
tests/test_gemini_extended.py::TestGeminiProvider::test_send_with_rag_custom_prefix PASSED                                                                                                              [ 48%]
tests/test_gemini_extended.py::TestGeminiProvider::test_create_embeddings_empty PASSED                                                                                                                  [ 48%]
tests/test_gemini_extended.py::TestGeminiProvider::test_create_embeddings_no_sdk_raises PASSED                                                                                                          [ 48%]
tests/test_gemini_extended.py::TestGeminiProvider::test_query_vector_db_delegates_to_db PASSED                                                                                                          [ 48%]
tests/test_gemini_extended.py::TestGeminiProvider::test_query_vector_db_embedding_fail_raises PASSED                                                                                                    [ 48%]
tests/test_gemini_extended.py::TestGeminiProvider::test_index_document_mismatch_raises PASSED                                                                                                           [ 48%]
tests/test_gemini_extended.py::TestGeminiProvider::test_index_document_vector_db_error_raises PASSED                                                                                                    [ 49%]
tests/test_gemini_extended.py::TestGeminiProvider::test_health_check_mock_path PASSED                                                                                                                   [ 49%]
tests/test_gemini_extended.py::TestGeminiSendImpl::test_send_impl_with_text_response PASSED                                                                                                             [ 49%]
tests/test_gemini_extended.py::TestGeminiSendImpl::test_send_impl_exception_returns_fallback PASSED                                                                                                     [ 49%]
tests/test_gemini_extended.py::TestGeminiSendImpl::test_send_impl_no_text_returns_fallback PASSED                                                                                                       [ 49%]
tests/test_gemini_extended.py::TestGeminiSendImpl::test_health_check_legacy_success PASSED                                                                                                              [ 50%]
tests/test_gemini_extended.py::TestGeminiSendImpl::test_health_check_legacy_failure PASSED                                                                                                              [ 50%]
tests/test_gemini_extended.py::TestGeminiSendImpl::test_health_check_no_text PASSED                                                                                                                     [ 50%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_cosine_zero_norm PASSED                                                                                                                    [ 50%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_cosine_normal PASSED                                                                                                                       [ 50%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_query_empty PASSED                                                                                                                         [ 50%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_query_all_results PASSED                                                                                                                   [ 51%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_query_heap_branch PASSED                                                                                                                   [ 51%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_chunk_empty PASSED                                                                                                                         [ 51%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_chunk_single PASSED                                                                                                                        [ 51%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_chunk_multiple PASSED                                                                                                                      [ 51%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_embeddings_empty PASSED                                                                                                                    [ 51%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_embeddings_api_missing PASSED                                                                                                              [ 52%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_embeddings_no_data PASSED                                                                                                                  [ 52%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_embeddings_missing_vector PASSED                                                                                                           [ 52%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_embeddings_exception PASSED                                                                                                                [ 52%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_index_document_no_chunks PASSED                                                                                                            [ 52%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_index_document_count_mismatch PASSED                                                                                                       [ 52%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_index_document_upsert_exception PASSED                                                                                                     [ 53%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_query_embedding_failure PASSED                                                                                                             [ 53%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_query_vector_db_exception PASSED                                                                                                           [ 53%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_context_none PASSED                                                                                                                        [ 53%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_context_join PASSED                                                                                                                        [ 53%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_send_with_rag_no_embedding_model PASSED                                                                                                    [ 54%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_send_with_rag_no_context PASSED                                                                                                            [ 54%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_send_with_rag_append_context PASSED                                                                                                        [ 54%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_send_mock PASSED                                                                                                                           [ 54%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_health_exception PASSED                                                                                                                    [ 54%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_health_dict_response PASSED                                                                                                                [ 54%]
tests/test_gemini_provider.py::TestGeminiCoverageBoost::test_health_empty PASSED                                                                                                                        [ 55%]
tests/test_hello_world.py::test_hello_world PASSED                                                                                                                                                      [ 55%]
tests/test_hello_world.py::test_cli_valid_prompt PASSED                                                                                                                                                 [ 55%]
tests/test_hello_world.py::test_main_invalid_args PASSED                                                                                                                                                [ 55%]
tests/test_hello_world.py::test_main_missing_prompt PASSED                                                                                                                                              [ 55%]
tests/test_imports.py::test_import_all_modules PASSED                                                                                                                                                   [ 55%]
tests/test_loader_coverage.py::test_loader_import_exec PASSED                                                                                                                                           [ 56%]
tests/test_monitoring.py::test_monitoring_module_imports PASSED                                                                                                                                         [ 56%]
tests/test_monitoring.py::test_monitoring_has_expected_attributes PASSED                                                                                                                                [ 56%]
tests/test_monitoring.py::test_monitoring_functions_execute_safely PASSED                                                                                                                               [ 56%]
tests/test_monitoring.py::test_monitoring_module_is_import_safe_multiple_times PASSED                                                                                                                   [ 56%]
tests/test_monitoring_coverage.py::test_monitoring_basic_calls PASSED                                                                                                                                   [ 56%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_success_rate_zero_requests PASSED                                                                                                      [ 57%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_success_rate_all_success PASSED                                                                                                        [ 57%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_success_rate_partial_failures PASSED                                                                                                   [ 57%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_success_rate_all_failures PASSED                                                                                                       [ 57%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_avg_latency_zero_requests PASSED                                                                                                       [ 57%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_avg_latency_with_data PASSED                                                                                                           [ 58%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_hallucination_rate_zero PASSED                                                                                                         [ 58%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_hallucination_rate_with_data PASSED                                                                                                    [ 58%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_avg_embedding_latency_zero PASSED                                                                                                      [ 58%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_avg_embedding_latency_with_data PASSED                                                                                                 [ 58%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_avg_vector_query_latency_zero PASSED                                                                                                   [ 58%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_avg_vector_query_latency_with_data PASSED                                                                                              [ 59%]
tests/test_monitoring_extended.py::TestModelQualityMetrics::test_rag_counters_default_zero PASSED                                                                                                       [ 59%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_record_request_noop PASSED                                                                                                                     [ 59%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_record_failure_noop PASSED                                                                                                                     [ 59%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_record_latency_noop PASSED                                                                                                                     [ 59%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_record_chunks_noop PASSED                                                                                                                      [ 59%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_record_embedding_noop PASSED                                                                                                                   [ 60%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_record_vector_query_noop PASSED                                                                                                                [ 60%]
tests/test_monitoring_extended.py::TestMetricsNoOp::test_close_noop PASSED                                                                                                                              [ 60%]
tests/test_monitoring_extended.py::TestNoopMetric::test_labels_returns_self PASSED                                                                                                                      [ 60%]
tests/test_monitoring_extended.py::TestNoopMetric::test_inc_is_noop PASSED                                                                                                                              [ 60%]
tests/test_monitoring_extended.py::TestNoopMetric::test_set_is_noop PASSED                                                                                                                              [ 60%]
tests/test_monitoring_extended.py::TestTracer::test_tracer_disabled_span_yields_none PASSED                                                                                                             [ 61%]
tests/test_monitoring_extended.py::TestTracer::test_tracer_shutdown_no_provider PASSED                                                                                                                  [ 61%]
tests/test_monitoring_extended.py::TestTracer::test_tracer_shutdown_with_provider PASSED                                                                                                                [ 61%]
tests/test_monitoring_extended.py::TestTracer::test_tracer_init_creates_instance PASSED                                                                                                                 [ 61%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_start_trace PASSED                                                                                                                      [ 61%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_start_trace_default PASSED                                                                                                              [ 62%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_end_trace PASSED                                                                                                                        [ 62%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_end_trace_default PASSED                                                                                                                [ 62%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_log_event PASSED                                                                                                                        [ 62%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_log_event_no_data PASSED                                                                                                                [ 62%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_record_metric PASSED                                                                                                                    [ 62%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_telemetry_track PASSED                                                                                                                  [ 63%]
tests/test_monitoring_extended.py::TestPublicTelemetryAPI::test_telemetry_multiple_events PASSED                                                                                                        [ 63%]
tests/test_monitoring_extended.py::TestSafeMetricHelpers::test_safe_counter_without_prometheus PASSED                                                                                                   [ 63%]
tests/test_monitoring_extended.py::TestSafeMetricHelpers::test_safe_gauge_without_prometheus PASSED                                                                                                     [ 63%]
tests/test_monitoring_extended.py::TestSafeMetricHelpers::test_find_existing_metric_without_prom_core PASSED                                                                                            [ 63%]
tests/test_monitoring_extended.py::TestNoopMetricExtra::test_noop_metric_methods PASSED                                                                                                                 [ 63%]
tests/test_monitoring_extended.py::TestSafeMetricCreation::test_find_existing_metric_not_found PASSED                                                                                                   [ 64%]
tests/test_monitoring_extended.py::TestTelemetryExtra::test_track_many PASSED                                                                                                                           [ 64%]
tests/test_monitoring_extended.py::TestTracerExtra::test_span_enabled PASSED                                                                                                                            [ 64%]
tests/test_monitoring_extended.py::TestTracerExtra::test_shutdown_provider_exception PASSED                                                                                                             [ 64%]
tests/test_prompt_corrector.py::test_prompt_corrector_basic PASSED                                                                                                                                      [ 64%]
tests/test_prompt_corrector.py::test_prompt_corrector_inputs[] PASSED                                                                                                                                   [ 64%]
tests/test_prompt_corrector.py::test_prompt_corrector_inputs[   ] PASSED                                                                                                                                [ 65%]
tests/test_prompt_corrector.py::test_prompt_corrector_inputs[None] PASSED                                                                                                                               [ 65%]
tests/test_prompt_corrector.py::test_prompt_corrector_inputs[hello world] PASSED                                                                                                                        [ 65%]
tests/test_provider_contract.py::test_factory_build_provider_success PASSED                                                                                                                             [ 65%]
tests/test_provider_contract.py::test_no_import_time_crashes PASSED                                                                                                                                     [ 65%]
tests/test_provider_contract.py::test_factory_unknown_provider PASSED                                                                                                                                   [ 66%]
tests/test_provider_contract.py::test_factory_build_provider_none_kwargs PASSED                                                                                                                         [ 66%]
tests/test_provider_contracts.py::test_all_chat_providers_have_ask PASSED                                                                                                                               [ 66%]
tests/test_provider_contracts.py::test_registry_builds PASSED                                                                                                                                           [ 66%]
tests/test_provider_contracts.py::test_fake_cohere PASSED                                                                                                                                               [ 66%]
tests/test_provider_contracts.py::test_fake_deepseek PASSED                                                                                                                                             [ 66%]
tests/test_provider_contracts.py::test_fake_xai PASSED                                                                                                                                                  [ 67%]
tests/test_provider_contracts.py::test_fake_zai PASSED                                                                                                                                                  [ 67%]
tests/test_provider_unified.py::test_gemini_provider_basic PASSED                                                                                                                                       [ 67%]
tests/test_provider_unified.py::test_cohere_provider_basic PASSED                                                                                                                                       [ 67%]
tests/test_provider_unified.py::test_xai_provider PASSED                                                                                                                                                [ 67%]
tests/test_provider_unified.py::test_zai_provider PASSED                                                                                                                                                [ 67%]
tests/test_provider_unified.py::test_deepseek_provider PASSED                                                                                                                                           [ 68%]
tests/test_providers.py::test_simple_openai_provider_init PASSED                                                                                                                                        [ 68%]
tests/test_providers.py::test_simple_openai_provider_ask PASSED                                                                                                                                         [ 68%]
tests/test_providers.py::test_echo_provider PASSED                                                                                                                                                      [ 68%]
tests/test_providers.py::test_openai_provider_send PASSED                                                                                                                                               [ 68%]
tests/test_providers.py::test_gemini_provider PASSED                                                                                                                                                    [ 68%]
tests/test_providers.py::test_cohere_provider PASSED                                                                                                                                                    [ 69%]
tests/test_providers.py::test_deepseek_provider PASSED                                                                                                                                                  [ 69%]
tests/test_providers.py::test_perplexity_provider PASSED                                                                                                                                                [ 69%]
tests/test_providers.py::test_xai_provider PASSED                                                                                                                                                       [ 69%]
tests/test_providers.py::test_cohere_api PASSED                                                                                                                                                         [ 69%]
tests/test_providers.py::test_deepseek_timeout PASSED                                                                                                                                                   [ 70%]
tests/test_providers.py::test_deepseek_health_check PASSED                                                                                                                                              [ 70%]
tests/test_providers.py::test_deepseek_embeddings PASSED                                                                                                                                                [ 70%]
tests/test_providers.py::test_deepseek_chat_response PASSED                                                                                                                                             [ 70%]
tests/test_providers.py::test_zai_success PASSED                                                                                                                                                        [ 70%]
tests/test_providers.py::test_zai_error PASSED                                                                                                                                                          [ 70%]
tests/test_providers.py::test_cohere_clear_index PASSED                                                                                                                                                 [ 71%]
tests/test_providers.py::test_cohere_retrieve_empty PASSED                                                                                                                                              [ 71%]
tests/test_providers_extended.py::TestBaseProvider::test_send_raises_not_implemented PASSED                                                                                                             [ 71%]
tests/test_providers_extended.py::TestBaseProvider::test_ask_delegates_to_send PASSED                                                                                                                   [ 71%]
tests/test_providers_extended.py::TestBaseProvider::test_init_stores_api_key_and_model PASSED                                                                                                           [ 71%]
tests/test_providers_extended.py::TestEchoProvider::test_send PASSED                                                                                                                                    [ 71%]
tests/test_providers_extended.py::TestEchoProvider::test_ask PASSED                                                                                                                                     [ 72%]
tests/test_providers_extended.py::TestEchoProvider::test_provider_name PASSED                                                                                                                           [ 72%]
tests/test_providers_extended.py::TestEchoProviderModule::test_send_returns_echoed PASSED                                                                                                               [ 72%]
tests/test_providers_extended.py::TestEchoProviderModule::test_ask_returns_echoed PASSED                                                                                                                [ 72%]
tests/test_providers_extended.py::TestEchoProviderModule::test_has_provider_name_attribute PASSED                                                                                                       [ 72%]
tests/test_providers_extended.py::TestRegistry::test_register_and_retrieve_provider PASSED                                                                                                              [ 72%]
tests/test_providers_extended.py::TestRegistry::test_register_provider_as_decorator PASSED                                                                                                              [ 73%]
tests/test_providers_extended.py::TestRegistry::test_register_chat_provider PASSED                                                                                                                      [ 73%]
tests/test_providers_extended.py::TestRegistry::test_list_providers_sorted PASSED                                                                                                                       [ 73%]
tests/test_providers_extended.py::TestRegistry::test_build_provider_unknown_raises PASSED                                                                                                               [ 73%]
tests/test_providers_extended.py::TestRegistry::test_get_chat_provider_unknown_raises PASSED                                                                                                            [ 73%]
tests/test_providers_extended.py::TestRegistry::test_ensure_initialized_is_idempotent PASSED                                                                                                            [ 74%]
tests/test_providers_extended.py::TestAutoProvider::test_send_with_custom_fallback PASSED                                                                                                               [ 74%]
tests/test_providers_extended.py::TestAutoProvider::test_ask_delegates_to_send PASSED                                                                                                                   [ 74%]
tests/test_providers_extended.py::TestAutoProvider::test_send_skips_missing_provider PASSED                                                                                                             [ 74%]
tests/test_providers_extended.py::TestAutoProvider::test_send_all_fail_raises PASSED                                                                                                                    [ 74%]
tests/test_providers_extended.py::TestAutoProvider::test_default_fallback_order PASSED                                                                                                                  [ 74%]
tests/test_providers_extended.py::TestBuiltinsOpenAIProvider::test_init_sets_timeout PASSED                                                                                                             [ 75%]
tests/test_providers_extended.py::TestBuiltinsOpenAIProvider::test_send_no_api_key_raises PASSED                                                                                                        [ 75%]
tests/test_providers_extended.py::TestBuiltinsOpenAIProvider::test_send_import_error_raises PASSED                                                                                                      [ 75%]
tests/test_providers_extended.py::TestBuiltinsOpenAIProvider::test_send_success PASSED                                                                                                                  [ 75%]
tests/test_providers_extended.py::TestBuiltinsOpenAIProvider::test_send_api_error_raises PASSED                                                                                                         [ 75%]
tests/test_providers_extended.py::TestBuiltinsOpenAIProvider::test_send_invalid_response_raises PASSED                                                                                                  [ 75%]
tests/test_providers_extended.py::TestBuiltinsOpenAICompatibleProvider::test_get_openai_client_no_key_raises PASSED                                                                                     [ 76%]
tests/test_providers_extended.py::TestBuiltinsOpenAICompatibleProvider::test_get_openai_client_import_error PASSED                                                                                      [ 76%]
tests/test_providers_extended.py::TestBuiltinsOpenAICompatibleProvider::test_send_success PASSED                                                                                                        [ 76%]
tests/test_providers_extended.py::TestBuiltinsOpenAICompatibleProvider::test_send_api_failure_raises PASSED                                                                                             [ 76%]
tests/test_providers_extended.py::TestBuiltinsOpenAICompatibleProvider::test_send_empty_content_raises PASSED                                                                                           [ 76%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_perplexity_provider_name PASSED                                                                                                        [ 77%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_deepseek_api_base PASSED                                                                                                               [ 77%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_groq_api_base PASSED                                                                                                                   [ 77%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_openrouter_api_base PASSED                                                                                                             [ 77%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_together_api_base PASSED                                                                                                               [ 77%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_fireworks_api_base PASSED                                                                                                              [ 77%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_xai_api_base PASSED                                                                                                                    [ 78%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_gemini_api_base PASSED                                                                                                                 [ 78%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_zai_all_response_shapes PASSED                                                                                                         [ 78%]
tests/test_providers_extended.py::TestBuiltinsSubProviders::test_zai_http_error PASSED                                                                                                                  [ 78%]
tests/test_providers_extended.py::TestBuiltinsCohereProvider::test_send_no_api_key_raises PASSED                                                                                                        [ 78%]
tests/test_providers_extended.py::TestBuiltinsCohereProvider::test_send_import_error_raises PASSED                                                                                                      [ 78%]
tests/test_providers_extended.py::TestBuiltinsCohereProvider::test_send_success PASSED                                                                                                                  [ 79%]
tests/test_providers_extended.py::TestBuiltinsCohereProvider::test_send_api_failure_raises PASSED                                                                                                       [ 79%]
tests/test_providers_extended.py::TestBuiltinsCohereProvider::test_send_empty_response_raises PASSED                                                                                                    [ 79%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_send_success PASSED                                                                                                                    [ 79%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_send_no_choices_raises PASSED                                                                                                          [ 79%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_send_api_error_raises PASSED                                                                                                           [ 79%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_ask_delegates_to_send PASSED                                                                                                           [ 80%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_health_check_success PASSED                                                                                                            [ 80%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_health_check_failure PASSED                                                                                                            [ 80%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_ensure_key_missing_raises PASSED                                                                                                       [ 80%]
tests/test_providers_extended.py::TestOpenAIProviderModule::test_missing_api_key_on_init_raises PASSED                                                                                                  [ 80%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_send_mock_key PASSED                                                                                                               [ 81%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_send_non_rag PASSED                                                                                                                [ 81%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cosine_similarity_identical PASSED                                                                                                 [ 81%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cosine_similarity_orthogonal PASSED                                                                                                [ 81%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_retrieve_empty PASSED                                                                                                              [ 81%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_clear_index PASSED                                                                                                                 [ 81%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_query_documents_without_rag_raises PASSED                                                                                          [ 82%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_upsert_documents_empty PASSED                                                                                                      [ 82%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_embed_empty PASSED                                                                                                                 [ 82%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_send_with_rag_no_context PASSED                                                                                                    [ 82%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_send_with_rag_with_context PASSED                                                                                                  [ 82%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_requires_api_key PASSED                                                                                                     [ 82%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_import_failure PASSED                                                                                                       [ 83%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_chat_wraps_exception PASSED                                                                                                 [ 83%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_chat_client_response PASSED                                                                                                 [ 83%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_embed_empty_returns_empty PASSED                                                                                                   [ 83%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_embed_success PASSED                                                                                                               [ 83%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_send_without_rag_calls_chat PASSED                                                                                                 [ 83%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_send_with_rag_context PASSED                                                                                                       [ 84%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_chat_returns_mock_when_client_none PASSED                                                                                          [ 84%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_upsert_documents_without_metadata PASSED                                                                                    [ 84%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_upsert_embedding_mismatch PASSED                                                                                            [ 84%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_retrieve_scores_sorted PASSED                                                                                               [ 84%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_query_documents_rag_enabled PASSED                                                                                                 [ 85%]
tests/test_providers_extended.py::TestCohereProviderStandalone::test_cohere_import_failure_test_key PASSED                                                                                              [ 85%]
tests/test_providers_extended.py::TestPerplexityProvider::test_send_success PASSED                                                                                                                      [ 85%]
tests/test_providers_extended.py::TestPerplexityProvider::test_send_failure_raises PASSED                                                                                                               [ 85%]
tests/test_providers_extended.py::TestPerplexityProvider::test_send_no_choices_returns_empty PASSED                                                                                                     [ 85%]
tests/test_providers_extended.py::TestPerplexityProvider::test_send_no_message_returns_empty PASSED                                                                                                     [ 85%]
tests/test_registry.py::test_build_provider_invalid PASSED                                                                                                                                              [ 86%]
tests/test_registry.py::test_build_provider_normal PASSED                                                                                                                                               [ 86%]
tests/test_registry.py::test_hallucination_empty_response PASSED                                                                                                                                        [ 86%]
tests/test_registry.py::test_hallucination_short_response PASSED                                                                                                                                        [ 86%]
tests/test_registry.py::test_hallucination_single_pattern PASSED                                                                                                                                        [ 86%]
tests/test_registry.py::test_hallucination_all_patterns PASSED                                                                                                                                          [ 86%]
tests/test_registry.py::test_hallucination_placeholder_only PASSED                                                                                                                                      [ 87%]
tests/test_registry.py::test_hallucination_clean_response PASSED                                                                                                                                        [ 87%]
tests/test_registry.py::test_response_validator_empty PASSED                                                                                                                                            [ 87%]
tests/test_registry.py::test_response_validator_short PASSED                                                                                                                                            [ 87%]
tests/test_registry.py::test_response_validator_valid PASSED                                                                                                                                            [ 87%]
tests/test_resilience.py::test_resilience_fallback PASSED                                                                                                                                               [ 87%]
tests/test_resilience_cache.py::test_cache_set_get_delete PASSED                                                                                                                                        [ 88%]
tests/test_resilience_cache.py::test_cache_clear PASSED                                                                                                                                                 [ 88%]
tests/test_resilience_extended.py::test_execute_primary_success PASSED                                                                                                                                  [ 88%]
tests/test_resilience_extended.py::test_execute_fallback_on_failure PASSED                                                                                                                              [ 88%]
tests/test_resilience_extended.py::test_rate_limiter PASSED                                                                                                                                             [ 88%]
tests/test_resilience_extended.py::test_circuit_breaker_opens PASSED                                                                                                                                    [ 89%]
tests/test_resilience_extended.py::test_cache_memory_operations PASSED                                                                                                                                  [ 89%]
tests/test_resilience_extended.py::test_retry_engine_success PASSED                                                                                                                                     [ 89%]
tests/test_resilience_extended.py::test_retry_engine_retry_then_success PASSED                                                                                                                          [ 89%]
tests/test_resilience_extended.py::test_retry_engine_failure PASSED                                                                                                                                     [ 89%]
tests/test_resilience_extended.py::test_rate_limiter_acquire_timeout PASSED                                                                                                                             [ 89%]
tests/test_resilience_extended.py::test_circuit_breaker_reopens PASSED                                                                                                                                  [ 90%]
tests/test_resilience_extended.py::test_async_retry_engine_success PASSED                                                                                                                               [ 90%]
tests/test_resilience_extended.py::test_async_retry_engine_failure PASSED                                                                                                                               [ 90%]
tests/test_resilience_extended.py::test_circuit_breaker_success PASSED                                                                                                                                  [ 90%]
tests/test_resilience_extended.py::test_retry_engine_decorator PASSED                                                                                                                                   [ 90%]
tests/test_resilience_extended.py::test_cache_expiry PASSED                                                                                                                                             [ 90%]
tests/test_resilience_extended.py::test_cache_delete_clear PASSED                                                                                                                                       [ 91%]
tests/test_resilience_extended.py::test_retry_engine_rejects_async PASSED                                                                                                                               [ 91%]
tests/test_resilience_extended.py::test_circuit_breaker_failure PASSED                                                                                                                                  [ 91%]
tests/test_resilience_extended.py::test_rate_limiter_refill PASSED                                                                                                                                      [ 91%]
tests/test_resilience_extended.py::test_async_retry_decorator_rejects_sync PASSED                                                                                                                       [ 91%]
tests/test_resilience_extended.py::test_retry_engine_exhausted PASSED                                                                                                                                   [ 91%]
tests/test_resilience_extended.py::test_async_retry_decorator PASSED                                                                                                                                    [ 92%]
tests/test_resilience_extended.py::test_rate_limiter_acquire_success PASSED                                                                                                                             [ 92%]
tests/test_resilience_extended.py::test_execute_with_fallback_exception_path PASSED                                                                                                                     [ 92%]
tests/test_resilience_extended.py::test_cache_eviction PASSED                                                                                                                                           [ 92%]
tests/test_resilience_extended.py::test_retry_engine_decorator_executes PASSED                                                                                                                          [ 92%]
tests/test_resilience_extended.py::test_retry_engine_retry_filter PASSED                                                                                                                                [ 93%]
tests/test_resilience_extended.py::test_circuit_breaker_async_wrap PASSED                                                                                                                               [ 93%]
tests/test_resilience_extended.py::test_rate_limiter_constructor_values PASSED                                                                                                                          [ 93%]
tests/test_resilience_extended.py::test_retry_engine_rejects_coroutine PASSED                                                                                                                           [ 93%]
tests/test_resilience_extended.py::test_circuit_breaker_async_wrap_threshold PASSED                                                                                                                     [ 93%]
tests/test_resilience_extended.py::test_circuit_breaker_async_wrap_records_failure PASSED                                                                                                               [ 93%]
tests/test_resilience_extended.py::test_async_retry_engine_decorator_all_attempts_fail PASSED                                                                                                           [ 94%]
tests/test_resilience_extended.py::test_execute_with_fallback_returns_none_without_fallback PASSED                                                                                                      [ 94%]
tests/test_resilience_retry.py::test_retry_success PASSED                                                                                                                                               [ 94%]
tests/test_resilience_retry.py::test_retry_failure PASSED                                                                                                                                               [ 94%]
tests/test_secrets.py::test_secret_chunk_text PASSED                                                                                                                                                    [ 94%]
tests/test_secrets.py::test_chunk_text_forward_progress_branch PASSED                                                                                                                                   [ 94%]
tests/test_secrets.py::test_chunk_text_invalid_chunk_size_zero PASSED                                                                                                                                   [ 95%]
tests/test_secrets.py::test_chunk_text_invalid_chunk_size_type PASSED                                                                                                                                   [ 95%]
tests/test_secrets.py::test_chunk_text_invalid_overlap_negative PASSED                                                                                                                                  [ 95%]
tests/test_secrets.py::test_chunk_text_invalid_overlap_type PASSED                                                                                                                                      [ 95%]
tests/test_secrets.py::test_chunk_text_overlap_greater_than_chunk_size PASSED                                                                                                                           [ 95%]
tests/test_secrets.py::test_chunk_text_empty_returns_empty_list PASSED                                                                                                                                  [ 95%]
tests/test_secrets.py::test_chunk_text_short_text_returns_single_chunk PASSED                                                                                                                           [ 96%]
tests/test_validation.py::test_hallucination_empty_response PASSED                                                                                                                                      [ 96%]
tests/test_validation.py::test_hallucination_suspicious_phrase PASSED                                                                                                                                   [ 96%]
tests/test_validation.py::test_hallucination_todo PASSED                                                                                                                                                [ 96%]
tests/test_validation.py::test_response_validator_empty PASSED                                                                                                                                          [ 96%]
tests/test_validation.py::test_register_provider_direct_registration PASSED                                                                                                                             [ 97%]
tests/test_validation.py::test_register_provider_decorator_registration PASSED                                                                                                                          [ 97%]
tests/test_validation.py::test_register_chat_provider_registers_both_maps PASSED                                                                                                                        [ 97%]
tests/test_validation.py::test_get_chat_provider_success PASSED                                                                                                                                         [ 97%]
tests/test_validation.py::test_get_chat_provider_unknown PASSED                                                                                                                                         [ 97%]
tests/test_validation.py::test_list_providers_sorted PASSED                                                                                                                                             [ 97%]
tests/test_validation.py::test_build_provider_success PASSED                                                                                                                                            [ 98%]
tests/test_validation.py::test_build_provider_unknown PASSED                                                                                                                                            [ 98%]
tests/test_validation.py::test_provider_registry_getitem_returns_registered_class PASSED                                                                                                                [ 98%]
tests/test_xai_provider_all.py::test_xai_provider_all PASSED                                                                                                                                            [ 98%]
tests/test_xai_provider_error.py::test_xai_provider_empty_response PASSED                                                                                                                               [ 98%]
tests/test_zai_provider.py::test_zai_provider_init PASSED                                                                                                                                               [ 98%]
tests/test_zai_provider.py::test_zai_provider_missing_key PASSED                                                                                                                                        [ 99%]
tests/test_zai_provider.py::test_zai_provider_send_success_text PASSED                                                                                                                                  [ 99%]
tests/test_zai_provider.py::test_zai_provider_send_success_choices PASSED                                                                                                                               [ 99%]
tests/test_zai_provider.py::test_zai_provider_network_error PASSED                                                                                                                                      [ 99%]
tests/test_zai_provider.py::test_zai_provider_http_error PASSED                                                                                                                                         [ 99%]
tests/test_zai_provider.py::test_zai_success PASSED                                                                                                                                                     [100%]

=============================================================================================== tests coverage ================================================================================================
______________________________________________________________________________ coverage: platform darwin, python 3.13.14-final-0 ______________________________________________________________________________

Name                                          Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------------------------
src/ai_cli/ai_chat.py                            51      5     30      7    83%   90, 97, 103->126, 108-110, 112->117, 114->117, 118->121
src/ai_cli/cli.py                               318     31    114     17    88%   196->198, 198->200, 200->202, 294, 371-373, 376, 393, 402-403, 426-444, 453, 477-481, 495, 538-541, 545-547, 589, 606->615, 616, 625, 650
src/ai_cli/core/prompt_corrector.py              93      0     28      1    99%   101->108
src/ai_cli/core/resilience.py                   168      4     52      4    96%   99, 173->179, 176->179, 184, 238, 242
src/ai_cli/core/service/ask_service.py            7      0      2      1    89%   13->16
src/ai_cli/plugins/builtins.py                  139     22     12      1    85%   24-29, 55-56, 70-76, 89, 105-106, 117, 125, 133, 141, 149, 157, 165, 173, 180-185, 211-212
src/ai_cli/providers/auto_provider.py            41      4      8      2    88%   40-41, 51-52
src/ai_cli/providers/cohere_provider.py          90      1     32      1    98%   167
src/ai_cli/providers/contracts.py                11      2      0      0    82%   10, 15
src/ai_cli/providers/deepseek_provider.py        78      6     28      6    87%   63-64, 129->134, 135->138, 139, 164-167
src/ai_cli/providers/gemini_provider.py         192     21     56      6    89%   36, 46, 56, 67, 76-82, 225-227, 251, 274, 315->320, 348->351, 372-373, 451-452, 542
src/ai_cli/providers/openai_provider.py          56      4     14      3    90%   35, 62, 66, 93
src/ai_cli/providers/perplexity_provider.py      26      2      6      1    91%   23, 50
src/ai_cli/providers/xAI_provider.py             71      6     14      1    92%   39-44, 85-89
src/ai_cli/providers/zAI_provider.py            102      7     42      6    91%   14-15, 75->80, 83, 120-121, 134->157, 151->157, 153->157, 160-161
src/ai_cli/rag/pipeline.py                       20      1      6      1    92%   40
src/ai_cli/telemetry/monitoring.py              280     55     78     11    80%   79-88, 108, 112->116, 118-122, 136-142, 154-160, 174-183, 192->exit, 202-204, 223->exit, 302->exit, 312-319, 331-332, 340-341, 351-352, 367-368, 382->exit, 386-387, 413-414
src/ai_cli/utils/secrets.py                      32      1     22      3    93%   23, 34->47, 37->39
-----------------------------------------------------------------------------------------
TOTAL                                          2080    172    602     72    91%

28 files skipped due to complete coverage.
============================================================================================= 574 passed in 8.81s =============================================================================================

---

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — CLI workflows and examples
- [docs/API.md](docs/API.md) — Python API reference
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — Contributing guide

---

## License

MIT License — see [LICENSE](LICENSE) for details.