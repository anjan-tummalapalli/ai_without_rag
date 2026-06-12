from ai_cli.rag.vector_store import InMemoryVectorStore


def test_vector_store_init():
    store = InMemoryVectorStore()
    assert store is not None


def test_vector_store_add_search():
    store = InMemoryVectorStore()
    store.add(
        "hello world",
        [0.1, 0.2, 0.3]
    )
    result = store.search(
        [0.1, 0.2, 0.3],
        k=1
    )
    assert result is not None
