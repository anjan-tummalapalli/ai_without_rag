from ai_cli.rag.vector_store import InMemoryVectorStore


def test_vector_store_add_search():

    store = InMemoryVectorStore(dim=3)

    store.add(
        text="hello",
        embedding=[1,0,0],
        metadata={"x":1}
    )

    result = store.search(
        [1,0,0],
        top_k=1
    )

    assert len(result) == 1


def test_vector_store_empty():

    store = InMemoryVectorStore(dim=3)

    result = store.search(
        [1,0,0],
        top_k=5
    )

    assert result == []
