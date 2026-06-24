from unittest.mock import MagicMock

p.client.embeddings.create = MagicMock(
    return_value=type(
        "R",
        (),
        {
            "data": [
                type(
                    "D",
                    (),
                    {"embedding": [0.1,0.2]}
                )()
            ]
        }
    )()
)

p.client.chat.completions.create = MagicMock(
    return_value=mock_response
)