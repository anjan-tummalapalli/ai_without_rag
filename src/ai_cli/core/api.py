from types import SimpleNamespace

def build_provider(name: str, model: str = None):
	"""
	Minimal local fallback for build_provider to avoid NameError.
	Returns a SimpleNamespace with name and model attributes.
	Replace this with the real implementation or import from the provider module.
	"""
	return SimpleNamespace(name=name, model=model)

# Provide a default provider value to avoid NameError; override where available.
provider = None

# Replace in ask()
ai_provider = build_provider(name=provider, model=None)
# Replace in rag_query()
emb_provider = build_provider(name=provider, model=None)
gen_provider = build_provider(name=provider, model=None)