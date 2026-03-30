# life-core

Moteur IA backend de FineFab (routeur LLM, RAG, cache, orchestration).

## Role
- Porter les capacites coeur ex-mascarade (inference et orchestration).
- Exposer des services backend stables pour `life-reborn` et `makelife-cad`.
- Fournir une couche de resilence (fallback, circuit breaker, cache).

## Stack
- Python 3.12+
- FastAPI
- pytest

## Structure cible
- `finefab_core/` et `life_core/`: services et modules coeur
- `tests/`: tests unitaires et integration
- `pyproject.toml`: dependances et outillage

## Demarrage rapide
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pytest -q
```

## Roadmap immediate
- Consolidation routeur multi-provider.
- Stabilisation pipeline RAG + cache multi-tier.
- Gate CI coverage >= 80%.
