.PHONY: install test demo eval

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/python -m pytest -q

demo:
	.venv/bin/python -m agent_research_pipeline.demo

eval:
	.venv/bin/python -m agent_research_pipeline.evals
