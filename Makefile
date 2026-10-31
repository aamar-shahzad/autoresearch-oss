.PHONY: install test demo example use-case docker
install: ; pip install -e ".[dev]"
test: ; python -m pytest -q
demo: ; python -m autoresearch_oss quickstart
example: ; python examples/make_example.py && python -m autoresearch_oss train --config examples/config.json --promote
use-case: ; python examples/run_maintenance_use_case.py --fresh
docker: ; docker build -t autoresearch-oss . && docker run --rm autoresearch-oss
