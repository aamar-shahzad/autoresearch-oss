.PHONY: install test demo example use-case platform docker
install: ; pip install -e ".[dev,platform]"
test: ; python -m pytest -q
demo: ; python -m autoresearch_oss quickstart
example: ; python examples/make_example.py && python -m autoresearch_oss train --config examples/config.json --promote
use-case: ; python examples/run_maintenance_use_case.py --fresh
platform: ; pip install -e ".[platform]" && echo 'Run: autoresearch platform api --db myproj/registry.db --config myproj/autoresearch.json'
docker: ; docker build -t autoresearch-oss . && docker run --rm autoresearch-oss
