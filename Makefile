.PHONY: help

help: ## Shows this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

test: ## Run the tests
	@pytest ./tests -vv -s;

test-report: ## Run the tests and return a coverage report
	@pytest --cov-report term-missing --cov=bbaug tests/
