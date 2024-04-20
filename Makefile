lint:
	flake8 *.py
fmt:
	@# testing excludes:
	@#black --exclude='third_party|venv' --check --verbose .
	black --exclude='third_party|venv' .
	clang-format -i $(shell find ./agent_fn -name '*.cu' -or -name '*.cuh')
.PHONY: fmt
