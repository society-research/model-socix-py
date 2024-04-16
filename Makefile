lint:
	flake8 *.py
fmt:
	@# testing excludes:
	@#black --exclude='third_party|venv' --check --verbose .
	black --exclude='third_party|venv' .
.PHONY: fmt