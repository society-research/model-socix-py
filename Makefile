lint:
	flake8 *.py
.PHONY: lint
fmt:
	@# testing excludes:
	@#black --exclude='third_party|venv' --check --verbose .
	black --exclude='third_party|venv' .
	clang-format -i $(shell find ./agent_fn -name '*.cu' -or -name '*.cuh')
.PHONY: fmt
test:
	while true; do inotifywait -e modify,close_write,moved_to,move,create,delete $(shell find -maxdepth 2 -name '*.cu' -or -name '*.py'); pytest .; done
.PHONY: test
testv:
	while true; do inotifywait -e modify,close_write,moved_to,move,create,delete $(shell find -maxdepth 2 -name '*.cu' -or -name '*.py'); pytest . -vv; done
.PHONY: testv
test-run:
	while true; do inotifywait -e modify,close_write,moved_to,move,create,delete $(shell find -maxdepth 2 -name '*.cu' -or -name '*.py'); python src/sx.py -s 2; done
.PHONY: test-run
analysis:
	python3 ./src/analysis.py
.PHONY: analysis
analysis-w:
	while true; do inotifywait -e modify,close_write,moved_to,move,create,delete $(shell find -maxdepth 2 -name '*.cu' -or -name '*.py'); make analysis; done
.PHONY: analysis
get-fonts:
	mkdir -p ~/.local/share/fonts/Libertinus
	wget -O libertinus.zip "https://www.fontsquirrel.com/fonts/download/libertinus"
	unzip -d ~/.local/share/fonts/Libertinus libertinus.zip
	rm libertinus.zip
.PHONY: get-fonts
