CSPELL := npx cspell
TEXTLINT := npx textlint

CONTENT := **/*.md

.DEFAULT_GOAL := help
.PHONY: help spell spell-words spell-version prose prose-fix lint install clean

help:
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

spell: node_modules
	$(CSPELL) --no-progress --show-suggestions --gitignore "$(CONTENT)"

spell-words: node_modules
	@$(CSPELL) --no-progress --words-only --unique --gitignore "$(CONTENT)" | sort -uf

spell-version: node_modules
	@$(CSPELL) --version

prose: node_modules
	$(TEXTLINT) -f stylish "$(CONTENT)"

prose-fix: node_modules
	$(TEXTLINT) --fix "$(CONTENT)"

lint: spell prose

install: node_modules

node_modules: package-lock.json
	npm ci
	@touch node_modules

clean:
	rm -rf node_modules
