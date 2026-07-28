CSPELL := npx cspell
TEXTLINT := npx textlint

CONTENT := **/*.md

DOCKER := docker
RUBY_IMAGE := ruby:3.3
GEM_VOLUME := davisethan-gems
PORT := 4000
TTY := -it

BUNDLE = $(DOCKER) run --rm \
	-v "$(CURDIR)":/site -w /site \
	-v $(GEM_VOLUME):/usr/local/bundle \
	$(RUBY_IMAGE) bundle
JEKYLL = $(BUNDLE) exec jekyll
HTMLPROOFER = $(BUNDLE) exec htmlproofer

.DEFAULT_GOAL := help
.PHONY: help spell spell-words spell-version prose prose-fix lint install clean \
	serve deps build links

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

serve:
	@$(DOCKER) info >/dev/null 2>&1 \
		|| { echo "Docker is not running. Start Docker Desktop, then retry."; exit 1; }
	@echo "Serving on http://localhost:$(PORT) — first run installs gems, which takes a few minutes."
	$(DOCKER) run --rm $(TTY) \
		-v "$(CURDIR)":/site -w /site \
		-v $(GEM_VOLUME):/usr/local/bundle \
		-p $(PORT):4000 \
		$(RUBY_IMAGE) \
		bash -c "bundle install && bundle exec jekyll serve --host 0.0.0.0 --force_polling"

deps:
	$(BUNDLE) install --quiet

build: deps
	$(JEKYLL) build

links: build
	$(HTMLPROOFER) --disable-external _site

install: node_modules

node_modules: package-lock.json
	npm ci
	@touch node_modules

clean:
	rm -rf node_modules _site .jekyll-cache
