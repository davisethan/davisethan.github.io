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
	serve deps build links links-external

help: ## list available targets
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

spell: node_modules ## spelling only
	$(CSPELL) --no-progress --show-suggestions --gitignore "$(CONTENT)"

spell-words: node_modules ## list unknown words to triage
	@$(CSPELL) --no-progress --words-only --unique --gitignore "$(CONTENT)" | sort -uf

spell-version: node_modules ## print the installed cspell version
	@$(CSPELL) --version

prose: node_modules ## terminology only
	$(TEXTLINT) -f stylish "$(CONTENT)"

prose-fix: node_modules ## auto-apply terminology fixes
	$(TEXTLINT) --fix "$(CONTENT)"

lint: spell prose ## spell + prose

serve: ## local preview in Docker
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

build: deps ## build the site to _site/
	$(JEKYLL) build

links: build ## build, then check internal links and anchors
	$(HTMLPROOFER) --disable-external _site

# timeout: doi.org redirects to zenodo.org, which is slow
# medium.com: 403s automated requests
# fonts.gstatic.com: preconnect target, not a page
links-external: build ## build, then also check external links (slow)
	$(HTMLPROOFER) _site \
		--typhoeus '{"timeout":90,"connecttimeout":30,"followlocation":true}' \
		--ignore-urls '/^https://medium\.com/,/^https://fonts\.gstatic\.com/'

install: node_modules ## install tooling from the lockfile

node_modules: package-lock.json
	npm ci
	@touch node_modules

clean: ## remove node_modules, _site, .jekyll-cache
	rm -rf node_modules _site .jekyll-cache
