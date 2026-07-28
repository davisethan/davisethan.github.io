# davisethan.github.io

The personal research site of Ethan Davis: MS thesis, future research directions,
additional research, teaching and mentoring, and links to the CV and resume.

Published at <https://davisethan.github.io/>, built by GitHub Pages from `master` on
every push. There is no build step to run — the site is plain Jekyll and GitHub builds
it.

## What is where

| Path | What it is |
| --- | --- |
| `index.md` | The whole site. One page, with a hand-maintained table of contents and a reference list per section. |
| `_layouts/default.html` | Page shell, customized from the Cayman theme. |
| `_includes/` | Head customization hooks. |
| `assets/css/style.scss` | Site styles, imported on top of the theme. |
| `assets/images/` | Figures and the favicon. |
| `assets/files/` | CV and resume. |
| `_config.yml` | Jekyll configuration. |
| `BACKLOG.md` | Cleanup backlog, and the policy for where an asset belongs. |

Anything citable is deposited externally and linked by DOI rather than committed here.
`BACKLOG.md` has the reasoning; the short version is that binaries do not
delta-compress, so every revision is stored close to in full, forever, in every clone.

## Previewing locally

GitHub Pages builds the published site, so this is only for checking a change before
pushing.

```sh
bundle install
bundle exec jekyll serve
```

Then open <http://localhost:4000>.

The `Gemfile` affects local previews only. This repository uses a legacy Pages build, so
GitHub builds with its own pinned gem set and ignores it.

## Linting

Markdown is spell checked with cspell and terminology checked with textlint. Both run in
CI on pull requests to `master` and on direct pushes, through
`.github/workflows/spellcheck.yml` and `.github/workflows/prose.yml`. Both workflows call
the Makefile rather than repeating the commands, so CI and local runs cannot drift.

```sh
make lint          # spell + prose, both CI checks
make spell         # spelling only
make spell-words   # list unknown words to triage
make prose         # terminology only
make spell-version # print the installed cspell version
make prose-fix     # auto-apply terminology fixes
make install       # install tooling from the lockfile
make clean         # remove node_modules
```

Targets install the tooling automatically when `node_modules/` is missing or the
lockfile is newer, so a fresh clone needs no setup step. `npm run lint` also works — the
npm scripts delegate to `make`.

When the spell check flags a word, fix the prose if it is a real misspelling, or add the
word to the `words:` list in `cspell.config.yaml` under the matching comment group if it
is a valid term. Do not silence a real error by whitelisting it. When the prose check
flags a term, `make prose-fix` applies every terminology fix; review the diff before
committing, since it edits published prose.

Versions are pinned exactly in `package.json`, without a caret, because a dictionary or
rule change in a new release can fail a build that has no content changes.

## Credits and license

The theme is [Cayman](https://github.com/pages-themes/cayman), released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). `_sass/`, `_layouts/`,
and `_includes/` derive from it, and this repository began as a fork of it.

`LICENSE` is Cayman's CC0 and covers that theme code. It does not cover the writing,
figures, CV, or resume in this repository.
