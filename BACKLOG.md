# Backlog

Cleanup backlog for `davisethan.github.io`. The repo began as a fork of
[pages-themes/cayman](https://github.com/pages-themes/cayman); everything at or before
commit `56aa6db` (2024-01-16) is upstream theme code. Sprints 2 through 5 removed that
scaffolding. What is left of the theme is two customized files — `_layouts/default.html`
and `assets/css/style.scss` — with the rest of Cayman coming from the
`jekyll-theme-cayman` gem that GitHub Pages ships.

Sprints are ordered by risk: user-visible fixes first, deletions second, judgment calls last.

---

## Sprint 1 — Correctness (user-visible bugs)

Small, high-value fixes to things that are currently broken on the live site.

<!-- cspell:disable -->

- [x] **Fix six misspellings in `index.md`.** Found by the spell check added below:
      `classifers`→`classifiers`, `undertainty`→`uncertainty` (x2), `Hamiltonean`→`Hamiltonian`,
      `geometrics`→`geometric`, `statistcal`→`statistical`, `differentation`→`differentiation`.

<!-- cspell:enable -->


- [x] **Fix three broken reference anchors in `index.md:138`.**
      `[(2016)]((#teaching-assistant-references))` had a doubled paren and rendered as a dead
      link, three times on that line (Bishop 2016, Deisenroth 2020 x2).

- [x] **Audit every remaining in-page anchor in `index.md`.**
      Checked all 60 in-page links against the 16 generated heading slugs. The audit found
      one defect the backlog had missed: an **empty link target** `[[2, 6]]()` at line 68,
      in Future Directions. Now points to `#future-directions-references`, matching every
      other citation in that section. No other target failed to resolve.

- [x] **Set or remove `description:` in `_config.yml`.**
      Removed the commented-out `description:` line, and dropped the `project-tagline`
      `<h2>` from `_layouts/default.html`. With no `site.description`, no `page.description`,
      and an empty GitHub repo description, that element rendered as an empty `<h2>` still
      carrying Cayman's `margin-bottom: 2rem`.
      _To restore a tagline later:_ add `description:` back to `_config.yml` and re-add
      `<h2 class="project-tagline">{{ page.description | default: site.description }}</h2>`
      to the header in `_layouts/default.html`. Both are needed — the config key alone has
      nothing to render into.

---

## Sprint 2 — Delete fork scaffolding

Pure deletions. None of these affect the built site. Recommend one commit per group so
anything can be reverted independently.

### 2a. Gem packaging

- [x] Delete `jekyll-theme-cayman.gemspec` — packaged Cayman as a RubyGem; not applicable.
- [x] Rewrite `Gemfile` — was just `gemspec`, which breaks once the gemspec is gone. Now
      `gem "github-pages", group: :jekyll_plugins`.
      Confirmed via the API that Pages `build_type` is `legacy`, meaning GitHub builds with
      its own pinned gem set and ignores this file — so it affects local previews only and
      cannot change the deployed site. It is load-bearing for those: since `_sass/` and
      `_includes/` were deleted, the theme reaches `make serve` only through this gem.
- [x] ~~Added `Gemfile.lock` to `.gitignore`.~~ **Reversed in sprint 7.** Correct when
      nothing consumed the `Gemfile`; wrong once CI does. This is an application, not a gem,
      so the lockfile is committed — it is what makes CI gem resolution reproducible, and
      what `ruby/setup-ruby`'s `bundler-cache` keys on. Pages ignores it either way.

### 2b. CI and lint config

- [x] Delete `.github/workflows/publish-gem.yml` — actively broken; ran
      `gem build github-pages.gemspec` against a file that does not exist in this repo.
- [x] Delete `.github/workflows/ci.yaml` — ran `script/cibuild` (html-proofer + W3C
      validation of the theme). Confirmed via `gh workflow list` that it was never a
      registered workflow, so nothing was running.
- [x] Delete `.travis.yml` — Travis CI, dead. Ran `script/bootstrap` then `script/cibuild`,
      both of which sprint 2d removes.
- [x] Delete `.rubocop.yml` — no Ruby source to lint, and already unresolvable: it did
      `inherit_gem: rubocop-github`, which was a dev dependency of the gemspec deleted in
      sprint 2a. Its only caller was `script/cibuild` (sprint 2d).

> **Keep `.github/workflows/spellcheck.yml` and `.github/workflows/prose.yml`.** Neither is
> upstream scaffolding — both were added for this site. See "Linting (in place)" below.

### 2c. Probot / bot config

None of these apps are installed on this repo.

- [x] Replace `.github/CODEOWNERS` — the upstream version required review from
      `@pages-themes/maintainers`, a team that does not exist here. Rewritten to
      `* @davisethan` in commit `15920a8` rather than deleted. **Keep it** — it is now a
      legitimate ownership file, not fork scaffolding.
- [x] Delete `.github/config.yml` — behaviorbot welcome/reply messages referencing the
      Cayman theme.
- [x] Delete `.github/settings.yml` — probot/settings repo config; also declared branch
      protection requiring the `script/cibuild` status check. Verified via the GitHub API
      that `master` has no branch protection configured, so nothing depended on it.
- [x] Delete `.github/stale.yml` and `.github/no-response.yml`.
- [x] `.github/` now contains only `CODEOWNERS`, `workflows/spellcheck.yml`, and
      `workflows/prose.yml`. Directory retained.

### 2d. Maintainer scripts and docs

- [x] Delete `script/` — all five were theme-maintainer tooling. `release` published the gem
      and tagged versions, `cibuild` ran the theme test suite, `validate-html` needed
      `w3c_validators` (a dev dependency of the gemspec deleted in 2a), `bootstrap` was
      `gem install bundler; bundle install`, and `server` was the single line
      `bundle exec jekyll serve` — no reason to keep a script for that.
- [x] Delete `docs/` — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SUPPORT.md` all described
      contributing to the Cayman theme project.
- [x] Removed the now-stale `docs/**` entries from `cspell.config.yaml` `ignorePaths` and
      from `.textlintignore`.

---

## Sprint 3 — Asset cleanup

Applies the asset hosting policy below. The three unreferenced PDFs are not one decision —
they fall into two different tiers, and only two of them are deletions.

- [x] **Deposit `pumps_poster.pdf` (1.3 MB) on Zenodo, then remove it from the repo.**
      A poster is citable scholarly output and a standard Zenodo deposit type, so it earns a
      DOI rather than a repo link. Cite it in Additional Research alongside the six Zenodo
      DOIs already there.
      _Done:_ DOI `10.5281/zenodo.21631478`, cited as reference 14 in Additional Research.
      Deposited with `zenodo/deposit_poster.sh` (outside this repo), which kebab-cases the
      file before upload.

- [x] **Delete `pumps_certification.pdf` (868 KB) and `osu_deans_list.pdf` (1.5 MB).**
      Credentials belong on the CV as line items, not published as scans — the reader already
      trusts the claim, so hosting the scan reads as padding. Neither is referenced anywhere,
      and the dean's list link was removed deliberately in commit `73fea61`
      ("Removed description of awards & honors").
      _Check first:_ that the CV lists both, so nothing is actually lost.
      _Also:_ certificate scans often carry student ID numbers, signatures, or addresses.
      Worth a look before they stay published anywhere at all.

- [x] **Keep `Ethan_Davis_CV_2025.pdf` and `Ethan_Davis_Resume_2025.pdf` in the repo.**
      Not a deletion — recorded so they are not swept up with the rest. Both are load-bearing
      (linked from `index.md:8`), must stay current, and are cheap: 132 KB over 6 revisions
      has cost 767 KB of history, 84 KB over 3 revisions has cost 243 KB.

- [x] **Delete `thumbnail.png`** — referenced only by the upstream Cayman README, and
      unreferenced once sprint 4 rewrote it.

- [x] Confirm `assets/images/favicon.png` stays — it is referenced from
      `_layouts/default.html` and is easy to sweep up by mistake.

> **This does not shrink the repository.** Git keeps deleted files in history, so every clone
> still pulls all 3.7 MB. `.git` is 7.6 MB today, roughly half of it files that will no longer
> ship. Sprint 3 cleans the working tree and the published site, not the clone size.
> Reclaiming that space needs `git filter-repo`, which rewrites every commit hash and forces a
> push — hard to justify for 3.7 MB. The lesson is preventive, which is what the policy below
> is for.

---

## Sprint 4 — Documentation

- [x] **Replace `README.md`.** It is currently the Cayman theme's documentation, complete
      with the theme's CI and gem-version badges. Replace with a short README covering: what
      the site is, where it is published, how to preview it locally, and a credit line for
      the Cayman theme (CC0, `pages-themes/cayman`).

- [x] **Decide on `LICENSE`.** Kept. `_layouts/default.html` is derived from CC0-licensed
      Cayman — the only such file left, now that `_sass/` and `_includes/` are gone. The
      README notes that the license covers the theme, not the CV/resume/research content.
      CC0 requires no attribution, so this is a courtesy rather than an obligation.

---

## Sprint 5 — Optional / needs a local preview

Deferred because these carry real risk of changing how the site renders. `make serve` now
provides the local preview these needed — see "Site build" below.

- [x] **Consider deleting `_sass/`.** `_config.yml` sets `theme: jekyll-theme-cayman`, which
      GitHub Pages ships as a supported theme, so `@import 'jekyll-theme-cayman'` in
      `assets/css/style.scss` would resolve to the gem's copy instead of the local one. The
      local files are unmodified since the fork, so this *should* be a no-op — but it is a
      live rendering dependency and the gem version may drift from the vendored copy.
      _Done._ All five files were byte-identical to the published `jekyll-theme-cayman`
      0.2.0 gem, which is exactly the version Pages ships. `_includes/` turned out to be
      identical too, so both directories were deleted. A local build confirms the compiled
      `style.css` still carries the Cayman rules, resolved from the gem.
      Drift is also less of a risk than assumed: the `github-pages` gem pins
      `jekyll-theme-cayman = 0.2.0` exactly, and 0.2.0 has been current since 2021-07-29.

- [x] **Remove dead Google Analytics plumbing.** Done as a side effect of deleting
      `_includes/`, plus dropping the empty `google_analytics:` key from `_config.yml`.
      Note the correction: deleting the local includes did not remove the GA snippet, it
      fell through to the gem's copy. What silences it is the missing config key, since the
      gem wraps the snippet in a `site.google_analytics` conditional.

- [x] **Remove `show_downloads: true` from `_config.yml`.** The customized
      `_layouts/default.html` dropped the `site.show_downloads` conditional, so the setting
      had no effect. Worth knowing it was a live trap rather than mere clutter: revert to
      the stock Cayman layout with the key still set and the "Download .zip" and
      "Download .tar.gz" buttons reappear.
      _Do not write that conditional out in full anywhere in this file_ — see "Site build"
      below for why an unclosed Liquid tag here once broke the production build.

- [x] **Trim `.gitignore`.** Regenerated from
      `toptal.com/developers/gitignore/api/macos,visualstudiocode,jekyll,node,sass` —
      dropped Emacs, added Visual Studio Code and Node. `node_modules/` is ignored and
      `package-lock.json` is not, both verified. `Gemfile.lock` was re-added manually in
      sprint 2a, since no toptal template carries it.

- [x] **Consider splitting `index.md`.** Considered; **not now**, revisit after the content
      rewrite. Findings, so this does not have to be re-derived:
      - 3,033 words across 3 H1 groups, 8 H2 sections, and 6 reference lists.
      - All 45 citation links are **section-local** — every one points at a reference list
        inside its own H1 group. Splitting on H1 boundaries yields zero cross-page links, so
        the mechanical work is small. The other 16 anchors are the hand-maintained table of
        contents, which navigation would replace anyway.
      - The real cost is that **Cayman ships no navigation at all** — stock layout is header,
        content, footer. Multi-page needs a nav built into the layout override.
      - The unfixable part: **URL fragments never reach the server**, so no redirect can
        rescue an existing `davisethan.github.io/#ms-thesis` link. Anything citing a section
        anchor — CV, resume, arXiv, Zenodo, LinkedIn — degrades silently.
      - Natural shape if revisited: `/` (photo, contact, bio, nav), `/research/` (~2,009
        words), `/teaching/` (~413), `/industry/` (~450).

---

## Sprint 6 — Analytics (Cloudflare Web Analytics)

Answer "who is visiting the site, and where from" without cookies, a consent banner, or
cost. Chosen over Google Analytics 4 for three reasons:

1. The theme's GA include is **dead code**. It uses `analytics.js` / `ga('create', …)` —
   Universal Analytics, which stopped processing hits 2023-07-01 and fully sunset
   2024-07-01. Pasting a GA4 `G-XXXXXXXX` measurement ID into `_config.yml` would load an
   obsolete library and silently collect nothing. The include now lives only in the gem,
   since `_includes/` was deleted in sprint 5.
2. GA4 sets cookies, so it pulls in a consent banner and a privacy policy.
3. This site's audience — ML researchers, engineers, recruiters — blocks trackers at high
   rates, and GA is the most-blocked script on the web. The undercount would be large and
   not correctable.

Cloudflare Web Analytics is free, uses no cookies or `localStorage`, and does not
fingerprint by IP or user agent.

### Steps

- [x] **1. Create a free Cloudflare account.** No paid plan needed.

- [x] **2. Add the site.** Web Analytics → *Add a site* → hostname `davisethan.github.io`.
      Use the **non-proxied** path. No DNS change and no proxying through Cloudflare is
      required — that is the whole reason this works on GitHub Pages.

- [x] **3. Copy the JS snippet** from *Manage site*. Cloudflare now ships a
      `<script type="module">` tag, not `<script defer>` — modules defer by default.
      It points at `static.cloudflareinsights.com/beacon.min.js` with a `data-cf-beacon`
      attribute containing a site-specific token. Copy it verbatim from the dashboard rather
      than reconstructing it — the token is generated per site.
      The token is **not a secret** (it ships in public page source), so committing it is fine.
      _Reformatted for readability, but the single quotes on `data-cf-beacon` are load-bearing:_
      the attribute value is JSON containing double quotes, so the outer pair must be single.

- [x] **4. Add the snippet to `_layouts/default.html`, immediately before `</body>`**
      (currently line 43). Cloudflare's docs specify before the closing body tag, not in
      `<head>`. The layout override is now the only place it can go — `_includes/`, the
      theme's usual customization hook, was deleted in sprint 5.

- [x] **5. Remove the dead GA plumbing.** Done ahead of this sprint, in sprint 5. The two
      `_includes/` files are gone and `google_analytics:` is out of `_config.yml`, which is
      the part that actually silences it.

- [x] **6. Verify.** Load `https://davisethan.github.io/`, open devtools → Network, confirm a
      request to `static.cloudflareinsights.com`. Then confirm the visit appears in the
      Cloudflare dashboard — allow a few minutes for first data.
      _Done when:_ a real page view shows in the dashboard.
      **This step cannot be done with `make serve`.** Cloudflare validates the beacon
      hostname by postfix match against the registered site, so hits from `localhost:4000`
      are discarded. A local build only confirms the tag renders and the script is fetched;
      the dashboard confirmation requires the deployed site.

### Notes

- **Consent banner:** with no cookies, no `localStorage`, and no fingerprinting, one is
  generally not required. Confirm against your own obligations before relying on that.
- **Still some undercount:** less blocked than GA, but some blocklists do include
  `cloudflareinsights.com`. Expect directionally accurate numbers, not exact ones.
- **Scope:** top pages, referrers, visitor counts, Core Web Vitals. No funnels, cohorts, or
  demographics. Sufficient for the question being asked; do not expect GA4 depth.
- **Not linted:** `make lint` only scans Markdown, so an HTML snippet in the layout is not
  covered by the spell or prose checks.
- **Not a substitute:** GitHub's repository Insights → Traffic reports views of the *repo*,
  not the Pages site.

### Alternatives if this does not fit

GoatCounter (free for non-commercial), Plausible (~$9/mo), Fathom (~$15/mo). All are
cookie-free and privacy-first; the paid ones offer more depth than Cloudflare.

---

## Sprint 7 — Build and link checking in CI

Two problems, one job. Neither linter can see a broken link, and nothing checks that the
site still builds until after a push — which is how the site stayed broken for a day when
`BACKLOG.md` took the Pages build down. Building in CI fixes the second problem, and once
the site is built, checking its links is nearly free.

Why this cannot be another `**/*.md` linter: anchors are **generated**, not written. The
built page has 62 anchor links against 18 kramdown-generated heading IDs. Checking the
Markdown source would mean reimplementing kramdown's slug algorithm; checking `_site/`
compares real `href` values against real `id` values.

Tool is `html-proofer` (5.2.1) — a Ruby gem run against `_site/`. It is the standard for
Jekyll, and it is what Cayman's own deleted `script/cibuild` used. The Ruby dependency is
free here only because the build step needs Ruby anyway; without that step a lighter tool
over the Markdown would be preferable, at the cost of reliable anchor checking.

### Steps

- [x] **1. Add `html-proofer` to the `Gemfile`.** A development dependency. Pages uses a
      legacy build and ignores this file, so there is no risk to production.

- [ ] **2. Add `build` and `links` targets to the `Makefile`.** `links` depends on `build`.
      Make the runtime a variable rather than duplicating commands — locally Jekyll runs in
      Docker, in CI it runs under native Ruby:

          JEKYLL := $(DOCKER) run --rm -v "$(CURDIR)":/site ... bundle exec jekyll
          build:
              $(JEKYLL) build
          links: build
              $(HTMLPROOFER) --disable-external _site

      CI then calls `make links JEKYLL="bundle exec jekyll" HTMLPROOFER="bundle exec htmlproofer"`
      — same target, same flags, different runtime.

- [ ] **3. Add the build + internal-link job.** Triggers on pull requests to `master` and on
      direct pushes, matching `spellcheck.yml` and `prose.yml`. **Blocking.** Uses
      `ruby/setup-ruby` with `bundler-cache: true`, then calls the Makefile target.
      The cache needs the committed `Gemfile.lock` (task 1) — without it every run installs
      115 gems from scratch.
      _Done when:_ a pull request with a deliberately broken anchor fails the check.

- [ ] **4. Add the external-link job.** Weekly `schedule` trigger, `continue-on-error: true`,
      **non-blocking**. Separate from step 3 on purpose: LinkedIn and Medium return 403/999
      to CI runners for reasons unrelated to this site, and a link checker that blocks merges
      for that gets switched off within a month. Run it anyway — 10 of the 23 external links
      are DOIs, and a hand-typed DOI is a live risk.
      Use `--ignore-status-codes` and `--ignore-urls` for hosts that are reliably hostile.

- [ ] **5. Verify against the two failures that already happened.** Not a hypothetical test:
      - Sprint 1's `[[2, 6]]()` produced `<a href="">` — should be flagged.
      - Sprint 3's six deleted images stayed referenced in `index.md` and shipped broken to
        production — should be flagged by the Images check.
      _Done when:_ both are caught by a local `make links` run before the workflow is trusted.

- [ ] **6. Document `make build` and `make links` in `README.md`,** alongside the existing
      lint targets.

- [ ] **7. Tune the ignore lists after the first scheduled external run,** once it is clear
      which hosts actually misbehave rather than which ones are predicted to.

### Notes

- **The build check is the larger win.** Link checking is the stated goal, but catching a
  Liquid or config error in a pull request — rather than from a "Page build failed" email
  after the fact — is what prevents the failure mode this repo has actually hit.
- **This does not replace `make serve`.** CI proves the site builds; only a local preview
  shows what it looks like.
- **Scope note:** `html-proofer` checks the built HTML, so it covers `index.md` and the
  layout. `BACKLOG.md` and `README.md` are excluded from the build and will not be checked.

---

## Site build (in place)

Added, not a todo. Recorded because a build failure here took the live site down for a day.

### Local preview

`make serve` runs Jekyll in a `ruby:3.3` container and serves on
[localhost:4000](http://localhost:4000). Nothing is installed on the host — system Ruby is
2.6.10 and `github-pages` needs 3.3.4. Gems are cached in a named Docker volume, so only
the first run is slow. It takes 20–40 seconds to come up; a connection error before then is
normal. `make serve PORT=4001` if 4000 is taken.

This is the only way to see a change before pushing, and the only way to reproduce a Pages
build failure — GitHub reports nothing more useful than "Page build failed."

### Why `_config.yml` has an `exclude:` list

**GitHub Pages enables `jekyll-optional-front-matter`**, so *every* Markdown file becomes a
page, front matter or not. That is not stock Jekyll behavior, and it is the trap:
`BACKLOG.md` quoted an unclosed Liquid `if` tag, Liquid runs before Markdown so backticks
do not protect it, and the build failed with a syntax error pointing at this file. Two
consecutive Pages builds errored before the cause was found.

So repository docs and tooling are excluded — they are not site content, and they must not
be parsed as pages. Consequences worth knowing:

- **Never write a bare unclosed Liquid tag in any `.md` file here.** The `exclude:` entry is
  the only thing making it safe, and deleting that entry re-breaks production.
- **Setting `exclude` replaces Jekyll's defaults rather than merging them**, so `Gemfile`,
  `Gemfile.lock`, and `.jekyll-cache` had to be listed again. Caught because `Gemfile`
  started appearing in `_site/`.
- `node_modules` is listed for local builds only. It is gitignored, so Pages never sees it,
  but `make serve` does — and a malformed Liquid tag in a dependency's README will fail the
  build.

### `assets/css/style.scss` needs its empty front matter

The `---` / `---` at the top is not decoration. It is what tells Jekyll to compile the file
to `style.css`. Remove it and the failure is silent rather than loud: the site still gets a
`style.css`, because the gem ships its own `assets/css/style.scss` — but it is the gem's,
without the `.icon-link` rules, and the raw `.scss` is published alongside it. Verified both
ways with a local build.

---

## Asset hosting policy

Standing rule, not a todo. Where a file belongs, decided by three questions in order.
Written down so the thesis-asset sprint does not have to re-derive it.

**1. Would anyone ever cite it?** → External archive, linked by DOI.
Thesis, papers, posters, datasets, code releases. Not a size question — a 200 KB poster still
belongs on Zenodo, because a DOI is worth more than a repo link. This is already the
established practice here: `index.md` cites six Zenodo DOIs and two arXiv preprints.

**2. Is it load-bearing for the site?** → In the repo.
The CV, the resume, and any figure embedded in `index.md`. Two reasons beyond size: they must
always be current, and the file a recruiter clicks should not depend on a third-party host
being reachable. In-repo files are served from GitHub's CDN alongside the page.
Budget: **size × expected lifetime revisions, up to about 5 MB per asset.** The CV passes
easily — roughly 3 MB over a five-year PhD at four revisions a year. A 25 MB thesis at three
revisions costs 75 MB and fails by 15x. Frequency only matters multiplied by bulk.

**3. Neither?** → Do not host it.
Certificates, transcripts, award letters. State them on the CV; that is the convention, and
the scan adds nothing the reader was doubting.

### Constraints that rule out the obvious alternatives

- **Git LFS does not work with GitHub Pages.** Pages does not resolve LFS pointers, so
  visitors would download the pointer text file instead of the PDF. This removes the usual
  answer for large binaries.
- **Pages limits are not the binding constraint.** 1 GB recommended repository size, 100 GB
  per month soft bandwidth, and GitHub blocks regular non-LFS files over 100 MB. A single
  25 MB PDF clears all three. The real cost is Git history: binaries do not delta-compress,
  so every revision is stored close to in full, forever, in every clone.
- **Not Google Drive.** No DOI, a virus-scan warning page on large files, links that rot when
  the folder is reorganized, and it looks out of place beside arXiv and Zenodo DOIs.

### Thesis assets (future sprint)

- [x] Deposit the thesis PDF (20.3 MB) on Zenodo. Link from `index.md` by DOI.
      **Never commit the PDF.** Metadata is prepared at `zenodo/metadata/thesis.json`
      (outside this repo) and dry-run clean; `zenodo/deposit_poster.sh` takes any file.
      _Blocked on:_ which access option was selected in the ProQuest ETD agreement —
      immediate open access or delayed. UW honors that choice for both ProQuest and the IR,
      so a delayed selection is an embargo and Zenodo must wait. ProQuest showing a 24-page
      preview proves nothing either way; that is its standard paywall for non-OA deposits.
- [x] Chase the UW ResearchWorks deposit — it is **not** PhD-only, contrary to first
      assumption. The ETD confirmation email names ProQuest *and* ResearchWorks, and the
      repository holds 38 master's theses from 2026 alone. Yours is not loaded yet; records
      arrive in batches (2026-04-20 for spring; 2025-08-01 and 2024-09-09 for the summer
      cohorts), so a June graduate lands in the summer batch.
      _Contact:_ `rworks@uw.edu`, manuscript ID 29520.
      ProQuest record is live: https://www.proquest.com/docview/3366477179, publication
      no. 32735698.
      When it appears, it also answers the embargo question for free — full text open means
      immediate open access was selected.
- [x] Figures pulled out of the thesis for the page stay in the repo, sized for web, under the
      same per-asset budget.

---

## Linting (in place)

Added, not a todo. Recorded here so it is not mistaken for fork scaffolding.

- `Makefile` — the commands. Single source of truth; everything else calls into it.
- `package.json` / `package-lock.json` — pins `cspell`, `textlint`, and
  `textlint-rule-terminology` to exact versions and locks the dependency tree. npm exists in
  this repo *only* for these linters; the site itself is still plain Jekyll and needs no
  build step.
- `.github/workflows/spellcheck.yml` — runs `npm ci` then `make spell`.
- `.github/workflows/prose.yml` — runs `npm ci` then `make prose`.
  Both trigger on PRs to `master` and on direct pushes to `master`, and both call the
  Makefile rather than repeating the command, so CI and local runs cannot drift.
- `cspell.config.yaml` — spelling dictionary and scope. Editors with the cSpell extension
  read the same file, so in-editor squiggles match CI.
- `.textlintrc.yml` — terminology rule config. `exclude` needs the rule's **regular expression patterns**
  (`"readme(s)?"`, `"repo\\b"`), not plain words; passing words silently does nothing.
- `.textlintignore` — textlint does not read `.gitignore`, so exclusions are listed here
  separately. Patterns need glob form (`_site/**`), not directory form (`_site/`).
- `.gitignore` — `node_modules/` ignored. `package-lock.json` is committed deliberately;
  `setup-node`'s npm cache and `npm ci` both require it.

Scope for both is `**/*.md`, which today means `index.md`, `README.md`, and `BACKLOG.md`.

**When the spell check flags a word:** if it is a real misspelling, fix the prose. If it is
a valid term (an author surname, an acronym, a method name), add it to the `words:` list in
`cspell.config.yaml` under the matching comment group. Do not silence a real error by
whitelisting it.

**When the prose check flags a term:** `make prose-fix` auto-applies every terminology fix.
Review the diff before committing — it edits published prose.

    make serve         # local preview in Docker, http://localhost:4000
    make lint          # spell + prose, both CI checks
    make spell         # spelling only
    make spell-words   # list unknown words to triage
    make spell-version # print the installed cspell version
    make prose         # terminology only
    make prose-fix     # auto-apply terminology fixes
    make install       # install tooling from the lockfile
    make clean         # remove node_modules

Targets install the tooling automatically if `node_modules/` is missing or the lockfile is
newer, so a fresh clone needs no setup step. `npm run lint`, `npm run spell`, and
`npm run prose` also work — the npm scripts delegate to `make` rather than duplicating the
commands.

Versions are pinned exactly (no caret) in `package.json` because dictionary or rule changes
in a new release can fail a build that has no content changes. To bump one:

    npm install --save-exact cspell@<version>   # then commit both package files

### Follow-ups

- [ ] **`make help` prints nothing.** It greps for `## ` doc comments that were later removed
      from the `Makefile`. Either restore the comments or replace `help` with a static list.
- [ ] Widen scope to `_layouts/*.html` if prose starts living there. Skipped for now — it is
      mostly markup. (`_includes/` no longer exists.)
- [x] Consider adding a link checker. Neither linter can see a broken anchor; sprint 1's
      were found by a one-off script, which is not repeatable protection.
      _Considered and scoped:_ see sprint 7, which pairs it with a CI build.
- [x] Mention the linters in the new `README.md` when sprint 4 rewrites it.
