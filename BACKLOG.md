# Backlog

Cleanup backlog for `davisethan.github.io`. The repo is a fork of
[pages-themes/cayman](https://github.com/pages-themes/cayman); everything at or before
commit `56aa6db` (2024-01-16) is upstream theme code. Only `index.md`,
`_layouts/default.html`, `_config.yml`, `assets/css/style.scss`, and `assets/` have been
modified since the fork. The rest is theme-maintainer scaffolding that does nothing for
this site.

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
      `gem "github-pages", group: :jekyll_plugins`, kept rather than deleted because
      sprint 5's `_sass/` item cannot be verified without a local build.
      Confirmed via the API that Pages `build_type` is `legacy`, meaning GitHub builds with
      its own pinned gem set and ignores this file — so it affects local previews only and
      cannot change the deployed site.
- [x] Added `Gemfile.lock` to `.gitignore`. The regenerated `.gitignore` dropped it (no
      toptal template carries it), and `bundle install` would otherwise leave it untracked.

> **Leaves two dangling references until sprint 2d.** `script/release` and `script/cibuild`
> both run `gem build jekyll-theme-cayman.gemspec` against the now-deleted file. Both are
> dead theme-maintainer scripts that 2d removes; nothing runs them today (no workflow
> invokes them, and the CI that did was deleted in 2b).

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

> **Clears every dangling reference from 2a and 2b.** `script/cibuild` was the only caller of
> `.rubocop.yml`, and `script/{cibuild,release}` the only callers of the deleted gemspec.
> The one remaining broken link is `README.md:102` → `docs/CONTRIBUTING.md`, which sprint 4
> resolves by rewriting that file.

---

## Sprint 3 — Asset cleanup

Applies the asset hosting policy below. The three unreferenced PDFs are not one decision —
they fall into two different tiers, and only two of them are deletions.

- [x] **Deposit `pumps_poster.pdf` (1.3 MB) on Zenodo, then remove it from the repo.**
      A poster is citable scholarly output and a standard Zenodo deposit type, so it earns a
      DOI rather than a repo link. Cite it in Additional Research alongside the six Zenodo
      DOIs already there.
      _Done when:_ the poster has a DOI and `index.md` cites it.

- [x] **Delete `pumps_certification.pdf` (868 KB) and `osu_deans_list.pdf` (1.5 MB).**
      Credentials belong on the CV as line items, not published as scans — the reader already
      trusts the claim, so hosting the scan reads as padding. Neither is referenced anywhere,
      and the dean's list link was removed deliberately in commit `73fea61`
      ("Removed description of awards & honors").
      _Check first:_ that the CV lists both, so nothing is actually lost.
      _Also:_ certificate scans often carry student ID numbers, signatures, or addresses.
      Worth a look before they stay published anywhere at all.

- [ ] **Keep `Ethan_Davis_CV_2025.pdf` and `Ethan_Davis_Resume_2025.pdf` in the repo.**
      Not a deletion — recorded so they are not swept up with the rest. Both are load-bearing
      (linked from `index.md:8`), must stay current, and are cheap: 132 KB over 6 revisions
      has cost 767 KB of history, 84 KB over 3 revisions has cost 243 KB.

- [ ] **Delete `thumbnail.png`** — referenced only by the upstream Cayman README.
      _Blocked by:_ Sprint 4's README rewrite.

- [ ] Confirm `assets/images/favicon.png` stays — it is referenced from
      `_layouts/default.html` and is easy to sweep up by mistake.

> **This does not shrink the repository.** Git keeps deleted files in history, so every clone
> still pulls all 3.7 MB. `.git` is 7.6 MB today, roughly half of it files that will no longer
> ship. Sprint 3 cleans the working tree and the published site, not the clone size.
> Reclaiming that space needs `git filter-repo`, which rewrites every commit hash and forces a
> push — hard to justify for 3.7 MB. The lesson is preventive, which is what the policy below
> is for.

---

## Sprint 4 — Documentation

- [ ] **Replace `README.md`.** It is currently the Cayman theme's documentation, complete
      with the theme's CI and gem-version badges. Replace with a short README covering: what
      the site is, where it is published, how to preview it locally, and a credit line for
      the Cayman theme (CC0, `pages-themes/cayman`).
      _Unblocks:_ deleting `thumbnail.png` in Sprint 3.

- [ ] **Decide on `LICENSE`.** Recommendation: keep it. `_sass/` and `_layouts/` are derived
      from CC0-licensed Cayman. Note in the README that the license covers the theme, not the
      CV/resume/research content.

---

## Sprint 5 — Optional / needs a local preview

Deferred because these carry real risk of changing how the site renders. Do not attempt
without running the site locally and comparing before/after.

- [ ] **Consider deleting `_sass/`.** `_config.yml` sets `theme: jekyll-theme-cayman`, which
      GitHub Pages ships as a supported theme, so `@import 'jekyll-theme-cayman'` in
      `assets/css/style.scss` would resolve to the gem's copy instead of the local one. The
      local files are unmodified since the fork, so this *should* be a no-op — but it is a
      live rendering dependency and the gem version may drift from the vendored copy.
      _Done when:_ a local build renders identically with `_sass/` removed. Revert otherwise.

- [ ] **Remove dead Google Analytics plumbing.** `_config.yml` has an empty
      `google_analytics:` key, and `_includes/head-custom-google-analytics.html` is included
      via `_includes/head-custom.html`. Inert today.
      _Superseded by sprint 6_ — the same deletion is step 5 there. Do it in sprint 6, not
      here, so the site is never without analytics plumbing mid-change.

- [ ] **Remove `show_downloads: true` from `_config.yml`.** The customized
      `_layouts/default.html` dropped the `{% if site.show_downloads %}` block, so this
      setting has no effect. Verified as dead — no reference in `_layouts/`, `_includes/`,
      or `_sass/`.

- [x] **Trim `.gitignore`.** Regenerated from
      `toptal.com/developers/gitignore/api/macos,visualstudiocode,jekyll,node,sass` —
      dropped Emacs, added Visual Studio Code and Node. `node_modules/` is ignored and
      `package-lock.json` is not, both verified. `Gemfile.lock` was re-added manually in
      sprint 2a, since no toptal template carries it.

- [ ] **Consider splitting `index.md`.** At ~24 KB it holds the entire site in one page with
      a hand-maintained table of contents and six reference sections. Splitting into
      per-section pages would make the anchors in Sprint 1 much harder to break, at the cost
      of a real navigation layout. Larger project, not cleanup.

---

## Sprint 6 — Analytics (Cloudflare Web Analytics)

Answer "who is visiting the site, and where from" without cookies, a consent banner, or
cost. Chosen over Google Analytics 4 for three reasons:

1. The theme's GA include is **dead code**. `_includes/head-custom-google-analytics.html`
   uses `analytics.js` / `ga('create', …)` — Universal Analytics, which stopped processing
   hits 2023-07-01 and fully sunset 2024-07-01. Pasting a GA4 `G-XXXXXXXX` measurement ID
   into `_config.yml` would load an obsolete library and silently collect nothing.
2. GA4 sets cookies, so it pulls in a consent banner and a privacy policy.
3. This site's audience — ML researchers, engineers, recruiters — blocks trackers at high
   rates, and GA is the most-blocked script on the web. The undercount would be large and
   not correctable.

Cloudflare Web Analytics is free, uses no cookies or `localStorage`, and does not
fingerprint by IP or user agent.

### Steps

- [ ] **1. Create a free Cloudflare account.** No paid plan needed.

- [ ] **2. Add the site.** Web Analytics → *Add a site* → hostname `davisethan.github.io`.
      Use the **non-proxied** path. No DNS change and no proxying through Cloudflare is
      required — that is the whole reason this works on GitHub Pages.

- [ ] **3. Copy the JS snippet** from *Manage site*. It is a one-line `<script defer>` tag
      pointing at `static.cloudflareinsights.com/beacon.min.js` with a `data-cf-beacon`
      attribute containing a site-specific token. Copy it verbatim from the dashboard rather
      than reconstructing it — the token is generated per site.
      The token is **not a secret** (it ships in public page source), so committing it is fine.

- [ ] **4. Add the snippet to `_layouts/default.html`, immediately before `</body>`**
      (currently line 43). Cloudflare's docs specify before the closing body tag — *not* in
      `<head>`, so this does **not** go in `_includes/head-custom.html` despite that being
      the theme's usual customization hook.

- [ ] **5. Remove the dead GA plumbing.** Supersedes the sprint 5 item.
      - Delete `_includes/head-custom-google-analytics.html`
      - Remove the `{% include head-custom-google-analytics.html %}` line and its
        `<!-- Setup Google Analytics -->` comment from `_includes/head-custom.html`
      - Remove `google_analytics:` from `_config.yml`

- [ ] **6. Verify.** Load `https://davisethan.github.io/`, open devtools → Network, confirm a
      request to `static.cloudflareinsights.com`. Then confirm the visit appears in the
      Cloudflare dashboard — allow a few minutes for first data.
      _Done when:_ a real page view shows in the dashboard.

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

- [ ] Deposit the thesis PDF (~20–25 MB) on Zenodo, and check whether UW ResearchWorks
      deposit is required — it likely is, and gives a second permanent URL at no extra effort.
      Link from `index.md` by DOI. **Never commit the PDF.**
- [ ] Figures pulled out of the thesis for the page stay in the repo, sized for web, under the
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

    make               # list available targets
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
- [ ] Widen scope to `_layouts/*.html` and `_includes/*.html` if prose starts living there.
      Skipped for now — those are near-unmodified theme files and mostly markup.
- [ ] Consider adding a link checker. Neither linter can see a broken anchor; sprint 1's
      were found by a one-off script, which is not repeatable protection.
- [ ] Mention the linters in the new `README.md` when sprint 4 rewrites it.
