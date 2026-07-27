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

- [ ] Delete `jekyll-theme-cayman.gemspec` — packages Cayman as a RubyGem; not applicable.
- [ ] Delete or rewrite `Gemfile` — it is just `gemspec`, so it breaks once the gemspec is
      gone. Delete it (GitHub Pages builds fine without one), or replace the contents with
      `gem "github-pages", group: :jekyll_plugins` if local previews are wanted.
      _Blocked by:_ the gemspec deletion above — do both in the same commit.

### 2b. CI and lint config

- [x] Delete `.github/workflows/publish-gem.yml` — actively broken; ran
      `gem build github-pages.gemspec` against a file that does not exist in this repo.
- [x] Delete `.github/workflows/ci.yaml` — ran `script/cibuild` (html-proofer + W3C
      validation of the theme). Confirmed via `gh workflow list` that it was never a
      registered workflow, so nothing was running.
- [ ] Delete `.travis.yml` — Travis CI, dead.
- [ ] Delete `.rubocop.yml` — there is no Ruby source in this repo.

> **Keep `.github/workflows/spellcheck.yml`.** It is not upstream scaffolding — it is the
> spell check added for this site. See "Spell checking" below.

### 2c. Probot / bot config

None of these apps are installed on this repo.

- [x] Delete `.github/CODEOWNERS` — required review from `@pages-themes/maintainers`, a team
      that does not exist here.
- [x] Delete `.github/config.yml` — behaviorbot welcome/reply messages referencing the
      Cayman theme.
- [x] Delete `.github/settings.yml` — probot/settings repo config; also declared branch
      protection requiring the `script/cibuild` status check. Verified via the GitHub API
      that `master` has no branch protection configured, so nothing depended on it.
- [x] Delete `.github/stale.yml` and `.github/no-response.yml`.
- [x] `.github/` now contains only `workflows/spellcheck.yml`. Directory retained.

### 2d. Maintainer scripts and docs

- [ ] Delete `script/` — `release` publishes the gem, `cibuild` and `validate-html` test the
      theme, `bootstrap` installs the theme's dev dependencies. If local previews are wanted,
      `script/server` is the only one worth preserving, and `bundle exec jekyll serve` does
      the same thing.
- [ ] Delete `docs/` — `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SUPPORT.md` all describe
      contributing to the Cayman theme project.

---

## Sprint 3 — Asset cleanup

- [ ] **Delete three unreferenced PDFs from `assets/files/`** (~3.7 MB, roughly 95% of that
      directory and the bulk of the repo):
      - `osu_deans_list.pdf` (1.5 MB) — link removed in commit `73fea61`
        ("Removed description of awards & honors")
      - `pumps_certification.pdf` (868 KB) — no reference found in any tracked file
      - `pumps_poster.pdf` (1.3 MB) — no reference found in any tracked file

      Only `Ethan_Davis_CV_2025.pdf` (132 KB) and `Ethan_Davis_Resume_2025.pdf` (84 KB) are
      linked, from `index.md:8`.
      _Decide first:_ whether these should instead be *re-linked* from a certifications /
      awards section. They were clearly intentional uploads at one point.

- [ ] **Delete `thumbnail.png`** — referenced only by the upstream Cayman README.
      _Blocked by:_ Sprint 4's README rewrite.

- [ ] Confirm `assets/images/favicon.png` stays — it is referenced from
      `_layouts/default.html` and is easy to sweep up by mistake.

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

- [ ] **Trim `.gitignore`.** It is a generic toptal template covering Emacs, Sass, Jekyll,
      and macOS. Mostly noise, but harmless — low priority. Keep the `_site/` and
      `Gemfile.lock` entries.

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

## Spell checking (in place)

Added, not a todo. Recorded here so it is not mistaken for fork scaffolding.

- `Makefile` — the commands. Single source of truth; everything else calls into it.
- `package.json` / `package-lock.json` — pins the cspell version exactly and locks its
  dependency tree. npm exists in this repo *only* for the spell checker; the site itself is
  still plain Jekyll and needs no build step.
- `.github/workflows/spellcheck.yml` — runs `npm ci` then `make spell` on every PR to
  `master` and on direct pushes to `master`. Fails the build on an unknown word. It calls
  the Makefile rather than repeating the command, so CI and local runs cannot drift.
- `cspell.config.yaml` — dictionary and scope. Editors with the cSpell extension read the
  same file, so in-editor squiggles match CI.
- `.gitignore` — `node_modules/` added. `package-lock.json` is committed deliberately;
  `setup-node`'s npm cache and `npm ci` both require it.

Scope is `**/*.md`, which today means `index.md`, `README.md`, and `BACKLOG.md`.
`docs/`, `assets/`, and `_site/` are excluded.

**When CI flags a word:** if it is a real misspelling, fix the prose. If it is a valid term
(an author surname, an acronym, a method name), add it to the `words:` list in
`cspell.config.yaml` under the matching comment group. Do not silence a real error by
whitelisting it.

    make               # list available targets
    make spell         # check (exactly what CI runs)
    make spell-words   # list unknown words to triage
    make spell-version # print the installed cspell version
    make install       # install tooling from the lockfile
    make clean         # remove node_modules

`make spell` installs the tooling automatically if `node_modules/` is missing or the
lockfile is newer, so a fresh clone needs no setup step. `npm run spell` also works — the
npm scripts delegate to `make` rather than duplicating the command.

The cspell version is pinned exactly (no caret) in `package.json` because dictionary
changes in a new release can fail a build that has no content changes. To bump it:

    npm install --save-exact cspell@<version>   # then commit both package files

### Follow-ups

- [ ] Widen scope to `_layouts/*.html` and `_includes/*.html` if prose starts living there.
      Skipped for now — those are near-unmodified theme files and mostly markup.
- [ ] Consider adding a link checker alongside this, which would have caught the broken
      `index.md:138` anchors in sprint 1 that a spell checker cannot see.
- [ ] Mention the spell check in the new `README.md` when sprint 4 rewrites it.
