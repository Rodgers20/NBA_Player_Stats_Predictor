---
id: nba-conventional-commits
trigger: "when writing a commit message"
confidence: 0.8
domain: git
source: local-repo-analysis
analyzed_commits: 52
---

# Use Conventional Commits

## Action
Prefix all commit messages with a type:
- `feat:` — new feature
- `fix:` — bug fix
- `perf:` — performance improvement
- `chore:` — maintenance (deps, config)
- `docs:` — documentation
- `test:` — tests only
- `refactor:` — refactoring without behavior change

## Evidence
- Newer commits (last 10) consistently use conventional format
- Older commits are inconsistent ("Fix X to Y", "Add X", "Update X")
- Team is migrating toward conventional commits

## Example
```
feat: add shot quality metric to prop calculator
fix: handle missing player photo gracefully
perf: cache defensive stats with Parquet
```
