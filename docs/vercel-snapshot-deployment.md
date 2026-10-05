# Public Vercel snapshot deployment

The public Vercel site serves a **daily static snapshot** of the existing NBA/WNBA
FastAPI responses. GitHub Actions refreshes free historical data, starts the API
temporarily on its own runner, exports public JSON to `frontend/public/data`,
builds the Next.js static output, and deploys it. Vercel serves the result without
running the Python API. My Bets remains usable in the browser, but entries stay
on that device; there is no server journal, account sync, or live API write.

This is deliberately a daily refresh, not a live data stream. The scheduled job
runs at 11:17 UTC and can be delayed by GitHub; `workflow_dispatch` permits an
additional manual refresh after new data is available. A failed refresh/export
stops deployment, leaving the prior production snapshot intact. No paid Odds API
key is given to the runner or the public site. Game prices come from the existing
free ESPN path; unquoted props remain research rather than fabricated picks.

## One-time setup

1. Commit and push the app, `scripts/export_api.py`, both incremental refresh
   scripts, and `.github/workflows/deploy-vercel-snapshot.yml` to the repository's
   default branch. GitHub scheduled workflows only run from that branch.
2. Create a **Vercel project for `frontend/`**. Set the framework preset to
   **Other** and output directory to **`out`** so Vercel serves Next's static
   export rather than trying to run Next server routes. The GitHub workflow supplies the generated data at deploy time; do
   not rely on an ordinary Git-triggered build, which checks out no generated
   snapshot. A CLI-linked project without Git auto-deploy is suitable.
3. Add production environment variables in Vercel project settings:
   `NEXT_EXPORT=1` and `NEXT_PUBLIC_DATA_MODE=static`. No API URL or paid odds key
   is needed in this static project.
4. Create a Vercel personal access token with access to that project. Add this
   **GitHub Actions repository secret** under Settings → Secrets and variables →
   Actions:

   | Secret | Value |
   | --- | --- |
   | `VERCEL_TOKEN` | Vercel access token for the deployment project |

   The non-secret Vercel team and project IDs for the linked
   `court-vision-basketball` project are already in the workflow.

5. Run **Actions → Refresh data and deploy Vercel snapshot → Run workflow** once.
   Confirm the action shows `nba` and `wnba` export counts, a successful Vercel
   build, and a production deployment URL. Check Games, Props, and Analysis for
   both leagues and the `data/manifest.json` timestamp. Keep `frontend/.vercel/`
   local; do not commit token files or `.env` files.

The workflow uses `vercel build --prod` followed by
`vercel deploy --prebuilt --archive=tgz --prod`, Vercel's supported prebuilt
deployment flow. It passes `NEXT_EXPORT=1` at build time and requires no Vercel
system environment variables at build time. The archive option keeps generated
player files from hitting CLI file-count limits.

## Refresh and rate-limit behavior

The two refresh scripts fetch current-season historical box scores, not the
multi-gigabyte Kaggle archive. They make six NBA Stats calls at most on the
daily run, pause between calls, and retry failed requests at most twice with
bounded backoff (up to 30 seconds for a provider `Retry-After` response). The API reads and export then run from those local
files. The workflow does **not** call a paid odds-refresh endpoint, does not set
`ODDS_API_KEY`/`WNBA_ODDS_API_KEY`, and does not expose a public API endpoint for
refreshing paid quotes. A GitHub manual dispatch is for a free-data snapshot
refresh; do not invoke it in a tight loop when an upstream service is throttling.

The daily job starts from a clean checkout. Its refreshed files are included in
that deployment's exported JSON but are not written back to Git. Each run
therefore fetches and merges the **full current season** deterministically, so
early-season games remain available later in the year. A future incremental
provider with a true update cursor will need persistent state outside the
ephemeral runner.

## Storage and plan limits

The October 4 refreshed snapshot contains **1,130 JSON files, 17,614,368
bytes**. The verified preview build output contains **1,198 files, 18,927,004
bytes** (about 18.1 MiB); the production-target build of the same data was
18,926,376 bytes. This is the present deployment-asset requirement,
not a permanent fixed quota: new games and players will grow it. The CLI
uploaded 16.8 MB of source for the refreshed preview. Vercel's 100 MB Hobby
CLI limit applies to source uploads, not a stated persistent storage allowance.

The static deployment stores the built frontend and generated JSON as Vercel
deployment assets; it does not need Vercel Blob, a database, or a persistent
Vercel filesystem. To get the **exact size for this revision**, read the
`Snapshot payload: … bytes` and `Vercel build output: … bytes` lines in the
workflow log, then the Vercel deployment Resources tab. The JSON size changes
with player counts and seasons, so do not
mistake the source `data/` directory or Vercel's build disk for the deployed
storage size. If a deployment exceeds Vercel's CLI upload limits, it will fail
and the prior snapshot remains live.

Current official limits relevant to this design:

| Resource | Hobby limit / included use | Relevance |
| --- | --- | --- |
| CLI source upload | 100 MB | Vercel enforces an upload limit for Hobby CLI deployments. |
| Build disk | 23 GB | Temporary build capacity, **not** persistent app storage. |
| Build time | 45 minutes | The Vercel build must finish within this limit. |
| Fast Data Transfer | 100 GB included | Public traffic consumes transfer; this is not storage. |
| Vercel Blob | 1 GB-month included | **Not used** by this static deployment. |
| Vercel cron | At most once per day | **Not used**; GitHub Actions schedules this workflow. |

Vercel's [deployment limits](https://vercel.com/docs/limits),
[Blob limits](https://vercel.com/docs/vercel-blob/usage-and-pricing),
[data transfer pricing](https://vercel.com/docs/manage-cdn-usage), and
[cron limits](https://vercel.com/docs/cron-jobs/usage-and-pricing) can change.
Hobby is restricted to [personal, non-commercial use](https://vercel.com/docs/plans/hobby);
use an appropriate paid plan before offering a commercial public product.

For updates within minutes of every new game, the static daily design is not
sufficient. That requires a persistent data store and a scheduler/API worker
with a known provider quota; it should be designed and measured separately.
