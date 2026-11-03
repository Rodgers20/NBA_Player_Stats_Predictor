# API-Football-only player portraits

Player lists and profile banners use API-Football portraits. Wikipedia and
unidentified legacy entries are refused by both cached-path lookup and resolution.
Missing or ambiguous names render initials. Failed CDN downloads retry after five
minutes; a newly available squad portrait bypasses a previous miss immediately.

The headshot URL includes `v=api-football-1` to bypass previously cached Wikipedia
images. File responses infer the correct media type, including PNG. The info
endpoint includes the provider name. Original files are preserved in a backup.

Refresh results: 1963 cached portrait entries, 72 names
without a matched/downloaded portrait. Names can include both abbreviated and
full forms of the same player. The run used 2 API operations and
reused 60 cached squads. The active cache is on T7.

Backup: `/Volumes/T7/FUTBALL-data/backups/headshots-before-provider-only-1789363711727069000`.

Verification: provider-only regression tests passed; full suite had 221 passes
and one skip. Frontend production build and TypeScript passed. Browser verified
Saka/Yamal PNG responses, a loaded player-profile portrait, initials after a
simulated missing image, no Wikipedia requests and no JavaScript errors.
