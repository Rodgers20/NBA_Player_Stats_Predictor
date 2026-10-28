const ISO_DAY = /^\d{4}-\d{2}-\d{2}$/;

/** The Eastern calendar day ("YYYY-MM-DD"), which is how the NBA/WNBA slate is dated. */
export function todayEt(now = new Date()) {
  return now.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}

/**
 * True when a snapshot's slate date is before today (ET), so its picks are yesterday's.
 * A missing or malformed date proves nothing, so it is never reported as past.
 */
export function isPastSlate(targetDate, now = new Date()) {
  if (typeof targetDate !== 'string' || !ISO_DAY.test(targetDate)) return false;
  if (Number.isNaN(Date.parse(`${targetDate}T12:00:00Z`))) return false;
  return targetDate < todayEt(now);
}
