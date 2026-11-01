import type { CSSProperties, ReactNode } from "react";
import styles from "./hero.module.css";

export type HeroSide = { kicker?: string; name: ReactNode; sub: string; logo: ReactNode; color: string };

/**
 * Split team-colour banner shared by NBA, WNBA and college game pages: logo + name on each side, and in the middle a
 * status line, the big score (or AWAY @ HOME), and spread / total pills.
 */
export function GameHero({ away, home, status, center, label, pills, foot }: {
  away: HeroSide; home: HeroSide; status: string; center: ReactNode; label?: string; pills: string[]; foot?: string;
}) {
  const style = { "--away-color": away.color, "--home-color": home.color } as CSSProperties;
  const side = (s: HeroSide, right: boolean) => (
    <div className={`${styles.team} ${right ? styles.home : ""}`}>
      {!right && s.logo}
      <div><small>{s.kicker}</small><span className={styles.name}>{s.name}</span><small>{s.sub}</small></div>
      {right && s.logo}
    </div>
  );
  return (
    <div className={styles.hero} style={style}>
      <div className={styles.texture} aria-hidden />
      {side(away, false)}
      <div className={styles.center}>
        <p className="text-xs opacity-80">{status}</p>
        <div className={styles.score}>{center}</div>
        {label && <p className="text-xs opacity-80">{label}</p>}
        {pills.length > 0 && <div className={styles.pills}>{pills.map(p => <span key={p}>{p}</span>)}</div>}
        {foot && <p className="mt-2 text-xs opacity-75">{foot}</p>}
      </div>
      {side(home, true)}
    </div>
  );
}

/** "120.7 – 114.4" or "ATL @ NOP" in the hero's score slot. */
export const HeroPair = ({ a, b, sep = "–" }: { a: ReactNode; b: ReactNode; sep?: string }) => <>{a}<span className="mx-3 opacity-50">{sep}</span>{b}</>;
