import type { ChartRecord } from "@/lib/types";
import { gradeSide } from "@/lib/hit-rate.mjs";
import styles from "./AnalysisPanels.module.css";

/** The chart's data as a table: same games, same line, with the result written out in words. */
export function GameLogTable({ games, line, stat }: { games: ChartRecord[]; line: number; stat: string }) {
  if (!games.length) return null;
  return (
    <details className={styles.log}>
      <summary>View these {games.length} games as a table</summary>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <caption className="sr-only">{stat} by game against a line of {line}</caption>
          <thead><tr><th scope="col">Date</th><th scope="col">Opponent</th><th scope="col">{stat}</th><th scope="col">vs {line}</th></tr></thead>
          <tbody>
            {games.map((game, index) => {
              const result = game.value == null ? null : gradeSide(game.value, line, "Over");
              return (
                <tr key={`${game.date}-${index}`}>
                  <td>{game.date}</td>
                  <td>{game.opponent}</td>
                  <td>{game.value ?? "—"}</td>
                  <td className={result === "hit" ? styles.over : result === "miss" ? styles.under : ""}>
                    {result === "hit" ? "Over" : result === "miss" ? "Under" : result === "push" ? "Push" : "No result"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}
