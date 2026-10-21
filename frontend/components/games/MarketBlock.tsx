"use client";

import { Check, Plus } from "lucide-react";
import { HitRibbon } from "@/components/ui/HitRibbon";
import type { SideWithAdd, SlipAdd } from "./useGameSlip";
import styles from "./MarketBlock.module.css";

function AddButton({ add }: { add: SlipAdd }) {
  return (
    <button type="button" className={styles.add} aria-pressed={add.inSlip} title={add.inSlip ? "Remove from slip" : "Add to slip"}
      aria-label={`${add.inSlip ? "Remove" : "Add"} ${add.label} ${add.inSlip ? "from" : "to"} slip`} onClick={add.onToggle}>
      {add.inSlip ? <Check size={14} aria-hidden="true" /> : <Plus size={14} aria-hidden="true" />}
    </button>
  );
}

/**
 * One market (money line, spread or total): two sides, each a big number, an optional "+" to put it on the slip,
 * and a real-data ribbon. `actions` adds over/under buttons for a total, which belongs to the game, not a team.
 */
export function MarketBlock({ title, sides, actions }: { title: string; sides: [SideWithAdd, SideWithAdd] | null; actions?: { text: string; add: SlipAdd }[] }) {
  return (
    <section className={styles.block}>
      <h3 className={styles.title}>{title}</h3>
      {sides && actions?.length ? (
        <div className={styles.actions}>
          {actions.map(({ text, add }) => (
            <button key={text} type="button" className={styles.action} aria-pressed={add.inSlip} onClick={add.onToggle}
              aria-label={`${add.inSlip ? "Remove" : "Add"} ${add.label} ${add.inSlip ? "from" : "to"} slip`}>
              {text} {add.inSlip ? <Check size={13} aria-hidden="true" /> : <Plus size={13} aria-hidden="true" />}
            </button>
          ))}
        </div>
      ) : null}
      {sides ? (
        <div className={styles.sides}>
          {sides.map(side => (
            <div key={side.label} className={styles.side}>
              <div className={styles.head}>
                <span className={styles.label}>{side.label}</span>
                {side.value ? <strong className={styles.value}>{side.value}</strong> : null}
                {side.add ? <AddButton add={side.add} /> : null}
              </div>
              {side.form ? <HitRibbon values={side.form.values} line={side.form.line} label={side.form.caption} flat={side.form.flat} note={side.form.note} /> : null}
            </div>
          ))}
        </div>
      ) : (
        <p className={styles.none}>No line yet</p>
      )}
    </section>
  );
}
