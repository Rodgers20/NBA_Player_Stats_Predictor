import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { legKey, toggleLeg, sanitizePersistedSlip, MAX_LEGS, type SlipLeg, type SlipMode } from '@/lib/bet-slip.mjs';
import { DEFAULT_BANKROLL, sanitizeBankroll, type Bankroll } from '@/lib/bankroll.mjs';

interface SlipState {
  legs: SlipLeg[];
  mode: SlipMode;
  stake: number | null;
  /** Units × unit size feed the suggested stake. Kept when the slip is cleared. */
  bankroll: Bankroll;
  open: boolean;
  /** Bumped on every add, so the slip pill pulses once. */
  added: number;
  toggle: (leg: SlipLeg) => void;
  remove: (key: string) => void;
  clear: () => void;
  load: (legs: SlipLeg[], mode: SlipMode, stake: number | null) => void;
  setMode: (mode: SlipMode) => void;
  setStake: (stake: number | null) => void;
  setBankroll: (bankroll: Bankroll) => void;
  setOpen: (open: boolean) => void;
}
/** The slip lives in this browser only; nothing leaves it until it is shared as a link. */
export const useSlip = create<SlipState>()(persist((set) => ({
  legs: [], mode: 'parlay', stake: null, bankroll: DEFAULT_BANKROLL, open: false, added: 0,
  toggle: (leg) => set(state => {
    const legs = toggleLeg(state.legs, leg);
    if (legs.length > MAX_LEGS) return {};
    return { legs, added: legs.includes(leg) ? state.added + 1 : state.added };
  }),
  remove: (key) => set(state => ({ legs: state.legs.filter(leg => legKey(leg) !== key) })),
  clear: () => set({ legs: [], open: false }),
  load: (legs, mode, stake) => set({ legs, mode, stake, open: true }),
  setMode: (mode) => set({ mode }),
  setStake: (stake) => set({ stake }),
  setBankroll: (bankroll) => set({ bankroll: sanitizeBankroll(bankroll) }),
  setOpen: (open) => set({ open }),
}), {
  name: 'basketball-slip-v1', skipHydration: true,
  partialize: ({ legs, mode, stake, bankroll }) => ({ legs, mode, stake, bankroll }),
  // Slips saved before the bankroll existed have no such field: keep their legs, default the rest.
  merge: (persisted, current) => ({ ...current, ...sanitizePersistedSlip(persisted) }),
}));
