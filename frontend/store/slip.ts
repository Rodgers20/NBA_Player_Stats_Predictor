import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { legKey, toggleLeg, MAX_LEGS, type SlipLeg, type SlipMode } from '@/lib/bet-slip.mjs';

interface SlipState {
  legs: SlipLeg[];
  mode: SlipMode;
  stake: number | null;
  open: boolean;
  /** Bumped on every add, so the slip pill pulses once. */
  added: number;
  toggle: (leg: SlipLeg) => void;
  remove: (key: string) => void;
  clear: () => void;
  load: (legs: SlipLeg[], mode: SlipMode, stake: number | null) => void;
  setMode: (mode: SlipMode) => void;
  setStake: (stake: number | null) => void;
  setOpen: (open: boolean) => void;
}
/** The slip lives in this browser only; nothing leaves it until it is shared as a link. */
export const useSlip = create<SlipState>()(persist((set) => ({
  legs: [], mode: 'parlay', stake: null, open: false, added: 0,
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
  setOpen: (open) => set({ open }),
}), { name: 'basketball-slip-v1', skipHydration: true, partialize: ({ legs, mode, stake }) => ({ legs, mode, stake }) }));
