import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { legKey, toggleLeg, type SlipLeg } from '@/lib/bet-slip.mjs';

interface SlipState {
  legs: SlipLeg[];
  toggle: (leg: SlipLeg) => void;
  remove: (key: string) => void;
  clear: () => void;
}
export const useSlip = create<SlipState>()(persist((set) => ({
  legs: [],
  toggle: (leg) => set(state => ({ legs: toggleLeg(state.legs, leg) })),
  remove: (key) => set(state => ({ legs: state.legs.filter(leg => legKey(leg) !== key) })),
  clear: () => set({ legs: [] }),
}), { name: 'basketball-slip-v1', skipHydration: true }));
