import type { SlipLeg } from './bet-slip.mjs';
import type { Prop } from './types';

export function modifyLeg(leg: SlipLeg, prop: Prop | null | undefined, change: { direction?: string; line?: number }): SlipLeg;
