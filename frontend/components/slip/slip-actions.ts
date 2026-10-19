import { encodeSlip, slipText, type SlipLeg, type SlipMode } from '@/lib/bet-slip.mjs';
import { drawSlipImage } from './slip-image';

export const slipPath = (legs: SlipLeg[], mode: SlipMode, stake: number | null) => `/slip?s=${encodeSlip(legs, mode, stake)}`;

export async function copy(text: string): Promise<boolean> {
  try { await navigator.clipboard.writeText(text); return true; } catch { return false; }
}
export const copySlipText = (legs: SlipLeg[], mode: SlipMode, stake: number | null) => copy(slipText(legs, mode, stake));

/** The slip as a PNG: the phone share sheet where there is one, otherwise a download. */
export async function shareSlipImage(legs: SlipLeg[], mode: SlipMode, stake: number | null): Promise<'shared' | 'saved' | 'failed'> {
  try {
    const blob = await drawSlipImage(legs, mode, stake);
    const file = new File([blob], 'basketball-slip.png', { type: 'image/png' });
    if (navigator.canShare?.({ files: [file] })) {
      try { await navigator.share({ files: [file], title: 'Bet slip' }); return 'shared'; }
      catch (e) { if ((e as Error).name === 'AbortError') return 'shared'; }
    }
    const url = URL.createObjectURL(blob);
    Object.assign(document.createElement('a'), { href: url, download: file.name }).click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return 'saved';
  } catch { return 'failed'; }
}
