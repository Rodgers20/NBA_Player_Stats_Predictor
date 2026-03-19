# Implementation Plan: Dashboard Premium Redesign

## Task Type
- [x] Frontend (CSS + Dash layout)
- [ ] Backend
- [ ] Fullstack

## Technical Solution

Full CSS overhaul of `dashboard/assets/custom.css` with glassmorphism, premium dark aesthetics,
micro-animations, and polished typography. Targeted layout tweaks in `dashboard/app.py` to
add gradient/glow classes. No backend changes required.

**Design System:**
- Style: Dark Mode (OLED) + Glassmorphism hybrid
- Colors: Deep navy `#060B18` bg, slate cards with glass effect, electric teal `#14b8a6` accents, violet `#8B5CF6` secondary accent
- Typography: Google Fonts — `Inter` (UI text) + `Fira Code` (numbers/stats)
- Effects: backdrop-blur, animated mesh gradient, glow halos, shimmer hover, smooth spring transitions

---

## Implementation Steps

### Step 1 — Design Token Overhaul
File: `dashboard/assets/custom.css`
- Replace all CSS variables with expanded premium design token set
- Add: gradient palette, glow values, blur levels, animation timing vars
- Add Google Fonts import (Inter + Fira Code)

### Step 2 — Animated Background
- Multi-layer mesh gradient on `body` using `radial-gradient` at multiple positions
- Subtle animated "aurora" effect using keyframe animation on a pseudo-element
- Deep `#060B18` base with teal + violet glows

### Step 3 — Premium Navigation Bar
- Floating pill navbar (add margin from edges, rounded corners)
- Glassmorphism: `backdrop-filter: blur(20px)`, `rgba(6,11,24,0.8)` bg
- Brand logo: animated gradient text (teal→violet)
- Nav links: pill shape with smooth fill-in on active
- Teal glow on active nav link

### Step 4 — Glassmorphism Cards
- All `.card` → glass effect: `backdrop-filter: blur(12px)`, translucent bg
- Gradient border using `border-image` or pseudo-element technique
- Hover: elevate with `box-shadow` glow + subtle translate(-2px)
- Inner top highlight: thin white `1px solid rgba(255,255,255,0.08)` top border

### Step 5 — Premium Stat Cards (Mini)
- Larger number typography using Fira Code
- Gradient left-border accent (3px teal gradient)
- Hover: teal glow box-shadow + slide-up animation
- Active state: full teal gradient background with glow

### Step 6 — Tab Groups
- Pill-shaped tabs with smooth sliding indicator
- Active tab: gradient background (teal), white text, glow
- Tab group container: glass pill background

### Step 7 — Badges & Labels
- Bigger, rounder badges with backdrop blur
- High: teal gradient | Mid: amber gradient | Low: red gradient
- Subtle pulsing animation on "live" badges

### Step 8 — Props Cards (Best Props page)
- Premium gradient border cards (animated gradient border)
- EV badge: large, glowing amber number
- Hit rate: circular progress indicator concept (CSS-only arc)
- Odds badge: electric teal pill
- Hover: full card glass lift + gradient border brightens

### Step 9 — Charts Color Scheme
- Update Plotly figure defaults in `app.py`
- Bar color: teal `#14b8a6` above threshold, red `#ef4444` below
- Line charts: electric teal with 20% fill area
- Grid: very subtle `rgba(255,255,255,0.04)`
- Chart bg: transparent (shows glass card beneath)

### Step 10 — Micro-Animations
- Page load: fade-in + slide-up on `.card` elements (staggered with nth-child delays)
- Stat number count-up: CSS counter animation on load
- Skeleton loading: shimmer gradient sweep animation
- Button/tab click: subtle scale(0.97) press feedback
- Scrollbar: thin custom dark scrollbar with teal hover

### Step 11 — Responsive Polish
- Mobile: condensed nav (icon only), single column, swipeable tabs
- Tablet: 2-column grid for props cards

---

## Key Files

| File | Operation | Description |
|------|-----------|-------------|
| `dashboard/assets/custom.css` | Rewrite | Complete premium CSS overhaul (~600 lines) |
| `dashboard/app.py` | Targeted edits | Update Plotly chart colors + add CSS classes to key layout elements |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| `backdrop-filter` not supported everywhere | Add `-webkit-backdrop-filter` prefix + solid fallback bg |
| Glassmorphism reduces text contrast | Test each glass card — maintain 4.5:1 contrast ratio |
| Dash dropdown CSS conflicts | Keep existing Dash token overrides, enhance them |
| Animated bg causes janky performance | Use `will-change: transform` on animated elements; `@media (prefers-reduced-motion)` fallback |
| app.py layout structure is complex (2400 lines) | Only add className attributes where clearly needed; no structural changes |

---

## SESSION_ID (for /ccg:execute use)
- CODEX_SESSION: N/A (planning only)
- GEMINI_SESSION: N/A (planning only)
