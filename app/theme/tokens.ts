/**
 * Design tokens shared across web and native (Expo).
 * Values aligned with the CarLens AI design system spec.
 */
export const colors = {
  background: '#0A0B0D',
  backgroundSoft: '#0E1012',
  surface: '#141720',
  surfaceElevated: '#141720',
  surfaceGlass: '#1A1D22',
  surfaceInset: '#1A1D22',
  border: '#2E3138',
  borderStrong: '#2E3138',
  divider: '#2E3138',
  textPrimary: '#ECEDEF',
  textSecondary: '#9DA1AD',
  textMuted: '#666870',
  textFaint: '#3A3D44',
  accent: '#FFAA28',
  accentDark: '#E07C00',
  accentSoft: 'rgba(255,170,40,0.10)',
  accentDeep: '#1A0E00',
  teal: '#5BC4A0',
  blue: '#378ADD',
  danger: '#E24B4A',
  dangerSoft: 'rgba(226,75,74,0.06)',
  warningSurface: 'rgba(255,170,40,0.05)',
  gridLine: 'rgba(255,170,40,0.03)',
} as const;

export const space = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 20,
  xxl: 28,
  xxxl: 40,
} as const;

export const radius = {
  sm: 6,
  md: 9,
  lg: 12,
  xl: 16,
  xxl: 20,
  pill: 999,
} as const;

export const font = {
  hero: 28,
  title: 20,
  heading: 16,
  body: 14,
  caption: 12,
  micro: 11,
  nano: 10,
} as const;
