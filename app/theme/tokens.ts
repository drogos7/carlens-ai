/**
 * Design tokens shared across web and native (Expo).
 * Adjust here to keep iOS / Android / web visually aligned.
 */
export const colors = {
  background: '#0d0f12',
  surface: '#161a20',
  surfaceElevated: '#1e232b',
  border: '#2a3140',
  textPrimary: '#f0f2f5',
  textSecondary: '#9aa3b2',
  accent: '#e8a838',
  accentMuted: '#c4922f',
  danger: '#e85d5d',
  success: '#5de8a8',
} as const;

export const space = {
  xs: 6,
  sm: 10,
  md: 16,
  lg: 24,
  xl: 32,
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
} as const;

export const font = {
  title: 22,
  heading: 17,
  body: 15,
  caption: 13,
} as const;
