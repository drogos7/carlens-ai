import { useCallback, useMemo, useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Image,
  Platform,
  Pressable,
  ScrollView,
  StatusBar as RNStatusBar,
  StyleSheet,
  Text,
  TextInput,
  useWindowDimensions,
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import { Feather } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import {
  analyzeErrorImage,
  lookupDiagnosticCode,
  type AnalyzeResponse,
  type DecodeSegment,
} from './lib/api';
import { colors, font, radius, space } from './theme/tokens';

const amberGradient = [colors.accent, colors.accentDark] as const;

const monoFont = Platform.select({ ios: 'Menlo', android: 'monospace', default: 'monospace' });

function formatApiError(err: unknown): string {
  if (err instanceof Error) {
    const anyErr = err as Error & { body?: unknown; status?: number };
    if (anyErr.body && typeof anyErr.body === 'object' && anyErr.body !== null && 'detail' in anyErr.body) {
      const detail = (anyErr.body as { detail: unknown }).detail;
      if (typeof detail === 'object' && detail !== null && 'message' in detail) {
        const msg = (detail as { message?: string }).message;
        if (msg) return msg;
      }
      if (typeof detail === 'string') return detail;
    }
    return err.message;
  }
  return 'An unknown error occurred.';
}

function difficultyTone(value: AnalyzeResponse['estimated_difficulty']) {
  if (value === 'Easy') return colors.teal;
  if (value === 'Medium') return colors.accent;
  return colors.danger;
}

function ScanIcon({ size = 18, color }: { size?: number; color: string }) {
  const stroke = 2;
  const arm = size * 0.38;
  const corner = size * 0.24;
  const cornerStyle = {
    position: 'absolute' as const,
    width: arm,
    height: corner,
    borderColor: color,
  };

  return (
    <View style={{ width: size, height: size }}>
      <View
        style={[
          cornerStyle,
          { top: 0, left: 0, borderTopWidth: stroke, borderLeftWidth: stroke, borderTopLeftRadius: 3 },
        ]}
      />
      <View
        style={[
          cornerStyle,
          { top: 0, right: 0, borderTopWidth: stroke, borderRightWidth: stroke, borderTopRightRadius: 3 },
        ]}
      />
      <View
        style={[
          cornerStyle,
          { bottom: 0, left: 0, borderBottomWidth: stroke, borderLeftWidth: stroke, borderBottomLeftRadius: 3 },
        ]}
      />
      <View
        style={[
          cornerStyle,
          { bottom: 0, right: 0, borderBottomWidth: stroke, borderRightWidth: stroke, borderBottomRightRadius: 3 },
        ]}
      />
      <View
        style={{
          position: 'absolute',
          left: size * 0.18,
          right: size * 0.18,
          top: size / 2 - stroke / 2,
          height: stroke,
          backgroundColor: color,
          borderRadius: stroke,
        }}
      />
    </View>
  );
}

function WordMark() {
  return (
    <View style={styles.wordmark}>
      <LinearGradient colors={[...amberGradient]} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.wordmarkIcon}>
        <Text style={styles.wordmarkIconText}>C</Text>
      </LinearGradient>
      <View style={styles.wordmarkCopy}>
        <Text style={styles.wordmarkText}>
          CarLens <Text style={styles.wordmarkAccent}>AI</Text>
        </Text>
        <Text style={styles.wordmarkTagline}>Scan Detect Repair</Text>
      </View>
    </View>
  );
}

function NavLink({ label, active }: { label: string; active?: boolean }) {
  return (
    <Pressable style={[styles.navLink, active && styles.navLinkActive]}>
      <Text style={[styles.navLinkText, active && styles.navLinkTextActive]}>{label}</Text>
    </Pressable>
  );
}


function CardEyebrow({ icon, label }: { icon: string; label: string }) {
  return (
    <View style={styles.eyebrowRow}>
      <Text style={styles.eyebrowIcon}>{icon}</Text>
      <Text style={styles.eyebrowText}>{label}</Text>
    </View>
  );
}

function DifficultyBadge({ value }: { value: AnalyzeResponse['estimated_difficulty'] }) {
  const tone = difficultyTone(value);
  return (
    <View style={[styles.difficultyBadge, { backgroundColor: `${tone}1E`, borderColor: `${tone}55` }]}>
      <View style={[styles.difficultyDot, { backgroundColor: tone }]} />
      <Text style={[styles.difficultyText, { color: tone }]}>{value}</Text>
    </View>
  );
}

function BulletItem({ text, tone = colors.accent }: { text: string; tone?: string }) {
  return (
    <View style={styles.bulletRow}>
      <View style={[styles.bulletDot, { backgroundColor: tone }]} />
      <Text style={styles.bulletText}>{text}</Text>
    </View>
  );
}

function StepItem({ index, text }: { index: number; text: string }) {
  return (
    <View style={styles.stepRow}>
      <View style={styles.stepNumber}>
        <Text style={styles.stepNumberText}>{index + 1}</Text>
      </View>
      <Text style={styles.stepText}>{text}</Text>
    </View>
  );
}

function DecodeBreakdown({
  title,
  summary,
  segments,
}: {
  title: string;
  summary: string;
  segments: DecodeSegment[];
}) {
  return (
    <View style={styles.decodeCard}>
      <CardEyebrow icon="◇" label={title} />
      <Text style={styles.decodeSummary}>{summary}</Text>
      {segments.map((seg, idx) => (
        <View key={`${seg.label}-${idx}`} style={styles.decodeRow}>
          <View style={styles.decodeMeta}>
            <Text style={styles.decodePos}>{seg.positions ?? seg.position ?? ''}</Text>
            <Text style={styles.decodeVal}>{seg.value}</Text>
          </View>
          <Text style={styles.decodeLabel}>{seg.label}</Text>
          <Text style={styles.decodeMeaning}>{seg.meaning}</Text>
        </View>
      ))}
    </View>
  );
}

function UploadGlyph() {
  return (
    <View style={styles.uploadGlyph}>
      <View style={styles.uploadGlyphInner}>
        <Text style={styles.uploadGlyphArrow}>↑</Text>
      </View>
    </View>
  );
}

export default function App() {
  const { width } = useWindowDimensions();
  const isWide = width >= 880;
  const [uri, setUri] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [lookupLoading, setLookupLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [codeInput, setCodeInput] = useState('');

  const statusText = useMemo(() => {
    if (loading) return 'Scanning';
    if (result) return 'Ready';
    if (uri) return 'Image loaded';
    return 'Ready';
  }, [loading, result, uri]);

  const pickImage = useCallback(async (mode: 'library' | 'camera') => {
    setResult(null);
    setErrorMessage(null);

    const permission =
      mode === 'camera'
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permission.granted) {
      Alert.alert(
        'Permission required',
        mode === 'camera' ? 'Camera access is required.' : 'Photo access is required.',
      );
      return;
    }

    const picker =
      mode === 'camera'
        ? ImagePicker.launchCameraAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.9 })
        : ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.9 });

    const response = await picker;
    if (!response.canceled && response.assets[0]?.uri) setUri(response.assets[0].uri);
  }, []);

  const scanImage = useCallback(async () => {
    if (!uri) return;
    setLoading(true);
    setResult(null);
    setErrorMessage(null);
    try {
      const data = await analyzeErrorImage(uri);
      setResult(data);
    } catch (e) {
      const anyErr = e as Error & { status?: number; body?: unknown };
      let message = formatApiError(e);
      if (anyErr.status === 422 && anyErr.body && typeof anyErr.body === 'object') {
        const detail = (anyErr.body as { detail?: { hints?: string[]; ocr_text?: string } }).detail;
        message = 'Could not find an error code in the image.';
        if (detail && typeof detail === 'object' && Array.isArray(detail.hints)) {
          message = [message, ...detail.hints].join('\n- ');
        }
        if (detail && typeof detail === 'object' && typeof detail.ocr_text === 'string' && detail.ocr_text.trim()) {
          message = `${message}\n\nDetected text:\n${detail.ocr_text.trim().slice(0, 420)}`;
        }
      }
      setErrorMessage(message);
      if (Platform.OS !== 'web') Alert.alert('Scan result', message);
    } finally {
      setLoading(false);
    }
  }, [uri]);

  const lookupCode = useCallback(async () => {
    const code = codeInput.trim();
    if (!code) {
      setErrorMessage('Enter an OBD code (e.g. P0420) or a 17-character VIN.');
      return;
    }
    setLookupLoading(true);
    setResult(null);
    setErrorMessage(null);
    try {
      const data = await lookupDiagnosticCode(code);
      setResult(data);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not look up the code.';
      setErrorMessage(message);
      if (Platform.OS !== 'web') Alert.alert('Code lookup', message);
    } finally {
      setLookupLoading(false);
    }
  }, [codeInput]);

  const topInset = Platform.OS === 'android' ? RNStatusBar.currentHeight ?? 0 : 0;

  return (
    <View style={[styles.root, { paddingTop: topInset }]}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <View style={styles.workspace}>
          <View pointerEvents="none" style={styles.gridBackdrop}>
            {Array.from({ length: 36 }).map((_, i) => (
              <View key={`h-${i}`} style={[styles.gridLine, styles.gridLineH, { top: i * 32 }]} />
            ))}
            {Array.from({ length: 40 }).map((_, i) => (
              <View key={`v-${i}`} style={[styles.gridLine, styles.gridLineV, { left: i * 32 }]} />
            ))}
          </View>
          <View style={styles.workspaceInner}>
            <View style={styles.topBar}>
              <WordMark />
              <View style={styles.navRow}>
                <NavLink label="Settings" />
              </View>
            </View>

            <View style={styles.hero}>
              <Text style={styles.heroTitle}>Diagnose your car in seconds.</Text>
              <Text style={styles.heroCopy}>
                Photo your scanner screen — we detect the code & give you a repair guide.
              </Text>
            </View>

            <View style={[styles.grid, isWide && styles.gridWide]}>
              {/* LEFT — Capture scanner output */}
              <View style={[styles.panel, isWide && styles.panelLeft]}>
                <View style={styles.panelHeader}>
                  <CardEyebrow icon="◉" label="Capture scanner output" />
                  <View style={styles.statusPill}>
                    <View style={styles.statusDot} />
                    <Text style={styles.statusLabel}>{statusText}</Text>
                  </View>
                </View>

                <Pressable style={styles.uploadZone} onPress={() => void pickImage('camera')}>
                  {uri ? (
                    <Image source={{ uri }} style={styles.previewImage} resizeMode="cover" />
                  ) : (
                    <>
                      <UploadGlyph />
                      <Text style={styles.uploadLabel}>Drop a photo or capture the scanner display</Text>
                      <Text style={styles.uploadHint}>PNG / JPG up to ~10 MB. Keep the code line in focus.</Text>
                    </>
                  )}
                </Pressable>

                <View style={styles.buttonRow}>
                  <Pressable style={styles.secondaryButton} onPress={() => void pickImage('camera')}>
                    <Feather name="camera" size={14} color={colors.accent} />
                    <Text style={styles.secondaryButtonText}>Camera</Text>
                  </Pressable>
                  <Pressable style={styles.secondaryButton} onPress={() => void pickImage('library')}>
                    <Feather name="image" size={14} color={colors.accent} />
                    <Text style={styles.secondaryButtonText}>Gallery</Text>
                  </Pressable>
                </View>

                <Pressable
                  onPress={() => void scanImage()}
                  disabled={!uri || loading}
                  style={(!uri || loading) && styles.primaryButtonDisabled}
                >
                  <LinearGradient
                    colors={[...amberGradient]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[styles.primaryButton, (!uri || loading) && styles.primaryButtonInnerDisabled]}
                  >
                    {loading ? (
                      <ActivityIndicator color={colors.accentDeep} />
                    ) : (
                      <>
                        <ScanIcon size={18} color={colors.accentDeep} />
                        <Text style={styles.primaryButtonText}>Scan & Detect Code</Text>
                      </>
                    )}
                  </LinearGradient>
                </Pressable>

                <View style={styles.lookupBlock}>
                  <Text style={styles.lookupLabel}>Already know the code or VIN?</Text>
                  <View style={styles.lookupRow}>
                    <TextInput
                      value={codeInput}
                      onChangeText={setCodeInput}
                      autoCapitalize="characters"
                      autoCorrect={false}
                      placeholder="P0420 or 17-char VIN"
                      placeholderTextColor={colors.textFaint}
                      style={styles.codeInput}
                    />
                    <Pressable
                      style={[styles.lookupButton, (!codeInput.trim() || lookupLoading) && styles.lookupButtonDisabled]}
                      onPress={() => void lookupCode()}
                      disabled={!codeInput.trim() || lookupLoading}
                    >
                      {lookupLoading ? (
                        <ActivityIndicator color={colors.accent} />
                      ) : (
                        <Text
                          style={[
                            styles.lookupButtonText,
                            !codeInput.trim() && styles.lookupButtonTextDisabled,
                          ]}
                        >
                          Lookup
                        </Text>
                      )}
                    </Pressable>
                  </View>
                </View>
              </View>

              {/* RIGHT — Diagnostic result */}
              <View style={[styles.panel, isWide && styles.panelRight]}>
                <View style={styles.panelHeader}>
                  <CardEyebrow icon="◈" label="Diagnostic result" />
                  {result ? <DifficultyBadge value={result.estimated_difficulty} /> : <Text style={styles.panelHint}>Awaiting scan</Text>}
                </View>

                {result ? (
                  <View style={styles.resultBody}>
                    <View style={styles.codeBlock}>
                      <Text style={styles.resultKind}>
                        {result.scan_type === 'vin' ? 'Vehicle VIN' : 'Fault code'}
                      </Text>
                      <Text style={styles.codeText}>
                        {result.scan_type === 'vin' ? result.detected_vin ?? result.detected_code : result.detected_code}
                      </Text>
                      {result.scan_type === 'vin' && result.vehicle_make ? (
                        <Text style={styles.vehicleMeta}>
                          {[result.vehicle_year, result.vehicle_make, result.vehicle_model, result.vehicle_engine]
                            .filter(Boolean)
                            .join(' · ')}
                        </Text>
                      ) : null}
                      {result.scan_type !== 'vin' ? (
                        <Text style={styles.codeDescription}>{result.probable_cause}</Text>
                      ) : null}
                    </View>

                    {result.dtc_decode ? (
                      <DecodeBreakdown
                        title="OBD code structure"
                        summary={result.dtc_decode.summary}
                        segments={result.dtc_decode.segments}
                      />
                    ) : null}

                    {result.scan_type !== 'vin' ? (
                      <View style={styles.section}>
                        <CardEyebrow icon="◆" label="Probable cause" />
                        <View style={styles.sectionContent}>
                          {result.probable_cause
                            .split(/\.\s+/)
                            .filter((s) => s.trim().length > 4)
                            .slice(0, 4)
                            .map((sentence, idx) => (
                              <BulletItem key={`cause-${idx}`} text={sentence.replace(/\.$/, '').trim()} />
                            ))}
                        </View>
                      </View>
                    ) : (
                      <View style={styles.section}>
                        <CardEyebrow icon="◆" label="Vehicle details" />
                        <View style={styles.sectionContent}>
                          {result.probable_cause
                            .split('\n')
                            .filter((line) => line.trim().length > 0)
                            .map((line, idx) => (
                              <BulletItem key={`vin-detail-${idx}`} text={line.trim()} />
                            ))}
                        </View>
                      </View>
                    )}

                    <View style={styles.section}>
                      <CardEyebrow icon="⚙" label={result.scan_type === 'vin' ? 'Next steps' : 'Recommended workflow'} />
                      <View style={styles.sectionContent}>
                        {result.step_by_step_fix.map((step, idx) => (
                          <StepItem key={`step-${idx}`} index={idx} text={step} />
                        ))}
                      </View>
                    </View>

                    <View style={styles.warningCard}>
                      <View style={styles.warningHeader}>
                        <Text style={styles.warningIcon}>!</Text>
                        <Text style={styles.warningTitle}>Safety alert</Text>
                      </View>
                      <Text style={styles.warningText}>{result.safety_warning}</Text>
                    </View>
                  </View>
                ) : (
                  <View style={styles.emptyState}>
                    <View style={styles.emptyGlyph}>
                      <Text style={styles.emptyGlyphText}>?</Text>
                    </View>
                    <Text style={styles.emptyTitle}>Repair guide will appear here</Text>
                    <Text style={styles.emptyText}>
                      Capture a scanner display or enter a code on the left to generate a step-by-step diagnostic report.
                    </Text>
                    <View style={styles.emptyHints}>
                      <BulletItem text="Probable cause analysis" tone={colors.accent} />
                      <BulletItem text="Inspection checklist" tone={colors.teal} />
                      <BulletItem text="Repair workflow with safety notes" tone={colors.blue} />
                    </View>
                  </View>
                )}

                {errorMessage ? (
                  <View style={styles.errorCard}>
                    <Text style={styles.errorTitle}>Could not complete scan</Text>
                    <Text style={styles.errorText}>{errorMessage}</Text>
                  </View>
                ) : null}
              </View>
            </View>
          </View>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.background,
  },
  scroll: {
    flexGrow: 1,
    paddingHorizontal: space.xl,
    paddingVertical: space.xl,
    alignItems: 'center',
  },
  workspace: {
    width: '100%',
    maxWidth: 1200,
    borderRadius: radius.xxl,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.backgroundSoft,
    overflow: 'hidden',
  },
  workspaceInner: {
    padding: space.xl,
    gap: space.xl,
  },
  gridBackdrop: {
    ...StyleSheet.absoluteFillObject,
    overflow: 'hidden',
  },
  gridLine: {
    position: 'absolute',
    backgroundColor: colors.gridLine,
  },
  gridLineH: {
    left: 0,
    right: 0,
    height: 1,
  },
  gridLineV: {
    top: 0,
    bottom: 0,
    width: 1,
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: space.lg,
    flexWrap: 'wrap',
  },
  wordmark: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.md,
  },
  wordmarkIcon: {
    width: 36,
    height: 36,
    borderRadius: 9,
    alignItems: 'center',
    justifyContent: 'center',
  },
  wordmarkIconText: {
    color: colors.accentDeep,
    fontSize: 18,
    fontWeight: '700',
  },
  wordmarkCopy: {
    gap: 2,
  },
  wordmarkText: {
    color: colors.textPrimary,
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: -0.2,
    lineHeight: 20,
  },
  wordmarkAccent: {
    color: colors.accent,
  },
  wordmarkTagline: {
    color: colors.accent,
    fontSize: font.micro,
    fontWeight: '600',
    letterSpacing: 0.4,
  },
  navRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.xs,
  },
  navLink: {
    paddingHorizontal: space.md,
    paddingVertical: 6,
    borderRadius: radius.pill,
  },
  navLinkActive: {
    backgroundColor: colors.surface,
    borderWidth: 0.5,
    borderColor: colors.border,
  },
  navLinkText: {
    color: colors.textMuted,
    fontSize: font.caption,
    fontWeight: '500',
  },
  navLinkTextActive: {
    color: colors.textPrimary,
  },
  hero: {
    gap: 6,
    paddingVertical: space.sm,
  },
  heroTitle: {
    color: colors.textPrimary,
    fontSize: font.hero,
    fontWeight: '500',
    lineHeight: 34,
    letterSpacing: -0.56,
  },
  heroCopy: {
    color: colors.textSecondary,
    fontSize: font.body,
    lineHeight: 22,
    marginTop: 2,
    maxWidth: 640,
  },
  grid: {
    gap: space.xl,
  },
  gridWide: {
    flexDirection: 'row',
    alignItems: 'flex-start',
  },
  panel: {
    flex: 1,
    borderRadius: radius.xl,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    padding: space.xl,
    gap: space.lg,
  },
  panelLeft: {
    flex: 1,
  },
  panelRight: {
    flex: 1.15,
  },
  panelHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: space.sm,
  },
  panelHint: {
    color: colors.textMuted,
    fontSize: font.nano,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  eyebrowRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
  },
  eyebrowIcon: {
    color: colors.accent,
    fontSize: 14,
  },
  eyebrowText: {
    color: colors.textSecondary,
    fontSize: font.nano,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1.2,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.pill,
    backgroundColor: colors.surfaceInset,
    borderWidth: 0.5,
    borderColor: colors.border,
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.teal,
  },
  statusLabel: {
    color: colors.textMuted,
    fontSize: font.nano,
    fontWeight: '500',
  },
  uploadZone: {
    minHeight: 200,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: colors.border,
    backgroundColor: colors.surfaceInset,
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.md,
    padding: space.xl,
    overflow: 'hidden',
  },
  previewImage: {
    width: '100%',
    height: 200,
  },
  uploadGlyph: {
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: colors.accentSoft,
    borderWidth: 1,
    borderColor: 'rgba(255,170,40,0.35)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: space.xs,
  },
  uploadGlyphInner: {
    width: 36,
    height: 36,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  uploadGlyphArrow: {
    color: colors.accent,
    fontSize: 18,
    fontWeight: '600',
    lineHeight: 20,
  },
  uploadLabel: {
    color: colors.textPrimary,
    fontSize: 13,
    fontWeight: '600',
    textAlign: 'center',
  },
  uploadHint: {
    color: colors.textMuted,
    fontSize: font.micro,
    textAlign: 'center',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: space.sm,
  },
  secondaryButton: {
    flex: 1,
    minHeight: 40,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    borderRadius: radius.md,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.surfaceInset,
  },
  secondaryButtonText: {
    color: colors.textPrimary,
    fontSize: font.caption,
    fontWeight: '500',
  },
  primaryButton: {
    minHeight: 48,
    borderRadius: radius.lg,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.sm,
  },
  primaryButtonDisabled: {
    opacity: 0.5,
  },
  primaryButtonInnerDisabled: {
    opacity: 0.85,
  },
  primaryButtonText: {
    color: colors.accentDeep,
    fontSize: 14,
    fontWeight: '600',
    letterSpacing: 0.2,
  },
  lookupBlock: {
    gap: space.xs,
  },
  lookupLabel: {
    color: colors.textMuted,
    fontSize: font.nano,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginTop: space.xs,
  },
  lookupRow: {
    flexDirection: 'row',
    gap: space.sm,
  },
  codeInput: {
    flex: 1,
    minHeight: 40,
    borderRadius: radius.md,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.surfaceInset,
    color: colors.textPrimary,
    paddingHorizontal: space.md,
    fontSize: 13,
    fontFamily: monoFont,
    letterSpacing: 1,
  },
  lookupButton: {
    paddingHorizontal: space.lg,
    minHeight: 40,
    borderRadius: radius.md,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.surfaceGlass,
    alignItems: 'center',
    justifyContent: 'center',
  },
  lookupButtonText: {
    color: colors.textSecondary,
    fontSize: font.caption,
    fontWeight: '500',
  },
  lookupButtonDisabled: {
    opacity: 0.5,
  },
  lookupButtonTextDisabled: {
    color: colors.textMuted,
  },
  resultBody: {
    gap: space.lg,
  },
  codeBlock: {
    padding: space.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.surfaceInset,
    borderWidth: 0.5,
    borderColor: colors.border,
    gap: space.xs,
  },
  resultKind: {
    color: colors.textMuted,
    fontSize: font.nano,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  codeText: {
    color: colors.accent,
    fontSize: 34,
    fontFamily: monoFont,
    fontWeight: '600',
    letterSpacing: 1.5,
  },
  vehicleMeta: {
    color: colors.teal,
    fontSize: font.caption,
    fontWeight: '500',
    marginTop: 4,
    marginBottom: 4,
  },
  decodeCard: {
    borderRadius: radius.lg,
    borderWidth: 0.5,
    borderColor: colors.border,
    backgroundColor: colors.surfaceInset,
    padding: space.lg,
    gap: space.sm,
  },
  decodeSummary: {
    color: colors.textSecondary,
    fontSize: font.caption,
    lineHeight: 18,
    marginBottom: space.xs,
  },
  decodeRow: {
    paddingVertical: space.sm,
    borderTopWidth: 0.5,
    borderTopColor: colors.divider,
    gap: 4,
  },
  decodeMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
  },
  decodePos: {
    color: colors.textMuted,
    fontSize: font.nano,
    fontFamily: monoFont,
    minWidth: 28,
  },
  decodeVal: {
    color: colors.accent,
    fontSize: font.caption,
    fontWeight: '600',
    fontFamily: monoFont,
    letterSpacing: 0.6,
  },
  decodeLabel: {
    color: colors.textPrimary,
    fontSize: font.caption,
    fontWeight: '500',
  },
  decodeMeaning: {
    color: colors.textMuted,
    fontSize: font.micro,
    lineHeight: 17,
  },
  codeDescription: {
    color: colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },
  section: {
    gap: space.sm,
  },
  sectionContent: {
    gap: 6,
    paddingTop: space.xs,
  },
  bulletRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: space.sm,
    paddingVertical: 4,
  },
  bulletDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginTop: 7,
  },
  bulletText: {
    flex: 1,
    color: colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: space.md,
    paddingVertical: 6,
  },
  stepNumber: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.accentSoft,
    borderWidth: 0.5,
    borderColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepNumberText: {
    color: colors.accent,
    fontSize: 11,
    fontWeight: '600',
  },
  stepText: {
    flex: 1,
    color: colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },
  difficultyBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.pill,
    borderWidth: 0.5,
  },
  difficultyDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  difficultyText: {
    fontSize: font.nano,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.6,
  },
  warningCard: {
    padding: space.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.dangerSoft,
    borderWidth: 0.5,
    borderColor: 'rgba(226,75,74,0.35)',
    gap: space.xs,
  },
  warningHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: space.sm,
  },
  warningIcon: {
    width: 18,
    height: 18,
    borderRadius: 9,
    textAlign: 'center',
    lineHeight: 18,
    color: '#FFFFFF',
    backgroundColor: colors.danger,
    fontSize: 12,
    fontWeight: '700',
    overflow: 'hidden',
  },
  warningTitle: {
    color: colors.danger,
    fontSize: font.nano,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  warningText: {
    color: colors.textSecondary,
    fontSize: 13,
    lineHeight: 20,
  },
  emptyState: {
    minHeight: 320,
    alignItems: 'center',
    justifyContent: 'center',
    gap: space.sm,
    padding: space.xl,
  },
  emptyGlyph: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.surfaceInset,
    borderWidth: 0.5,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: space.xs,
  },
  emptyGlyphText: {
    color: colors.textMuted,
    fontSize: 20,
    fontWeight: '500',
  },
  emptyTitle: {
    color: colors.textPrimary,
    fontSize: 15,
    fontWeight: '600',
  },
  emptyText: {
    color: colors.textMuted,
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
    maxWidth: 320,
  },
  emptyHints: {
    marginTop: space.md,
    width: '100%',
    maxWidth: 320,
    gap: 2,
  },
  errorCard: {
    padding: space.lg,
    borderRadius: radius.lg,
    backgroundColor: colors.dangerSoft,
    borderWidth: 0.5,
    borderColor: 'rgba(226,75,74,0.35)',
    gap: space.xs,
  },
  errorTitle: {
    color: colors.danger,
    fontSize: font.nano,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  errorText: {
    color: colors.textSecondary,
    fontSize: 12,
    lineHeight: 18,
  },
});
