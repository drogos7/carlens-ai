import { useCallback, useEffect, useState } from 'react';
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
  View,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { StatusBar } from 'expo-status-bar';
import { analyzeErrorImage, fetchHealth, lookupDiagnosticCode, type AnalyzeResponse } from './lib/api';
import { colors, font, radius, space } from './theme/tokens';

function formatApiError(err: unknown): string {
  if (err instanceof Error) {
    const anyErr = err as Error & { body?: unknown; status?: number };
    if (anyErr.body && typeof anyErr.body === 'object' && anyErr.body !== null && 'detail' in anyErr.body) {
      const d = (anyErr.body as { detail: unknown }).detail;
      if (typeof d === 'object' && d !== null && 'message' in d) {
        const msg = (d as { message?: string }).message;
        if (msg) return msg;
      }
      if (typeof d === 'string') {
        if (d.includes('OPENAI_API_KEY') || d.includes('ANTHROPIC_API_KEY')) {
          return 'The API key is missing on the server. Set OPENAI_API_KEY in backend\\.env and restart uvicorn.';
        }
        return d;
      }
    }
    if (anyErr.status === 503 && err.message.includes('API key')) {
      return err.message.includes('API key')
        ? err.message
        : 'The API key is missing on the server. Set OPENAI_API_KEY in backend\\.env and restart uvicorn.';
    }
    return err.message;
  }
  return 'An unknown error occurred.';
}

function DifficultyBadge({ value }: { value: AnalyzeResponse['estimated_difficulty'] }) {
  const tone =
    value === 'Easy' ? colors.success : value === 'Medium' ? colors.accent : colors.danger;
  return (
    <View style={[styles.badge, { borderColor: tone }]}>
      <Text style={[styles.badgeText, { color: tone }]}>{value}</Text>
    </View>
  );
}

export default function App() {
  const [uri, setUri] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [lookupLoading, setLookupLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [apiReady, setApiReady] = useState<boolean | null>(null);
  const [codeInput, setCodeInput] = useState('P0420');

  useEffect(() => {
    void fetchHealth()
      .then((h) => setApiReady(h.vision_configured === true))
      .catch(() => setApiReady(false));
  }, []);

  const pickImage = useCallback(async (mode: 'library' | 'camera') => {
    setResult(null);
    setErrorMessage(null);
    if (mode === 'camera') {
      const perm = await ImagePicker.requestCameraPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Permission required', 'Camera access is required.');
        return;
      }
    } else {
      const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (!perm.granted) {
        Alert.alert('Permission required', 'Photo library access is required.');
        return;
      }
    }

    const picker =
      mode === 'camera'
        ? ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.85,
          })
        : ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.85,
          });

    const res = await picker;
    if (!res.canceled && res.assets[0]?.uri) {
      setUri(res.assets[0].uri);
    }
  }, []);

  const analyze = useCallback(async () => {
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
        const detail = (anyErr.body as { detail?: { hints?: string[] } }).detail;
        if (detail && typeof detail === 'object' && Array.isArray(detail.hints)) {
          message = [message, ...detail.hints].join('\n• ');
        }
      }
      setErrorMessage(message);
      if (Platform.OS !== 'web') {
        Alert.alert('Scan result', message);
      }
    } finally {
      setLoading(false);
    }
  }, [uri]);

  const lookupManualCode = useCallback(async () => {
    const code = codeInput.trim();
    if (!code) {
      setErrorMessage('Enter a code, for example P0420.');
      return;
    }
    setLookupLoading(true);
    setErrorMessage(null);
    setResult(null);
    try {
      const data = await lookupDiagnosticCode(code);
      setResult(data);
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Could not look up the code.';
      setErrorMessage(message);
      if (Platform.OS !== 'web') {
        Alert.alert('Code lookup', message);
      }
    } finally {
      setLookupLoading(false);
    }
  }, [codeInput]);

  const topInset = Platform.OS === 'android' ? (RNStatusBar.currentHeight ?? 0) : space.md;

  return (
    <View style={[styles.root, { paddingTop: topInset }]}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>Diagnostic scan</Text>
        <Text style={styles.subtitle}>Scanner screen photo to technical fix guide (MVP)</Text>

        {apiReady === false ? (
          <View style={styles.bannerWarn}>
            <Text style={styles.bannerTitle}>Free mode active</Text>
            <Text style={styles.bannerText}>
              Scanning uses local OCR + DB. It does not consume AI tokens. The AI key remains optional for later.
            </Text>
          </View>
        ) : null}

        <View style={styles.lookupCard}>
          <Text style={styles.sectionLabelNoTop}>Search knowledge base</Text>
          <View style={styles.lookupRow}>
            <TextInput
              value={codeInput}
              onChangeText={setCodeInput}
              autoCapitalize="characters"
              autoCorrect={false}
              placeholder="Ex: P0420"
              placeholderTextColor={colors.textSecondary}
              style={styles.input}
            />
            <Pressable
              style={[styles.btnLookup, lookupLoading && styles.btnDisabled]}
              onPress={() => void lookupManualCode()}
              disabled={lookupLoading}
            >
              {lookupLoading ? (
                <ActivityIndicator color={colors.background} />
              ) : (
                <Text style={styles.btnPrimaryText}>Search</Text>
              )}
            </Pressable>
          </View>
          <Text style={styles.hintText}>The local seed currently covers common Mercedes/OM651 codes.</Text>
        </View>

        <View style={styles.row}>
          <Pressable style={styles.btnSecondary} onPress={() => void pickImage('library')}>
            <Text style={styles.btnSecondaryText}>Gallery</Text>
          </Pressable>
          <Pressable style={styles.btnSecondary} onPress={() => void pickImage('camera')}>
            <Text style={styles.btnSecondaryText}>Camera</Text>
          </Pressable>
        </View>

        {uri ? (
          <Image source={{ uri }} style={styles.preview} resizeMode="contain" />
        ) : (
          <View style={styles.previewPlaceholder}>
            <Text style={styles.placeholderText}>No image selected</Text>
          </View>
        )}

        <Pressable
          style={[styles.btnPrimary, (!uri || loading) && styles.btnDisabled]}
          onPress={() => void analyze()}
          disabled={!uri || loading}
        >
          {loading ? (
            <ActivityIndicator color={colors.background} />
          ) : (
            <Text style={styles.btnPrimaryText}>Scan image locally</Text>
          )}
        </Pressable>

        {errorMessage ? (
          <View style={styles.bannerError}>
            <Text style={styles.bannerTitle}>Error</Text>
            <Text style={styles.bannerText}>{errorMessage}</Text>
          </View>
        ) : null}

        {result ? (
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.code}>{result.detected_code}</Text>
              <DifficultyBadge value={result.estimated_difficulty} />
            </View>
            <Text style={styles.sectionLabel}>Probable cause</Text>
            <Text style={styles.bodyText}>{result.probable_cause}</Text>
            <Text style={styles.sectionLabel}>Recommended steps</Text>
            {result.step_by_step_fix.map((step, i) => (
              <Text key={i} style={styles.step}>
                {i + 1}. {step}
              </Text>
            ))}
            <Text style={styles.sectionLabel}>Safety</Text>
            <Text style={styles.warningText}>{result.safety_warning}</Text>
          </View>
        ) : null}
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
    paddingHorizontal: space.lg,
    paddingBottom: space.xl,
  },
  title: {
    fontSize: font.title,
    fontWeight: '700',
    color: colors.textPrimary,
    letterSpacing: 0.3,
  },
  subtitle: {
    marginTop: space.xs,
    fontSize: font.caption,
    color: colors.textSecondary,
    marginBottom: space.lg,
  },
  row: {
    flexDirection: 'row',
    gap: space.sm,
    marginBottom: space.md,
  },
  btnSecondary: {
    flex: 1,
    paddingVertical: space.sm,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.surface,
    alignItems: 'center',
  },
  btnSecondaryText: {
    color: colors.textPrimary,
    fontSize: font.body,
    fontWeight: '600',
  },
  preview: {
    width: '100%',
    height: 220,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
    marginBottom: space.md,
  },
  previewPlaceholder: {
    width: '100%',
    height: 220,
    borderRadius: radius.lg,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderStyle: 'dashed',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: space.md,
  },
  placeholderText: {
    color: colors.textSecondary,
    fontSize: font.body,
  },
  btnPrimary: {
    backgroundColor: colors.accent,
    paddingVertical: space.md,
    borderRadius: radius.md,
    alignItems: 'center',
    marginBottom: space.lg,
  },
  btnDisabled: {
    opacity: 0.45,
  },
  btnPrimaryText: {
    color: colors.background,
    fontSize: font.body,
    fontWeight: '700',
  },
  card: {
    backgroundColor: colors.surfaceElevated,
    borderRadius: radius.lg,
    padding: space.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: space.md,
    gap: space.md,
  },
  code: {
    fontSize: 28,
    fontWeight: '800',
    color: colors.accent,
    letterSpacing: 1,
  },
  badge: {
    paddingHorizontal: space.sm,
    paddingVertical: space.xs,
    borderRadius: radius.sm,
    borderWidth: 1,
  },
  badgeText: {
    fontSize: font.caption,
    fontWeight: '700',
  },
  sectionLabel: {
    marginTop: space.md,
    marginBottom: space.xs,
    fontSize: font.caption,
    fontWeight: '700',
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  sectionLabelNoTop: {
    marginBottom: space.xs,
    fontSize: font.caption,
    fontWeight: '700',
    color: colors.textSecondary,
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  bodyText: {
    fontSize: font.body,
    color: colors.textPrimary,
    lineHeight: 22,
  },
  step: {
    fontSize: font.body,
    color: colors.textPrimary,
    lineHeight: 24,
    marginBottom: space.xs,
  },
  warningText: {
    fontSize: font.body,
    color: colors.danger,
    lineHeight: 22,
    fontWeight: '500',
  },
  bannerWarn: {
    backgroundColor: '#3d2e14',
    borderRadius: radius.md,
    padding: space.md,
    marginBottom: space.md,
    borderWidth: 1,
    borderColor: colors.accent,
  },
  bannerError: {
    backgroundColor: '#3d1818',
    borderRadius: radius.md,
    padding: space.md,
    marginBottom: space.md,
    borderWidth: 1,
    borderColor: colors.danger,
  },
  bannerTitle: {
    fontSize: font.caption,
    fontWeight: '700',
    color: colors.textPrimary,
    marginBottom: space.xs,
    textTransform: 'uppercase',
  },
  bannerText: {
    fontSize: font.body,
    color: colors.textPrimary,
    lineHeight: 22,
  },
  lookupCard: {
    backgroundColor: colors.surface,
    borderRadius: radius.lg,
    padding: space.md,
    marginBottom: space.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  lookupRow: {
    flexDirection: 'row',
    gap: space.sm,
    alignItems: 'center',
  },
  input: {
    flex: 1,
    minHeight: 48,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.background,
    color: colors.textPrimary,
    paddingHorizontal: space.md,
    fontSize: font.body,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  btnLookup: {
    minHeight: 48,
    paddingHorizontal: space.lg,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.accent,
  },
  hintText: {
    marginTop: space.sm,
    color: colors.textSecondary,
    fontSize: font.caption,
    lineHeight: 18,
  },
});
