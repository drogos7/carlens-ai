import { Platform } from 'react-native';

/** Set in `app/.env` as EXPO_PUBLIC_API_BASE_URL (e.g. http://10.0.2.2:8000 for Android emulator). */
const base = process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://127.0.0.1:8000';

export type DecodeSegment = {
  label: string;
  positions?: string;
  position?: string;
  value: string;
  meaning: string;
};

export type VinDecodeResponse = {
  vin: string;
  wmi: string;
  vds: string;
  vis: string;
  model_year: number | null;
  check_digit: string;
  check_digit_valid: boolean | null;
  summary: string;
  segments: DecodeSegment[];
};

export type DtcDecodeResponse = {
  code: string;
  is_standard_format: boolean;
  summary: string;
  segments: DecodeSegment[];
};

export type AnalyzeResponse = {
  scan_type?: 'dtc' | 'vin';
  detected_code: string;
  detected_vin?: string | null;
  vehicle_make?: string | null;
  vehicle_model?: string | null;
  vehicle_engine?: string | null;
  vehicle_year?: number | null;
  vin_decode?: VinDecodeResponse | null;
  dtc_decode?: DtcDecodeResponse | null;
  probable_cause: string;
  step_by_step_fix: string[];
  estimated_difficulty: 'Easy' | 'Medium' | 'Hard';
  safety_warning: string;
};

export type KnowledgeCodeResponse = {
  code: string;
  make: string;
  engine: string;
  title: string;
  description: string;
  probable_causes: string[];
  symptoms: string[];
  step_by_step_fix: string[];
  difficulty: 'Easy' | 'Medium' | 'Hard';
  safety_warning: string;
  sources: string[];
};

export type ApiErrorBody = {
  detail?: string | Record<string, unknown>;
};

function guessFilename(uri: string): { name: string; type: string } {
  const lower = uri.toLowerCase();
  if (lower.endsWith('.png') || lower.includes('image/png')) {
    return { name: 'capture.png', type: 'image/png' };
  }
  return { name: 'capture.jpg', type: 'image/jpeg' };
}

async function buildFormData(uri: string): Promise<FormData> {
  const { name, type } = guessFilename(uri);
  const formData = new FormData();

  if (Platform.OS === 'web') {
    const res = await fetch(uri);
    const blob = await res.blob();
    const file = new File([blob], name, { type: blob.type || type });
    formData.append('image', file);
  } else {
    formData.append('image', {
      uri,
      name,
      type,
    } as unknown as Blob);
  }

  return formData;
}

const SCAN_TIMEOUT_MS = 90_000;

export async function analyzeErrorImage(uri: string): Promise<AnalyzeResponse> {
  const formData = await buildFormData(uri);
  const url = `${base.replace(/\/$/, '')}/scan-error-local`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), SCAN_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      body: formData,
      signal: controller.signal,
    });
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      throw new Error('Scan timed out. Try a closer crop of the code or VIN line, then scan again.');
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }

  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }

  if (!res.ok) {
    let message = `Server error (${res.status})`;
    if (res.status === 422) {
      message = 'Could not read an error code from the image.';
    } else if ((res.status === 503 || res.status === 404) && json && typeof json === 'object' && 'detail' in json) {
      const detail = (json as { detail: unknown }).detail;
      if (typeof detail === 'string') {
        message =
          detail.includes('OPENAI_API_KEY') || detail.includes('ANTHROPIC_API_KEY')
            ? 'The API key is missing on the server. Open backend\\.env, add OPENAI_API_KEY=sk-..., and restart uvicorn.'
            : detail;
      } else if (typeof detail === 'object' && detail !== null && 'message' in detail) {
        const maybeMessage = (detail as { message?: unknown }).message;
        if (typeof maybeMessage === 'string') {
          message = maybeMessage;
        }
      }
    }
    const err: Error & { status?: number; body?: unknown } = new Error(message);
    err.status = res.status;
    err.body = json ?? text;
    throw err;
  }

  return json as AnalyzeResponse;
}

export async function lookupDiagnosticCode(code: string): Promise<AnalyzeResponse> {
  const normalized = code.trim().toUpperCase();
  const url = `${base.replace(/\/$/, '')}/lookup?q=${encodeURIComponent(normalized)}&make=Mercedes-Benz`;
  const res = await fetch(url);
  const text = await res.text();
  let json: unknown = null;
  try {
    json = text ? JSON.parse(text) : null;
  } catch {
    json = null;
  }

  if (!res.ok) {
    let message =
      res.status === 404
        ? `No information found yet for ${normalized}.`
        : `Server error (${res.status})`;
    if (json && typeof json === 'object' && 'detail' in json) {
      const detail = (json as { detail: unknown }).detail;
      if (typeof detail === 'string') {
        message = detail;
      } else if (typeof detail === 'object' && detail !== null && 'message' in detail) {
        const maybeMessage = (detail as { message?: unknown }).message;
        if (typeof maybeMessage === 'string') {
          message = maybeMessage;
        }
      }
    }
    const err: Error & { status?: number; body?: unknown } = new Error(message);
    err.status = res.status;
    err.body = json ?? text;
    throw err;
  }

  return json as AnalyzeResponse;
}

export function getApiBaseUrl(): string {
  return base.replace(/\/$/, '');
}

export type HealthResponse = {
  status: string;
  vision_configured?: boolean;
  llm_provider?: string;
  message?: string;
};

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${base.replace(/\/$/, '')}/health`);
  if (!res.ok) {
    throw new Error('Backend unavailable. Start uvicorn in the backend folder.');
  }
  return res.json() as Promise<HealthResponse>;
}
