import axios from 'axios';
import type { ApiError, ClassificationResponse, ModelStatus, QualityResponse } from '../types/screening';
import { supabase } from '../lib/supabase';

const api = axios.create({ baseURL: import.meta.env.VITE_API_BASE_URL || '', timeout: 45000 });
api.interceptors.request.use(async config => {
  const { data } = await supabase?.auth.getSession() ?? { data: { session: null } };
  if (data.session?.access_token) config.headers.Authorization = `Bearer ${data.session.access_token}`;
  return config;
});
export async function assessQuality(file: File): Promise<QualityResponse> {
  const form = new FormData(); form.append('file', file);
  const { data } = await api.post<QualityResponse>('/api/v1/quality/assess', form);
  return data;
}
export async function predictClassification(file: File, patientRef?: string): Promise<ClassificationResponse> {
  const form = new FormData(); form.append('file', file); if (patientRef) form.append('patient_ref', patientRef);
  const { data } = await api.post<ClassificationResponse>('/api/v1/classification/predict', form); return data;
}
export async function getModelStatus(): Promise<ModelStatus> { const { data } = await api.get<ModelStatus>('/api/v1/model/status'); return data; }
export async function checkHealth(): Promise<boolean> { try { await api.get('/health'); return true; } catch { return false; } }
export function toApiError(error: unknown): ApiError {
  if (axios.isAxiosError(error)) {
    const status = error.response?.status ?? null;
    const messages: Record<number, string> = {
      400: 'The image or patient reference is invalid.',
      401: 'Your session has expired. Sign in again.',
      413: 'The image is larger than the 20 MB limit.',
      415: 'Only JPEG and PNG images are supported.',
      502: 'The classification result could not be saved. Try again later.',
      503: 'The classification model or database is unavailable.',
    };
    return {
      status,
      title: status ? `Request error ${status}` : 'Backend unreachable',
      message: status && messages[status] ? messages[status] : typeof error.response?.data?.detail === 'string' ? error.response.data.detail : error.message,
    };
  }
  return { status: null, title: 'Application error', message: error instanceof Error ? error.message : 'Unexpected error' };
}
