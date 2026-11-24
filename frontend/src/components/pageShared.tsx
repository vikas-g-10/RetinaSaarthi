import { Card, Skeleton } from './DesignSystem';
import { supabase } from '../lib/supabase';

export const grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'];
export const colors = ['#2f8f67', '#227f9b', '#b9791d', '#c85b3e', '#ad3e67'];
export const readable = (value: string) => value.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
export const monthKey = (date: Date) => date.toLocaleString(undefined, { month: 'short' });
export type ResultRow = {
  id: string;
  patient_ref: string | null;
  predicted_label: string;
  probabilities: number[];
  review_status: string;
  model_version: string;
  created_at: string;
  assessments?: { filename: string; decision: string; created_at: string; metrics: unknown } | null;
};
export async function loadResults() {
  if (!supabase) throw new Error('Supabase is not configured.');
  const response = await supabase.from('classification_results')
    .select('id,patient_ref,predicted_label,probabilities,review_status,model_version,created_at,assessments(filename,decision,created_at,metrics)')
    .order('created_at', { ascending: false });
  if (response.error) {
    if (/relation|does not exist|schema cache/i.test(response.error.message)) throw new Error('The classification database migration has not been applied.');
    throw new Error(response.error.message);
  }
  return (response.data ?? []) as unknown as ResultRow[];
}
export function LoadingCards() {
  return <div className="stats-grid-new">{[1, 2, 3, 4].map(x => <Card className="stat-card" key={x}><Skeleton className="skeleton-icon" /><Skeleton /><Skeleton className="skeleton-number" /></Card>)}</div>;
}
