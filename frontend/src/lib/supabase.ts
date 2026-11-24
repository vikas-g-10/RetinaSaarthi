import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

function browserStorage() {
  return {
    getItem: (key: string) => {
      try { return localStorage.getItem('retinasaarthi-remember') === 'false' ? sessionStorage.getItem(key) : localStorage.getItem(key); } catch { return null; }
    },
    setItem: (key: string, value: string) => {
      try { (localStorage.getItem('retinasaarthi-remember') === 'false' ? sessionStorage : localStorage).setItem(key, value); } catch { /* Storage may be unavailable in restricted browser contexts. */ }
    },
    removeItem: (key: string) => {
      try { localStorage.removeItem(key); sessionStorage.removeItem(key); } catch { /* Storage may be unavailable in restricted browser contexts. */ }
    },
  };
}

export const supabase = url && anonKey
  ? createClient(url, anonKey, { auth: { autoRefreshToken: true, persistSession: true, detectSessionInUrl: true, storage: browserStorage() } })
  : null;

export function setRememberSession(remember: boolean) {
  try { localStorage.setItem('retinasaarthi-remember', String(remember)); } catch { /* Supabase will still manage the current in-memory session. */ }
}

export function requireSupabase() {
  if (!supabase) throw new Error('Authentication is not configured. Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.');
  return supabase;
}
