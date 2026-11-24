import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import type { Session, User } from '@supabase/supabase-js';
import { supabase } from '../lib/supabase';

export type Role = 'admin' | 'researcher' | 'reviewer';
export type Profile = { id: string; full_name: string; role: Role; is_active: boolean; created_at: string };
type AuthValue = { session: Session | null; user: User | null; profile: Profile | null; loading: boolean; configured: boolean; refreshProfile: () => Promise<void>; signOut: () => Promise<void> };
const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const getProfile = async (userId?: string) => {
    if (!supabase || !userId) { setProfile(null); return; }
    const { data, error } = await supabase.from('profiles').select('id,full_name,role,is_active,created_at').eq('id', userId).maybeSingle();
    setProfile(error ? null : data as Profile | null);
  };
  useEffect(() => {
    if (!supabase) { setLoading(false); return; }
    let alive = true;
    supabase.auth.getSession().then(async ({ data }) => {
      if (!alive) return;
      setSession(data.session);
      await getProfile(data.session?.user.id);
      if (alive) setLoading(false);
    }).catch(() => { if (alive) { setSession(null); setProfile(null); setLoading(false); } });
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next);
      // Defer the query to avoid calling Supabase while its auth lock is held.
      queueMicrotask(() => { if (alive) void getProfile(next?.user.id); });
    });
    return () => { alive = false; subscription.unsubscribe(); };
  }, []);
  useEffect(() => {
    const userId = session?.user.id;
    if (!supabase || !userId) return;
    const refresh = () => { void getProfile(userId); };
    const timer = window.setInterval(refresh, 30_000);
    window.addEventListener('focus', refresh);
    return () => { window.clearInterval(timer); window.removeEventListener('focus', refresh); };
  }, [session?.user.id]);
  const value: AuthValue = { session, user: session?.user ?? null, profile, loading, configured: !!supabase, refreshProfile: () => getProfile(session?.user.id), signOut: async () => { if (supabase) await supabase.auth.signOut(); setProfile(null); } };
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
export function useAuth() { const value = useContext(AuthContext); if (!value) throw new Error('useAuth must be used within AuthProvider'); return value; }
