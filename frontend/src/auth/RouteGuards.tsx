import { Navigate, Outlet, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';
import { useAuth, type Role } from './AuthProvider';

export function ProtectedRoute({ children }: { children?: ReactNode }) {
  const { user, profile, loading, configured } = useAuth(); const location = useLocation();
  if (loading) return <div className="auth-loading" role="status">Restoring your secure session…</div>;
  if (!configured) return <div className="auth-loading"><strong>Supabase is not configured</strong><span>Add the VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY values to frontend/.env.local.</span></div>;
  if (!user) return <Navigate to="/login" replace state={{ from: location }} />;
  if (!user.email_confirmed_at) return <Navigate to="/login" replace state={{ message: 'Verify your email before signing in.' }} />;
  if (!profile || !profile.is_active) return <Navigate to="/unauthorized" replace />;
  return children ?? <Outlet />;
}
export function RoleProtectedRoute({ roles, children }: { roles: Role[]; children?: ReactNode }) {
  const { profile, loading } = useAuth();
  if (loading) return <div className="auth-loading" role="status">Checking access…</div>;
  if (!profile || !roles.includes(profile.role)) return <Navigate to="/unauthorized" replace />;
  return children ?? <Outlet />;
}
