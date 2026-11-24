import { Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider, useAuth } from './auth/AuthProvider';
import { ProtectedRoute, RoleProtectedRoute } from './auth/RouteGuards';
import { ForgotPasswordPage, LoginPage, OAuthCallbackPage, ResetPasswordPage, SignupPage, UnauthorizedPage } from './auth/AuthPages';
import { AdminUsersPage } from './admin/AdminUsersPage';
import { DashboardPage } from './pages/DashboardPage';
import { QualityPage } from './pages/QualityPage';
import { ClassificationPage } from './pages/ClassificationPage';
import { HistoryPage } from './pages/HistoryPage';
import { ReportsPage } from './pages/ReportsPage';
import { ResultPage } from './pages/ResultPage';
import { ProfilePage } from './pages/ProfilePage';
import { ResearchPage } from './pages/ResearchPage';
import { AboutPage } from './pages/AboutPage';
import { ExplainabilityPage } from './pages/ExplainabilityPage';
import { AppShell } from './components/DesignSystem';

function RoutedApp() { const { loading, user } = useAuth(); return <Routes>
  <Route path="/login" element={<LoginPage />} /><Route path="/signup" element={<SignupPage />} /><Route path="/forgot-password" element={<ForgotPasswordPage />} /><Route path="/reset-password" element={<ResetPasswordPage />} /><Route path="/auth/callback" element={<OAuthCallbackPage />} /><Route path="/unauthorized" element={<UnauthorizedPage />} />
  <Route element={<ProtectedRoute />}>
    <Route path="/" element={<DashboardPage />} /><Route path="/history" element={<HistoryPage />} /><Route path="/reports" element={<ReportsPage />} /><Route path="/results/:id" element={<ResultPage />} /><Route path="/profile" element={<ProfilePage />} /><Route path="/about" element={<AboutPage />} />
    <Route path="/quality" element={<QualityPage />} />
    <Route path="/classification" element={<ClassificationPage />} />
    <Route path="/research" element={<ResearchPage />} />
    <Route path="/explainability" element={<ExplainabilityPage />} />
    <Route element={<RoleProtectedRoute roles={['admin']} />}><Route path="/admin/users" element={<AppShell><AdminUsersPage /></AppShell>} /></Route>
  </Route>
  <Route path="*" element={<Navigate to={loading || user ? '/' : '/login'} replace />} />
</Routes>; }
export default function App() { return <AuthProvider><RoutedApp /></AuthProvider>; }
