
import {
  forwardRef,
  useEffect,
  useState,
  type InputHTMLAttributes,
  type ReactNode,
} from 'react';
import { Link, Navigate, useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Eye,
  EyeOff,
  ShieldCheck,
  Activity,
  Mail,
  ArrowLeft,
  LoaderCircle,
  CheckCircle2,
} from 'lucide-react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { requireSupabase, setRememberSession } from '../lib/supabase';
import { useAuth } from './AuthProvider';

const googleConfigured =
  import.meta.env.VITE_ENABLE_GOOGLE_OAUTH === 'true';

const passwordRule = z
  .string()
  .min(12, 'Use at least 12 characters.')
  .max(128)
  .regex(/[a-z]/, 'Add a lowercase letter.')
  .regex(/[A-Z]/, 'Add an uppercase letter.')
  .regex(/[0-9]/, 'Add a number.');

const loginSchema = z.object({
  email: z.string().email('Enter a valid email address.'),
  password: z.string().min(1, 'Enter your password.'),
});

const signupSchema = z
  .object({
    fullName: z.string().trim().min(2, 'Enter your full name.').max(120),
    email: z.string().email('Enter a valid email address.'),
    password: passwordRule,
    confirm: z.string(),
  })
  .refine((v) => v.password === v.confirm, {
    path: ['confirm'],
    message: 'Passwords do not match.',
  });

type LoginData = z.infer<typeof loginSchema>;
type SignupData = z.infer<typeof signupSchema>;

function AuthFrame({
  eyebrow,
  title,
  description,
  children,
  footer,
}: {
  eyebrow: string;
  title: string;
  description: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <main className="auth-page">
      <div className="auth-glow auth-glow-one" />
      <div className="auth-glow auth-glow-two" />

      <motion.section
        className="auth-card"
        initial={{ opacity: 0, y: 18, scale: 0.985 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.45, ease: 'easeOut' }}
      >
        <Link
          className="auth-brand"
          to="/login"
          aria-label="RetinaSaarthi home"
        >
          <span className="auth-mark">
            <Activity size={20} />
          </span>
          <span>
            Retina<span className="brand-accent">Saarthi</span>
          </span>
        </Link>

        <div className="auth-heading">
          <p>{eyebrow}</p>
          <h1>{title}</h1>
          <span>{description}</span>
        </div>

        {children}

        {footer && <div className="auth-footer">{footer}</div>}
      </motion.section>

      <p className="auth-trust">
        <ShieldCheck size={14} />
        Secure access to the RetinaSaarthi research workspace
      </p>
    </main>
  );
}

/* Ref-forwarding field: required by React Hook Form. */
type FieldProps = Omit<InputHTMLAttributes<HTMLInputElement>, 'id'> & {
  label: string;
  id: string;
  error?: string;
};

const Field = forwardRef<HTMLInputElement, FieldProps>(
  ({ label, id, type = 'text', error, ...props }, ref) => (
    <div className="auth-field">
      <label htmlFor={id}>{label}</label>
      <input
        {...props}
        ref={ref}
        id={id}
        type={type}
        aria-invalid={!!error}
        aria-describedby={error ? `${id}-error` : undefined}
      />
      {error && (
        <span className="field-error" id={`${id}-error`}>
          {error}
        </span>
      )}
    </div>
  )
);

Field.displayName = 'Field';

function Notice({
  children,
  success = false,
}: {
  children: ReactNode;
  success?: boolean;
}) {
  return (
    <div
      className={success ? 'auth-notice success' : 'auth-notice'}
      role="alert"
    >
      {children}
    </div>
  );
}

function roleHome(_role?: string | null) {
  return '/';
}

function GoogleMark() {
  return (
    <svg viewBox="0 0 48 48" aria-hidden="true">
      <path
        fill="#EA4335"
        d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
      />
      <path
        fill="#4285F4"
        d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.76 7.18l7.73 6C44.42 37.95 46.98 31.8 46.98 24.55z"
      />
      <path
        fill="#FBBC05"
        d="M10.53 28.59A14.4 14.4 0 0 1 9.75 24c0-1.59.27-3.13.75-4.59l-7.98-6.19A23.9 23.9 0 0 0 0 24c0 3.88.93 7.56 2.56 10.78l7.97-6.19z"
      />
      <path
        fill="#34A853"
        d="M24 48c6.48 0 11.93-2.13 15.9-5.8l-7.73-6c-2.14 1.45-4.89 2.3-8.17 2.3-6.26 0-11.57-4.22-13.46-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"
      />
    </svg>
  );
}

export function LoginPage() {
  const { user, profile, loading, configured, refreshProfile } = useAuth();
  const nav = useNavigate();
  const location = useLocation();

  const [error, setError] = useState('');
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);

  const [remember, setRemember] = useState(() => {
    try {
      return localStorage.getItem('retinasaarthi-remember') !== 'false';
    } catch {
      return true;
    }
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginData>({
    resolver: zodResolver(loginSchema),
    defaultValues: { email: '', password: '' },
  });

  if (loading) {
    return (
      <div className="auth-loading" role="status">
        Restoring your secure session…
      </div>
    );
  }

  if (!loading && user && profile?.is_active) {
    return <Navigate to={roleHome(profile.role)} replace />;
  }

  const onSubmit = handleSubmit(async ({ email, password }) => {
    setBusy(true);
    setError('');
    setRememberSession(remember);

    try {
      const { data, error: authError } =
        await requireSupabase().auth.signInWithPassword({
          email,
          password,
        });

      if (authError) throw authError;

      if (!data.user?.email_confirmed_at) {
        await requireSupabase().auth.signOut();
        throw new Error('Verify your email address before signing in.');
      }

      const { data: p, error: profileError } = await requireSupabase()
        .from('profiles')
        .select('role,is_active')
        .eq('id', data.user.id)
        .single();

      if (profileError || !p?.is_active) {
        await requireSupabase().auth.signOut();
        throw new Error(
          'This account is inactive or its profile is unavailable. Contact an administrator.'
        );
      }

      await refreshProfile();

      const destination = (
        location.state as { from?: { pathname?: string } } | null
      )?.from?.pathname;

      nav(destination || roleHome(p.role), { replace: true });
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'Unable to sign in. Please try again.'
      );
    } finally {
      setBusy(false);
    }
  });

  const google = async () => {
    setBusy(true);
    setError('');
    setRememberSession(remember);

    try {
      const { error: e } =
        await requireSupabase().auth.signInWithOAuth({
          provider: 'google',
          options: {
            redirectTo: `${window.location.origin}/auth/callback`,
          },
        });

      if (e) throw e;
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'Google sign-in is unavailable.'
      );
      setBusy(false);
    }
  };

  return (
    <AuthFrame
      eyebrow="Research workspace"
      title="Welcome back"
      description="Sign in to continue to RetinaSaarthi"
      footer={
        <>
          New to RetinaSaarthi? <Link to="/signup">Create an account</Link>
        </>
      }
    >
      {!configured && (
        <Notice>
          Supabase configuration is missing. Set the VITE_SUPABASE_URL and
          VITE_SUPABASE_ANON_KEY values to enable sign-in.
        </Notice>
      )}

      {(location.state as { message?: string } | null)?.message && (
        <Notice>
          {(location.state as { message: string }).message}
        </Notice>
      )}

      {error && <Notice>{error}</Notice>}

      <form className="auth-form" onSubmit={onSubmit} noValidate>
        <Field
          id="email"
          label="Email address"
          type="email"
          autoComplete="email"
          placeholder="you@institution.org"
          error={errors.email?.message}
          {...register('email')}
        />

        <div className="password-wrap">
          <Field
            id="password"
            label="Password"
            type={show ? 'text' : 'password'}
            autoComplete="current-password"
            placeholder="Enter your password"
            error={errors.password?.message}
            {...register('password')}
          />

          <button
            className="password-toggle"
            type="button"
            onClick={() => setShow(!show)}
            aria-label={show ? 'Hide password' : 'Show password'}
          >
            {show ? <EyeOff size={17} /> : <Eye size={17} />}
          </button>
        </div>

        <div className="auth-options">
          <label className="check-label">
            <input
              type="checkbox"
              checked={remember}
              onChange={(e) => setRemember(e.target.checked)}
            />
            <span>Remember me</span>
          </label>
          <Link to="/forgot-password">Forgot password?</Link>
        </div>

        <button
          className="auth-submit"
          disabled={busy || !configured}
        >
          {busy ? (
            <>
              <LoaderCircle className="spin" size={17} />
              Signing in…
            </>
          ) : (
            <>
              Sign in <span>→</span>
            </>
          )}
        </button>
      </form>

      <div className="auth-divider">
        <span>or continue with</span>
      </div>

      <button
        className="google-button"
        type="button"
        onClick={google}
        disabled={busy || !configured || !googleConfigured}
        title={
          !googleConfigured
            ? 'Configure Google in Supabase and set VITE_ENABLE_GOOGLE_OAUTH=true'
            : undefined
        }
      >
        <GoogleMark /> Continue with Google
      </button>

      {!googleConfigured && (
        <p className="oauth-config-note">
          Google sign-in is available after its provider and redirect URLs are
          configured in Supabase.
        </p>
      )}
    </AuthFrame>
  );
}

export function SignupPage() {
  const [error, setError] = useState('');
  const [sent, setSent] = useState(false);
  const [busy, setBusy] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<SignupData>({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      fullName: '',
      email: '',
      password: '',
      confirm: '',
    },
  });

  const submit = handleSubmit(
    async ({ fullName, email, password }) => {
      setBusy(true);
      setError('');

      try {
        const { error: e } =
          await requireSupabase().auth.signUp({
            email,
            password,
            options: {
              data: { full_name: fullName },
              emailRedirectTo: `${window.location.origin}/auth/callback`,
            },
          });

        if (e) throw e;

        setSent(true);
      } catch (e) {
        setError(
          e instanceof Error
            ? e.message
            : 'Unable to create your account.'
        );
      } finally {
        setBusy(false);
      }
    }
  );

  return (
    <AuthFrame
      eyebrow="Join the research workspace"
      title={sent ? 'Check your inbox' : 'Create your account'}
      description={
        sent
          ? 'We sent a verification link to your email address.'
          : 'Register to access RetinaSaarthi research tools.'
      }
      footer={
        <>
          Already registered? <Link to="/login">Sign in</Link>
        </>
      }
    >
      {sent ? (
        <div className="success-panel">
          <CheckCircle2 size={28} />
          <p>
            Verify your email to activate your account. A researcher profile
            is created automatically, and protected access remains blocked
            until verification.
          </p>
          <Link to="/login">Return to sign in</Link>
        </div>
      ) : (
        <form
          className="auth-form"
          onSubmit={submit}
          noValidate
        >
          {error && <Notice>{error}</Notice>}

          <Field
            id="fullName"
            label="Full name"
            autoComplete="name"
            placeholder="Your name"
            error={errors.fullName?.message}
            {...register('fullName')}
          />

          <Field
            id="signupEmail"
            label="Email address"
            type="email"
            autoComplete="email"
            placeholder="you@institution.org"
            error={errors.email?.message}
            {...register('email')}
          />

          <Field
            id="newPassword"
            label="Password"
            type="password"
            autoComplete="new-password"
            placeholder="At least 12 characters"
            error={errors.password?.message}
            {...register('password')}
          />

          <Field
            id="confirm"
            label="Confirm password"
            type="password"
            autoComplete="new-password"
            placeholder="Re-enter password"
            error={errors.confirm?.message}
            {...register('confirm')}
          />

          <p className="password-hint">
            Use 12+ characters with uppercase, lowercase, and a number.
          </p>

          <button className="auth-submit" disabled={busy}>
            {busy ? (
              <>
                <LoaderCircle className="spin" size={17} />
                Creating account…
              </>
            ) : (
              <>
                Create account <span>→</span>
              </>
            )}
          </button>
        </form>
      )}
    </AuthFrame>
  );
}

export function ForgotPasswordPage() {
  const [sent, setSent] = useState(false);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const schema = z.object({
    email: z.string().email('Enter a valid email address.'),
  });

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<{ email: string }>({
    resolver: zodResolver(schema),
    defaultValues: { email: '' },
  });

  const submit = handleSubmit(async ({ email }) => {
    setBusy(true);
    setError('');

    try {
      const { error: e } =
        await requireSupabase().auth.resetPasswordForEmail(email, {
          redirectTo: `${window.location.origin}/reset-password`,
        });

      if (e) throw e;

      setSent(true);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'Unable to send the reset email.'
      );
    } finally {
      setBusy(false);
    }
  });

  return (
    <AuthFrame
      eyebrow="Account recovery"
      title={sent ? 'Check your inbox' : 'Reset your password'}
      description={
        sent
          ? 'If an account exists for this address, a recovery link is on its way.'
          : 'Enter your account email and we’ll send you a secure reset link.'
      }
      footer={
        <Link to="/login">
          <ArrowLeft size={14} /> Back to sign in
        </Link>
      }
    >
      {error && <Notice>{error}</Notice>}

      {sent ? (
        <div className="success-panel">
          <Mail size={26} />
          <p>
            Use the link in the email to choose a new password. The link
            expires according to your Supabase Auth settings.
          </p>
        </div>
      ) : (
        <form className="auth-form" onSubmit={submit}>
          <Field
            id="resetEmail"
            label="Email address"
            type="email"
            autoComplete="email"
            placeholder="you@institution.org"
            error={errors.email?.message}
            {...register('email')}
          />

          <button className="auth-submit" disabled={busy}>
            {busy ? 'Sending…' : <>Send recovery link <span>→</span></>}
          </button>
        </form>
      )}
    </AuthFrame>
  );
}

export function ResetPasswordPage() {
  const [error, setError] = useState('');
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);
  const nav = useNavigate();

  const schema = z
    .object({
      password: passwordRule,
      confirm: z.string(),
    })
    .refine((v) => v.password === v.confirm, {
      path: ['confirm'],
      message: 'Passwords do not match.',
    });

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<{ password: string; confirm: string }>({
    resolver: zodResolver(schema),
    defaultValues: { password: '', confirm: '' },
  });

  const submit = handleSubmit(async ({ password }) => {
    setBusy(true);
    setError('');

    try {
      const { error: e } =
        await requireSupabase().auth.updateUser({ password });

      if (e) throw e;

      setDone(true);
      await requireSupabase().auth.signOut();
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : 'The recovery link is invalid or expired. Request a new one.'
      );
    } finally {
      setBusy(false);
    }
  });

  return (
    <AuthFrame
      eyebrow="Secure account recovery"
      title={done ? 'Password updated' : 'Choose a new password'}
      description={
        done
          ? 'Your password has been changed successfully.'
          : 'Create a new password for your account.'
      }
      footer={<Link to="/forgot-password">Request another link</Link>}
    >
      {error && <Notice>{error}</Notice>}

      {done ? (
        <div className="success-panel">
          <CheckCircle2 size={28} />
          <Link to="/login" onClick={() => nav('/login')}>
            Continue to sign in
          </Link>
        </div>
      ) : (
        <form className="auth-form" onSubmit={submit}>
          <Field
            id="resetNewPassword"
            label="New password"
            type="password"
            autoComplete="new-password"
            placeholder="At least 12 characters"
            error={errors.password?.message}
            {...register('password')}
          />

          <Field
            id="resetConfirm"
            label="Confirm new password"
            type="password"
            autoComplete="new-password"
            placeholder="Re-enter password"
            error={errors.confirm?.message}
            {...register('confirm')}
          />

          <button className="auth-submit" disabled={busy}>
            {busy ? 'Updating…' : <>Update password <span>→</span></>}
          </button>
        </form>
      )}
    </AuthFrame>
  );
}

export function OAuthCallbackPage() {
  const { user, profile, loading, refreshProfile } = useAuth();
  const location = useLocation();
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const refreshTimer = window.setTimeout(() => {
      void refreshProfile().finally(() => setAttempt(1));
    }, 1200);

    const waitTimer = window.setTimeout(() => setAttempt(2), 10000);

    return () => {
      window.clearTimeout(refreshTimer);
      window.clearTimeout(waitTimer);
    };
  }, []);

  const params = new URLSearchParams(
    location.search || location.hash.slice(1)
  );

  const callbackError =
    params.get('error_description') || params.get('error');

  if (callbackError) {
    return (
      <Navigate
        to="/login"
        replace
        state={{
          message:
            'Sign-in could not be completed. Check your provider configuration or try again.',
        }}
      />
    );
  }

  if (!loading && user && profile?.is_active) {
    return <Navigate to={roleHome(profile.role)} replace />;
  }

  return (
    <div className="auth-loading">
      Completing secure sign-in…
      {attempt > 1 && (
        <>
          <span>
            Check that your account is active and verified, then try signing
            in.
          </span>
          <Link to="/login">Return to sign in</Link>
        </>
      )}
    </div>
  );
}

export function UnauthorizedPage() {
  const { user, profile, signOut } = useAuth();

  return (
    <AuthFrame
      eyebrow="Access restricted"
      title="You don’t have access"
      description="Your account is authenticated, but this page is not available to your role."
      footer={<Link to="/">Return to dashboard</Link>}
    >
      <div className="success-panel">
        <ShieldCheck size={28} />
        <p>
          {profile
            ? `Signed in as ${profile.role}. Ask an administrator if you need different access.`
            : 'This account is inactive or its profile is not available.'}
        </p>

        {user && (
          <button className="text-button" onClick={() => void signOut()}>
            Sign out
          </button>
        )}
      </div>
    </AuthFrame>
  );
}
