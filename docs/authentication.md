# RetinaSaarthi authentication setup

The frontend uses React Router, Supabase Auth, a profile row created by a database trigger, and database-backed role checks. The FastAPI quality endpoint validates the caller's current Supabase token and active profile before accepting uploads. Role changes and account status changes run through the `admin-users` Supabase Edge Function and an atomic, audited SQL function.

## Local setup

1. Create a Supabase project and copy its project URL and publishable/anon key into `frontend/.env.local` using `frontend/.env.example` as the template. The publishable/anon key is designed for browser use; never put a service-role key in a `VITE_` variable or frontend source.
2. Apply `supabase/migrations/202611110001_auth_profiles.sql` from the repository root with the Supabase CLI (`supabase link`, then `supabase db push`) or run it in the SQL editor.
3. Configure Auth → URL Configuration. Set the Site URL to the deployed app origin and add redirect URLs for `http://localhost:5173/auth/callback`, `http://localhost:5173/reset-password`, and the corresponding production URLs.
4. Enable email confirmation under Auth → Providers → Email. Configure SMTP for production delivery, set a reasonable confirmation/recovery expiry, and enable leaked-password protection where available.
5. For local FastAPI, copy `.env.backend.example` to the backend environment (or export those values) and start the API. The API needs only the project URL and publishable/anon key; it validates the user token through Supabase Auth and reads the caller's profile under RLS.
6. Start the frontend from `frontend/` with `npm run dev`.

## First administrator

Registration can only create researcher profiles. To bootstrap the first administrator, verify the person in Supabase Auth, copy their exact Auth user UUID, and run this statement in the trusted Supabase SQL editor:

```sql
update public.profiles set role = 'admin' where id = '<verified-auth-user-uuid>';
```

Do this only from the trusted SQL editor or another trusted operator environment. Browser users cannot update role or active status. Do not use a user-editable `user_metadata` field for authorization.

## Google OAuth

Google sign-in stays disabled until configured. In Google Cloud, create an OAuth client, add the Supabase project's Auth callback URL (shown under Supabase Auth → Providers → Google) as an authorized redirect URI, then enable Google under Supabase Auth → Providers and enter the Google client ID and secret there. Add the application callback URLs listed above to Supabase Auth → URL Configuration. Set `VITE_ENABLE_GOOGLE_OAUTH=true` in the frontend deployment only after the provider is enabled and redirects have been tested. Test with an allowed account. The application calls Supabase `signInWithOAuth` and returns through `/auth/callback`; it does not handle Google tokens itself.

## Edge Function deployment

Deploy the administrator function after applying the migration:

```sh
supabase functions deploy admin-users
```

The function uses Supabase's server-provided `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and `SUPABASE_SERVICE_ROLE_KEY` secrets. Do not add the service key to frontend environment files. Keep JWT verification enabled for this function. It independently validates the bearer token, checks the caller's active database role, rejects self privilege changes, validates requested values, and calls the service-role-only `admin_update_profile` transaction. That transaction rechecks the caller role and writes its audit row atomically with the profile update.

## Authorization model and deployment notes

- Profiles use RLS: users can read their own row, active administrators can read all rows, and users can update only `full_name`. Column grants and a trigger protect `role` and `is_active`.
- New profiles are created by an `auth.users` trigger that always writes `researcher` and `true`; it ignores any submitted role/status metadata.
- FastAPI quality upload accepts active `admin` or `researcher` accounts. It rejects missing/expired sessions, unverified email, inactive users, and reviewers. The token is validated remotely, so set the API's `SUPABASE_URL` and `SUPABASE_ANON_KEY` and use HTTPS in deployment.
- Configure `FRONTEND_ORIGINS` to an exact comma-separated list of trusted origins. Do not use `*` in production. Terminate TLS at the API gateway/load balancer and apply request rate limits there.
- Successful quality assessments save only the result, decision, metrics, a generic image label, and owner ID in `assessments`; the image and user-supplied filename are not stored. RLS lets active researchers read their own records, active reviewers read results, and active admins read all records. The dashboard's activity chart remains explicitly sample data; the saved list is the actual account-scoped history.
- The reviewer currently has access to the research dashboard, research performance, About, and the general dashboard. Research dataset content is currently bundled/static public research information, not private user data.
- Admin management lists account emails via the trusted Edge Function because the browser must not query `auth.users` directly. All changes are recorded in `admin_audit_log`, which has no client access policy.
- Frontend route guards and hidden navigation are usability controls. Database RLS, the Edge Function, and the FastAPI token/role check are the enforcement points.
- Passwords require at least 12 characters with upper/lowercase and a number in the UI; configure compatible Supabase Auth password rules. Email verification is enforced at login and before quality uploads.

## Verification checklist

There is no configured Supabase project or OAuth client in this repository, so the following end-to-end cases require a configured local/deployed Supabase instance:

1. Register and confirm email; inspect `profiles` and verify role is `researcher` even when signup metadata contains a forged role.
2. Sign in with correct and incorrect passwords; verify unconfirmed and inactive accounts cannot proceed.
3. Request a password reset, follow the recovery link, update the password, and verify expired links are rejected.
4. Configure Google as above and test callback handling; without provider configuration, Google sign-in is expected to return a Supabase provider error.
5. Refresh an authenticated page, sign out, and confirm protected pages redirect to `/login`.
6. Test reviewer and researcher route access, including direct URL entry to `/quality` and `/admin/users`.
7. As a user, attempt REST updates of `role` and `is_active`; PostgreSQL column grants/RLS must reject them. Verify own profile read and own-name-only update.
8. As a non-admin, call `admin-users` directly; it must return 403. As an admin, change another user's role/status and inspect the audit log. Self role change/deactivation must fail.
9. Insert/read assessment rows using two researcher sessions; each researcher must see only their own rows. Confirm active reviewers/admins can read permitted results and inactive accounts cannot. Users must not update or delete assessment rows.
10. Call FastAPI without a token, with an expired token, with an inactive account, and with reviewer credentials; uploads must be rejected. Active researcher/admin uploads should retain the existing image validation behavior and save an account-owned history record.

Local code checks: from `frontend/`, run `npm run build`. The API's existing suite can be run from repository root with the project's configured Python environment using `pytest`.
