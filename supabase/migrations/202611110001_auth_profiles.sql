
-- ============================================================
-- RetinaSaarthi: Authentication, Roles and Assessment History
-- Migration: 202611110001_auth_profiles.sql
-- ============================================================

create extension if not exists pgcrypto;

-- ------------------------------------------------------------
-- 1. User profiles
-- ------------------------------------------------------------

create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  full_name text not null default ''
    check (char_length(full_name) <= 120),
  role text not null default 'researcher'
    check (role in ('admin', 'researcher', 'reviewer')),
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- ------------------------------------------------------------
-- 2. Administrator audit log
-- ------------------------------------------------------------

create table if not exists public.admin_audit_log (
  id bigint generated always as identity primary key,
  actor_id uuid not null references auth.users(id),
  target_id uuid not null references auth.users(id),
  action text not null
    check (action in (
      'role_changed',
      'account_activated',
      'account_deactivated'
    )),
  old_value text,
  new_value text,
  created_at timestamptz not null default now()
);

-- ------------------------------------------------------------
-- 3. Assessment history
-- ------------------------------------------------------------

create table if not exists public.assessments (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  filename text not null default 'fundus-image'
    check (char_length(filename) <= 255),
  decision text not null
    check (decision in ('PASS', 'RECAPTURE', 'MANUAL_REVIEW')),
  reason text not null,
  metrics jsonb,
  created_at timestamptz not null default now()
);

create index if not exists assessments_user_created_idx
  on public.assessments(user_id, created_at desc);

-- ------------------------------------------------------------
-- 4. Automatically create a researcher profile on signup
-- ------------------------------------------------------------

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  insert into public.profiles (
    id,
    full_name,
    role,
    is_active
  )
  values (
    new.id,
    left(
      coalesce(
        nullif(trim(new.raw_user_meta_data ->> 'full_name'), ''),
        ''
      ),
      120
    ),
    'researcher',
    true
  );

  return new;
end;
$$;

revoke all on function public.handle_new_user() from public;

drop trigger if exists on_auth_user_created on auth.users;

create trigger on_auth_user_created
after insert on auth.users
for each row
execute function public.handle_new_user();

-- ------------------------------------------------------------
-- 5. Protect profile role and account status
-- ------------------------------------------------------------

create or replace function public.guard_profile_privileges()
returns trigger
language plpgsql
set search_path = ''
as $$
begin
  if auth.uid() = old.id
     and (
       new.role is distinct from old.role
       or new.is_active is distinct from old.is_active
     )
  then
    raise exception
      'Role and account status are managed by administrators';
  end if;

  new.updated_at = now();

  return new;
end;
$$;

drop trigger if exists profiles_guard_privileges
  on public.profiles;

create trigger profiles_guard_privileges
before update on public.profiles
for each row
execute function public.guard_profile_privileges();

-- ------------------------------------------------------------
-- 6. Role-checking functions
-- ------------------------------------------------------------

create or replace function public.is_active_admin()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.profiles p
    where p.id = auth.uid()
      and p.role = 'admin'
      and p.is_active = true
  );
$$;

create or replace function public.has_active_role(
  expected_role text
)
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1
    from public.profiles p
    where p.id = auth.uid()
      and p.role = expected_role
      and p.is_active = true
  );
$$;

revoke all on function public.is_active_admin() from public;
grant execute on function public.is_active_admin()
  to authenticated;

revoke all on function public.has_active_role(text) from public;
grant execute on function public.has_active_role(text)
  to authenticated;

-- ------------------------------------------------------------
-- 7. Enable Row Level Security
-- ------------------------------------------------------------

alter table public.profiles enable row level security;
alter table public.admin_audit_log enable row level security;
alter table public.assessments enable row level security;

-- ------------------------------------------------------------
-- 8. Profile access policies
-- ------------------------------------------------------------

drop policy if exists "read own profile"
  on public.profiles;

create policy "read own profile"
on public.profiles
for select
to authenticated
using (id = auth.uid());

drop policy if exists "admins read all profiles"
  on public.profiles;

create policy "admins read all profiles"
on public.profiles
for select
to authenticated
using (public.is_active_admin());

drop policy if exists "users update own name only"
  on public.profiles;

create policy "users update own name only"
on public.profiles
for update
to authenticated
using (id = auth.uid())
with check (id = auth.uid());

-- Restrict client access to permitted columns.

revoke all on public.profiles from anon, authenticated;

grant select on public.profiles to authenticated;

grant update (full_name)
on public.profiles
to authenticated;

grant select on public.profiles to service_role;

-- ------------------------------------------------------------
-- 9. Audit log access
-- ------------------------------------------------------------

revoke all on public.admin_audit_log
from anon, authenticated;

-- Server-side service role manages audit records.

-- ------------------------------------------------------------
-- 10. Assessment access policies
-- ------------------------------------------------------------

revoke all on public.assessments
from anon, authenticated;

grant select, insert
on public.assessments
to authenticated;

drop policy if exists "read permitted assessments"
  on public.assessments;

create policy "read permitted assessments"
on public.assessments
for select
to authenticated
using (
  (
    user_id = auth.uid()
    and public.has_active_role('researcher')
  )
  or public.is_active_admin()
  or public.has_active_role('reviewer')
);

drop policy if exists "insert own assessments"
  on public.assessments;

create policy "insert own assessments"
on public.assessments
for insert
to authenticated
with check (
  user_id = auth.uid()
  and (
    public.has_active_role('researcher')
    or public.is_active_admin()
  )
);

-- ------------------------------------------------------------
-- 11. Secure administrator profile management
-- ------------------------------------------------------------

create or replace function public.admin_update_profile(
  p_actor_id uuid,
  p_target_id uuid,
  p_role text default null,
  p_is_active boolean default null
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
  old_role text;
  old_active boolean;
begin
  -- Verify the administrator using the database.
  if not exists (
    select 1
    from public.profiles
    where id = p_actor_id
      and role = 'admin'
      and is_active = true
  ) then
    raise exception 'Administrator access required';
  end if;

  if p_role is not null
     and p_role not in ('admin', 'researcher', 'reviewer')
  then
    raise exception 'Unsupported role';
  end if;

  if p_target_id = p_actor_id
     and (
       p_role is not null
       or p_is_active = false
     )
  then
    raise exception
      'Administrators cannot change their own role or deactivate their account';
  end if;

  select role, is_active
  into old_role, old_active
  from public.profiles
  where id = p_target_id
  for update;

  if not found then
    raise exception 'Target account was not found';
  end if;

  update public.profiles
  set
    role = coalesce(p_role, role),
    is_active = coalesce(p_is_active, is_active)
  where id = p_target_id;

  if p_role is not null
     and p_role is distinct from old_role
  then
    insert into public.admin_audit_log (
      actor_id,
      target_id,
      action,
      old_value,
      new_value
    )
    values (
      p_actor_id,
      p_target_id,
      'role_changed',
      old_role,
      p_role
    );
  end if;

  if p_is_active is not null
     and p_is_active is distinct from old_active
  then
    insert into public.admin_audit_log (
      actor_id,
      target_id,
      action,
      old_value,
      new_value
    )
    values (
      p_actor_id,
      p_target_id,
      case
        when p_is_active then 'account_activated'
        else 'account_deactivated'
      end,
      old_active::text,
      p_is_active::text
    );
  end if;
end;
$$;

revoke all
on function public.admin_update_profile(uuid, uuid, text, boolean)
from public, anon, authenticated;

grant execute
on function public.admin_update_profile(uuid, uuid, text, boolean)
to service_role;

-- ------------------------------------------------------------
-- 12. First administrator bootstrap
-- ------------------------------------------------------------

-- Do not automatically create an administrator.
-- First register your account and verify your email.
-- Then obtain your verified user UUID from Supabase Auth.
--
-- Only in the trusted Supabase SQL Editor, execute:
--
-- update public.profiles
-- set role = 'admin'
-- where id = '<VERIFIED_USER_UUID>';
--
-- Never expose service_role credentials to the frontend.
