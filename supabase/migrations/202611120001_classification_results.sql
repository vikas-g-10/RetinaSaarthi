-- Additive classification-result storage for authenticated screening results.

create table if not exists public.classification_results (
  id uuid primary key default gen_random_uuid(),
  assessment_id uuid not null references public.assessments(id) on delete cascade,
  user_id uuid not null references auth.users(id),
  predicted_class smallint not null check (predicted_class between 0 and 4),
  predicted_label text not null,
  probabilities jsonb not null check (
    jsonb_typeof(probabilities) = 'array'
    and jsonb_array_length(probabilities) = 5
  ),
  model_name text not null,
  model_version text not null,
  patient_ref text null check (patient_ref is null or char_length(patient_ref) <= 120),
  review_status text not null default 'pending' check (review_status in ('pending', 'reviewed', 'flagged')),
  reviewed_by uuid null references auth.users(id),
  reviewed_at timestamptz null,
  review_note text null,
  created_at timestamptz not null default now()
);

create index if not exists classification_results_user_created_idx
  on public.classification_results(user_id, created_at desc);
create index if not exists classification_results_assessment_idx
  on public.classification_results(assessment_id);

alter table public.classification_results enable row level security;

revoke all on public.classification_results from anon, authenticated;
grant select, insert on public.classification_results to authenticated;
grant update (review_status, reviewed_by, reviewed_at, review_note)
  on public.classification_results to authenticated;

drop policy if exists "read permitted classification results" on public.classification_results;
create policy "read permitted classification results"
on public.classification_results
for select to authenticated
using (
  (
    user_id = auth.uid()
    and public.has_active_role('researcher')
  )
  or public.is_active_admin()
  or public.has_active_role('reviewer')
);

drop policy if exists "insert own classification results" on public.classification_results;
create policy "insert own classification results"
on public.classification_results
for insert to authenticated
with check (
  user_id = auth.uid()
  and (
    public.has_active_role('researcher')
    or public.is_active_admin()
  )
);

drop policy if exists "reviewers update review fields" on public.classification_results;
create policy "reviewers update review fields"
on public.classification_results
for update to authenticated
using (public.is_active_admin() or public.has_active_role('reviewer'))
with check (public.is_active_admin() or public.has_active_role('reviewer'));

create or replace function public.guard_classification_result_updates()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
begin
  if new.reviewed_by is not null and new.reviewed_by <> auth.uid() then
    raise exception 'reviewed_by must match the authenticated user';
  end if;

  if tg_op = 'INSERT' then
    if new.review_status <> 'pending'
       or new.reviewed_by is not null
       or new.reviewed_at is not null
       or new.review_note is not null
    then
      raise exception 'New classification results must begin in pending review state';
    end if;
  elsif not public.is_active_admin() then
    if new.assessment_id is distinct from old.assessment_id
       or new.user_id is distinct from old.user_id
       or new.predicted_class is distinct from old.predicted_class
       or new.predicted_label is distinct from old.predicted_label
       or new.probabilities is distinct from old.probabilities
       or new.model_name is distinct from old.model_name
       or new.model_version is distinct from old.model_version
       or new.patient_ref is distinct from old.patient_ref
       or new.created_at is distinct from old.created_at
    then
      raise exception 'Only review fields may be changed';
    end if;
  end if;
  return new;
end;
$$;

drop trigger if exists classification_result_update_guard
  on public.classification_results;
create trigger classification_result_update_guard
before insert or update on public.classification_results
for each row execute function public.guard_classification_result_updates();

drop policy if exists "admins delete classification results" on public.classification_results;
create policy "admins delete classification results"
on public.classification_results
for delete to authenticated
using (public.is_active_admin());
