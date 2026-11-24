-- Harden classification-result ownership, review transitions, and RPC execution.

create or replace function public.create_classification_result(
  p_assessment jsonb,
  p_classification jsonb
)
returns uuid
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  caller_id uuid := auth.uid();
  new_assessment_id uuid;
  new_classification_id uuid;
begin
  if caller_id is null then
    raise exception 'Authentication is required';
  end if;

  insert into public.assessments (
    user_id, filename, decision, reason, metrics
  )
  values (
    caller_id,
    p_assessment->>'filename',
    p_assessment->>'decision',
    p_assessment->>'reason',
    p_assessment->'metrics'
  )
  returning id into new_assessment_id;

  insert into public.classification_results (
    assessment_id, user_id, predicted_class, predicted_label,
    probabilities, model_name, model_version, patient_ref
  )
  values (
    new_assessment_id,
    caller_id,
    (p_classification->>'predicted_class')::smallint,
    p_classification->>'predicted_label',
    p_classification->'probabilities',
    p_classification->>'model_name',
    p_classification->>'model_version',
    p_classification->>'patient_ref'
  )
  returning id into new_classification_id;

  return new_classification_id;
end;
$$;

revoke all on function public.create_classification_result(jsonb, jsonb) from public, anon;
grant execute on function public.create_classification_result(jsonb, jsonb) to authenticated;

create or replace function public.guard_classification_result_updates()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  caller_is_admin boolean := public.is_active_admin();
begin
  if tg_op = 'INSERT' then
    if new.review_status <> 'pending'
       or new.reviewed_by is not null
       or new.reviewed_at is not null
       or new.review_note is not null then
      raise exception 'New classification results must begin in pending review state';
    end if;
    return new;
  end if;

  if not caller_is_admin then
    if new.assessment_id is distinct from old.assessment_id
       or new.user_id is distinct from old.user_id
       or new.predicted_class is distinct from old.predicted_class
       or new.predicted_label is distinct from old.predicted_label
       or new.probabilities is distinct from old.probabilities
       or new.model_name is distinct from old.model_name
       or new.model_version is distinct from old.model_version
       or new.patient_ref is distinct from old.patient_ref
       or new.created_at is distinct from old.created_at then
      raise exception 'Only review fields may be changed';
    end if;

    if old.reviewed_by is not null and new.reviewed_by is null then
      raise exception 'A reviewer cannot clear reviewed_by';
    end if;

    if new.reviewed_by is not null and new.reviewed_by <> auth.uid() then
      raise exception 'reviewed_by must match the authenticated user';
    end if;

    if new.review_status in ('reviewed', 'flagged') then
      if auth.uid() is null or new.reviewed_by is distinct from auth.uid() then
        raise exception 'The authenticated reviewer must be recorded';
      end if;
      new.reviewed_at := now();
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

revoke all on public.classification_results from anon;

drop policy if exists "read permitted classification results"
  on public.classification_results;
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
