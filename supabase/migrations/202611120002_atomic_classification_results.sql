-- Insert the assessment and its classification result atomically under the caller's JWT.
-- SECURITY INVOKER preserves auth.uid(), RLS, and the existing role policies.

create or replace function public.create_classification_result(
  p_assessment jsonb,
  p_classification jsonb
)
returns uuid
language plpgsql
security invoker
set search_path = public
as $$
declare
  new_assessment_id uuid;
  new_classification_id uuid;
begin
  insert into public.assessments (
    user_id,
    filename,
    decision,
    reason,
    metrics
  )
  values (
    (p_assessment->>'user_id')::uuid,
    p_assessment->>'filename',
    p_assessment->>'decision',
    p_assessment->>'reason',
    p_assessment->'metrics'
  )
  returning id into new_assessment_id;

  insert into public.classification_results (
    assessment_id,
    user_id,
    predicted_class,
    predicted_label,
    probabilities,
    model_name,
    model_version,
    patient_ref
  )
  values (
    new_assessment_id,
    (p_classification->>'user_id')::uuid,
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

revoke all on function public.create_classification_result(jsonb, jsonb) from public;
grant execute on function public.create_classification_result(jsonb, jsonb) to authenticated;
