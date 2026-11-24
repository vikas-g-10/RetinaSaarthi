-- Manual Supabase SQL verification for classification-result and profile RLS.
-- Replace UUID placeholders with users/rows in a disposable project.
-- Each prohibited statement must fail; the RPC spoof test must return the caller UUID.

begin;

-- Researcher A must not read Researcher B's result.
set local role authenticated;
select set_config('request.jwt.claims',
  '{"role":"authenticated","sub":"<researcher-a-uuid>"}', true);
select id, user_id
from public.classification_results
where user_id = '<researcher-b-uuid>';

-- Reviewer cannot insert.
select set_config('request.jwt.claims',
  '{"role":"authenticated","sub":"<reviewer-uuid>"}', true);
insert into public.classification_results
  (assessment_id, user_id, predicted_class, predicted_label, probabilities,
   model_name, model_version)
values
  ('<assessment-uuid>', '<reviewer-uuid>', 0, 'No DR',
   '[0.2,0.2,0.2,0.2,0.2]'::jsonb, 'test', 'test');

-- Reviewer cannot change predicted_class.
update public.classification_results
set predicted_class = 4
where id = '<result-uuid>';

-- Reviewer cannot spoof reviewed_by.
update public.classification_results
set review_status = 'reviewed', reviewed_by = '<different-user-uuid>'
where id = '<result-uuid>';

-- Non-admin cannot change profiles.role.
update public.profiles
set role = 'admin'
where id = '<researcher-a-uuid>';

-- Non-admin cannot delete a classification result.
delete from public.classification_results
where id = '<result-uuid>';

-- A reviewer cannot clear an existing reviewer or set another reviewer.
update public.classification_results
set reviewed_by = null
where id = '<reviewed-result-uuid>';

-- The RPC ignores spoofed ownership in both payload objects.
select public.create_classification_result(
  jsonb_build_object(
    'user_id', '<researcher-b-uuid>',
    'filename', 'rls-spoof.png',
    'decision', 'PASS',
    'reason', 'rls test',
    'metrics', null
  ),
  jsonb_build_object(
    'user_id', '<researcher-b-uuid>',
    'predicted_class', 0,
    'predicted_label', 'No DR',
    'probabilities', '[0.2,0.2,0.2,0.2,0.2]'::jsonb,
    'model_name', 'test',
    'model_version', 'test'
  )
);

-- Run the preceding RPC result id as <rpc-result-uuid> and verify ownership.
select a.user_id as assessment_user_id, c.user_id as classification_user_id,
       auth.uid() as authenticated_user_id
from public.assessments a
join public.classification_results c on c.assessment_id = a.id
where c.id = '<rpc-result-uuid>';

rollback;

-- Expected result for each prohibited statement: a PostgreSQL RLS or trigger error.
-- The select policy is authenticated-only; anon has no table privileges or policy.
