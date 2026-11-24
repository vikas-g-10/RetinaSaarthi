import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';

const cors = { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type', 'Access-Control-Allow-Methods': 'POST, OPTIONS' };
const reply = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { ...cors, 'Content-Type': 'application/json' } });
const roles = new Set(['admin', 'researcher', 'reviewer']);

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: cors });
  if (req.method !== 'POST') return reply({ error: 'Method not allowed' }, 405);
  const url = Deno.env.get('SUPABASE_URL'); const anon = Deno.env.get('SUPABASE_ANON_KEY'); const service = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY');
  if (!url || !anon || !service) return reply({ error: 'Server configuration is incomplete' }, 500);
  const authorization = req.headers.get('Authorization');
  if (!authorization?.startsWith('Bearer ')) return reply({ error: 'Authentication required' }, 401);
  const token = authorization.slice(7);
  const callerClient = createClient(url, anon, { auth: { persistSession: false, autoRefreshToken: false }, global: { headers: { Authorization: `Bearer ${token}` } } });
  const { data: { user: caller }, error: callerError } = await callerClient.auth.getUser(token);
  if (callerError || !caller) return reply({ error: 'Invalid or expired session' }, 401);
  const adminClient = createClient(url, service, { auth: { persistSession: false, autoRefreshToken: false } });
  const { data: callerProfile, error: profileError } = await adminClient.from('profiles').select('role,is_active').eq('id', caller.id).maybeSingle();
  if (profileError || callerProfile?.role !== 'admin' || !callerProfile.is_active) return reply({ error: 'Administrator access required' }, 403);
  let payload: { action?: string; user_id?: string; role?: string; is_active?: boolean };
  try { payload = await req.json(); } catch { return reply({ error: 'Invalid JSON request' }, 400); }

  if (payload.action === 'list') {
    const allUsers = []; let page = 1;
    while (true) {
      const { data, error } = await adminClient.auth.admin.listUsers({ page, perPage: 1000 });
      if (error) return reply({ error: 'Unable to retrieve user list' }, 500);
      allUsers.push(...data.users);
      if (data.users.length < 1000) break;
      page += 1;
    }
    const { data: profiles, error } = await adminClient.from('profiles').select('id,full_name,role,is_active,created_at');
    if (error) return reply({ error: 'Unable to retrieve profile list' }, 500);
    const profileById = new Map((profiles ?? []).map((p) => [p.id, p]));
    const users = allUsers.map((u) => { const p = profileById.get(u.id); return { id: u.id, email: u.email ?? '', full_name: p?.full_name ?? '', role: p?.role ?? 'researcher', is_active: p?.is_active ?? false, created_at: p?.created_at ?? u.created_at }; });
    return reply({ users });
  }

  if (payload.action !== 'update' || !payload.user_id) return reply({ error: 'Invalid action or target user' }, 400);
  if (payload.user_id === caller.id && (payload.role !== undefined || payload.is_active === false)) return reply({ error: 'Administrators cannot change their own role or deactivate their own account' }, 400);
  if (payload.role !== undefined && !roles.has(payload.role)) return reply({ error: 'Unsupported role' }, 400);
  if (payload.is_active !== undefined && typeof payload.is_active !== 'boolean') return reply({ error: 'is_active must be a boolean' }, 400);
  if (payload.role === undefined && payload.is_active === undefined) return reply({ error: 'No supported changes provided' }, 400);
  const { error: updateError } = await adminClient.rpc('admin_update_profile', {
    p_actor_id: caller.id, p_target_id: payload.user_id,
    p_role: payload.role ?? null, p_is_active: payload.is_active ?? null,
  });
  if (updateError) return reply({ error: updateError.message.includes('not found') ? 'Target account was not found' : 'Account update failed' }, updateError.message.includes('not found') ? 404 : 400);
  return reply({ success: true });
});
