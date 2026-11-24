import { expect, test, type Page } from '@playwright/test';
import path from 'node:path';

const widths = [1440, 1024, 768, 390];
const screenshotDir = path.resolve('e2e/screenshots');
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=', 'base64');

const resultRows = [{ id: 'result-1', patient_ref: 'P-001', predicted_label: 'Moderate', probabilities: [0.05, 0.1, 0.7, 0.1, 0.05], review_status: 'pending', model_version: 'experiment3', created_at: '2026-11-12T08:00:00Z', assessments: { filename: 'fundus.png', decision: 'PASS', created_at: '2026-11-12T08:00:00Z', metrics: {} } }];

async function installMocks(page: Page, mode: 'data' | 'empty' | 'error' | 'missing' = 'data') {
  await page.addInitScript(() => {
    localStorage.setItem('retinasaarthi-remember', 'true');
    localStorage.setItem('sb-mock-auth-token', JSON.stringify({ access_token: 'mock-token', refresh_token: 'mock-refresh', expires_at: 4102444800, expires_in: 3600, token_type: 'bearer', user: { id: 'user-1', aud: 'authenticated', role: 'authenticated', email: 'ava@example.com', email_confirmed_at: '2026-01-01T00:00:00Z', app_metadata: {}, user_metadata: {}, identities: [] } }));
  });
  await page.route('https://mock.supabase.co/**', async route => {
    const url = route.request().url();
    if (url.includes('/auth/v1/user')) return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 'user-1', aud: 'authenticated', role: 'authenticated', email: 'ava@example.com', email_confirmed_at: '2026-01-01T00:00:00Z', app_metadata: {}, user_metadata: {}, identities: [] }) });
    if (url.includes('/auth/v1/token')) return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ access_token: 'mock-token', refresh_token: 'mock-refresh', expires_in: 3600, expires_at: 4102444800, token_type: 'bearer', user: { id: 'user-1', aud: 'authenticated', role: 'authenticated', email: 'ava@example.com', email_confirmed_at: '2026-01-01T00:00:00Z', app_metadata: {}, user_metadata: {}, identities: [] } }) });
    if (mode === 'error' && url.includes('classification_results')) return route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ message: 'mock history failure' }) });
    if (url.includes('/rest/v1/profiles')) return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([{ id: 'user-1', full_name: 'Ava Retina', role: 'admin', is_active: true, created_at: '2026-01-01T00:00:00Z' }]) });
    if (url.includes('classification_results')) return route.fulfill({ status: 200, headers: { 'content-range': `0-${mode === 'empty' ? 0 : resultRows.length - 1}/${mode === 'empty' ? 0 : resultRows.length}` }, contentType: 'application/json', body: JSON.stringify(mode === 'empty' ? [] : resultRows) });
    if (url.includes('admin-users')) return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ users: [{ id: 'user-1', email: 'ava@example.com', full_name: 'Ava Retina', role: 'admin', is_active: true, created_at: '2026-01-01T00:00:00Z' }] }) });
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.route('**/api/v1/model/status', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ready: true, model_name: 'EfficientNet-B0', model_version: 'mock', classes: ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'] }) }));
  await page.route('**/api/v1/classification/predict', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ project: 'RetinaSaarthi', result_id: 'result-1', quality: { decision: 'PASS', reason: 'Pass', metrics: null }, prediction: { predicted_class: 2, predicted_label: 'Moderate', probabilities: [0.05, 0.1, 0.7, 0.1, 0.05], model_name: 'EfficientNet-B0', model_version: 'mock' } }) }));
  await page.route('**/api/v1/quality/assess', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ assessment: { decision: 'PASS', reason: 'Pass' }, metrics: null }) }));
  await page.route('**/health', route => route.fulfill({ status: 200, contentType: 'application/json', body: '{}' }));
}

async function markAndAssert(page: Page, name: string, width: number) {
  await page.evaluate(() => { const label = document.createElement('div'); label.textContent = 'layout check with mocked responses'; label.setAttribute('data-e2e-label', 'layout check with mocked responses'); Object.assign(label.style, { position: 'fixed', bottom: '5px', right: '5px', zIndex: '9999', fontSize: '9px', background: '#160f25', color: '#fff', padding: '3px 5px' }); document.body.append(label); });
  const dimensions = await page.evaluate(() => ({ scrollWidth: document.documentElement.scrollWidth, innerWidth: window.innerWidth }));
  expect(dimensions.scrollWidth <= dimensions.innerWidth, `${name}-${width} scrollWidth=${dimensions.scrollWidth} innerWidth=${dimensions.innerWidth}`).toBe(true);
  expect(await page.locator('[aria-current="page"]').count(), `${name}-${width} url=${page.url()}`).toBeGreaterThan(0);
  expect(await page.getByText('Ava').count()).toBeGreaterThan(0);
  await page.screenshot({ path: path.join(screenshotDir, `${name}-${width}.png`), fullPage: true });
}

test.beforeEach(async ({ page }) => { await installMocks(page); });

test('authenticated pages remain within every requested breakpoint', async ({ page }) => {
  for (const width of widths) {
    await page.setViewportSize({ width, height: 900 });
    for (const [name, url] of [['dashboard', '/'], ['history', '/history'], ['reports', '/reports'], ['profile', '/profile'], ['result', '/results/result-1'], ['admin-users', '/admin/users']] as const) {
      await page.goto(url);
      await page.waitForLoadState('domcontentloaded');
      await markAndAssert(page, name, width);
    }
    await page.goto('/classification');
    await page.waitForLoadState('domcontentloaded');
    await markAndAssert(page, 'classification-idle', width);
  }
});

test('responsive shell and profile menu keyboard behavior', async ({ page }) => {
  await page.setViewportSize({ width: 768, height: 900 });
  await page.goto('/');
  await expect(page.locator('.sidebar-new')).not.toHaveClass(/open/);
  await page.getByRole('button', { name: 'Open navigation' }).click();
  await expect(page.locator('.sidebar-new')).toHaveClass(/open/);
  await page.getByRole('button', { name: /Ava Retina|Ava/ }).press('Enter');
  await expect(page.locator('.profile-menu')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.profile-menu')).toHaveCount(0);
  await page.goto('/');
  await expect.poll(() => page.locator('.stats-grid-new').evaluate(element => getComputedStyle(element).gridTemplateColumns.split(' ').length)).toBe(2);
});

test('classification mocked result and recapture states', async ({ page }) => {
  await page.goto('/classification');
  await page.setInputFiles('input[type=file]', { name: 'fundus.png', mimeType: 'image/png', buffer: png });
  await page.getByRole('button', { name: /Run screening/ }).click();
  await expect(page.getByRole('heading', { name: 'Moderate' })).toBeVisible();
  await page.screenshot({ path: path.join(screenshotDir, 'classification-result-1440.png'), fullPage: true });
  await page.route('**/api/v1/classification/predict', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ quality: { decision: 'RECAPTURE', reason: 'Glare detected; recapture image.', metrics: null }, prediction: null }) }));
  await page.goto('/classification');
  await page.setInputFiles('input[type=file]', { name: 'fundus.png', mimeType: 'image/png', buffer: png });
  await page.getByRole('button', { name: /Run screening/ }).click();
  await expect(page.getByText('Glare detected; recapture image.')).toBeVisible();
  await expect(page.locator('.probability-row')).toHaveCount(0);
});

test('history empty, error, and database-not-migrated states', async ({ page }) => {
  await installMocks(page, 'empty');
  await page.goto('/history');
  await expect(page.getByText('No matching results')).toBeVisible();
  await installMocks(page, 'error');
  await page.goto('/history');
  await expect(page.getByText('Unable to load this data')).toBeVisible();
  await installMocks(page, 'missing');
  await page.route('https://mock.supabase.co/rest/v1/classification_results**', route => route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ message: 'relation classification_results does not exist' }) }));
  await page.goto('/history');
  await expect(page.getByText('Database not migrated')).toBeVisible();
});

test('empty dashboard and classification states at every breakpoint', async ({ page }) => {
  test.setTimeout(120_000);
  for (const width of widths) {
    await installMocks(page, 'empty');
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/');
    await expect(page.getByText('No screening activity')).toBeVisible();
    await markAndAssert(page, 'dashboard-empty', width);

    await installMocks(page, 'data');
    await page.goto('/classification');
    await page.setInputFiles('input[type=file]', { name: 'fundus.png', mimeType: 'image/png', buffer: png });
    await page.getByRole('button', { name: /Run screening/ }).click();
    await expect(page.getByRole('heading', { name: 'Moderate' })).toBeVisible();
    await markAndAssert(page, 'classification-result', width);

    await page.route('**/api/v1/classification/predict', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ quality: { decision: 'RECAPTURE', reason: 'Glare detected; recapture image.', metrics: null }, prediction: null }) }));
    await page.goto('/classification');
    await page.setInputFiles('input[type=file]', { name: 'fundus.png', mimeType: 'image/png', buffer: png });
    await page.getByRole('button', { name: /Run screening/ }).click();
    await expect(page.getByText('Glare detected; recapture image.')).toBeVisible();
    await markAndAssert(page, 'classification-recapture', width);
  }
});

test('history state screenshots at every breakpoint', async ({ page }) => {
  test.setTimeout(120_000);
  for (const width of widths) {
    await page.setViewportSize({ width, height: 900 });
    await installMocks(page, 'empty');
    await page.goto('/history');
    await expect(page.getByText('No matching results')).toBeVisible();
    await markAndAssert(page, 'history-empty', width);
    await installMocks(page, 'error');
    await page.goto('/history');
    await expect(page.getByText('Unable to load this data')).toBeVisible();
    await markAndAssert(page, 'history-error', width);
    await installMocks(page, 'missing');
    await page.route('https://mock.supabase.co/rest/v1/classification_results**', route => route.fulfill({ status: 404, contentType: 'application/json', body: JSON.stringify({ message: 'relation classification_results does not exist' }) }));
    await page.goto('/history');
    await expect(page.getByText('Database not migrated')).toBeVisible();
    await markAndAssert(page, 'history-database-not-migrated', width);
  }
});
