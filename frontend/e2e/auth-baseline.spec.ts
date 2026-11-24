import { test, expect } from '@playwright/test';

test('auth computed-style baseline', async ({ page }) => {
  await page.goto('/login');
  const values = await page.evaluate(() => {
    const pick = (selector: string) => getComputedStyle(document.querySelector(selector) as Element);
    const pageStyle = pick('.auth-page');
    const cardStyle = pick('.auth-card');
    const inputStyle = pick('.auth-field input');
    const buttonStyle = pick('.auth-submit');
    return { background: pageStyle.backgroundColor, cardRadius: cardStyle.borderRadius, inputRadius: inputStyle.borderRadius, buttonGradient: buttonStyle.backgroundImage, bodyFont: getComputedStyle(document.body).fontFamily, headingFont: pick('.auth-heading h1').fontFamily };
  });
  console.log(`AUTH_BASELINE ${JSON.stringify(values)}`);
  expect(values.background).toBe('rgb(7, 6, 13)');
  expect(values.cardRadius).toBe('22px');
  expect(values.inputRadius).toBe('9px');
  expect(values.buttonGradient).toContain('rgb(112, 67, 188)');
  expect(values.buttonGradient).toContain('rgb(128, 80, 213)');
  expect(values.bodyFont).toContain('DM Sans');
  expect(values.headingFont).toContain('Space Grotesk');
});
