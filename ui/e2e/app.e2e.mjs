// Сквозная проверка интерфейса с настоящим Python за ним (запускает tests/test_ui_e2e.py).
// Аргументы: адрес моста, папка для снимков экрана, ключ API для проверки.
// Печатает JSON с итогами шагов; проверки файлов и запросов — на стороне Python.

import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'

const require = createRequire(process.env.PLAYWRIGHT_MODULE || import.meta.url)
const { chromium } = require('playwright')

const [base, shots, key] = process.argv.slice(2)
const bridge = readFileSync(new URL('./bridge.js', import.meta.url), 'utf8')
const browser = await chromium.launch()
const results = {}
const errors = []

async function open(view, size) {
  const page = await browser.newPage({ viewport: size })
  page.on('pageerror', (error) => errors.push(`${view}: ${error}`))
  page.on('console', (message) => message.type() === 'error' && errors.push(`${view}: ${message.text()}`))
  await page.addInitScript(bridge)
  await page.goto(`${base}/index.html${view === 'main' ? '' : `?view=${view}`}`)
  return page
}

try {
  // Главное окно: состояние из Python.
  const main = await open('main', { width: 1100, height: 760 })
  await main.getByRole('heading', { level: 1 }).waitFor()
  results.heading = await main.getByRole('heading', { level: 1 }).textContent()
  results.llmCard = await main.getByText('DeepSeek').first().textContent()

  // Настройки: ChatGPT с ключом — проверка связи и сохранение.
  await main.getByRole('button', { name: 'Настройки' }).click()
  await main.getByRole('radio', { name: /ChatGPT/ }).click()
  await main.getByLabel('Ключ API').fill(key)
  await main.getByRole('button', { name: 'Проверить' }).click()
  await main.getByRole('status').waitFor({ timeout: 15000 })
  results.check = await main.getByRole('status').textContent()
  await main.getByRole('button', { name: 'Использовать ChatGPT' }).click()
  await main.getByText(/^Сохранено/).waitFor({ timeout: 15000 })
  results.saved = await main.getByText(/^Сохранено/).textContent()
  results.chatgptTile = await main.getByRole('radio', { name: /ChatGPT/ }).textContent()
  results.keyPlaceholder = await main.getByLabel('Ключ API').getAttribute('placeholder')
  await main.screenshot({ path: `${shots}/e2e-settings.png` })

  // Тема сохраняется в config.yaml и сразу применяется.
  await main.getByRole('radio', { name: 'Светлая' }).click()
  await main.waitForFunction(() => document.documentElement.dataset.theme === 'light')
  results.theme = await main.evaluate(() => document.documentElement.dataset.theme)

  // Быстрый чат: команда выполняется конвейером, ход приходит событиями.
  const chat = await open('chat', { width: 620, height: 400 })
  await chat.getByLabel('Команда').fill('сделай громче')
  await chat.keyboard.press('Enter')
  await chat.getByText(/^Готово/).waitFor({ timeout: 15000 })
  results.chatSteps = await chat.locator('li').first().textContent()
  await chat.screenshot({ path: `${shots}/e2e-chat.png` })

  // Главная получила эту команду в «Последних».
  await main.getByRole('button', { name: 'Главная' }).click()
  await main.getByText('сделай громче').first().waitFor({ timeout: 5000 })
  results.recent = await main.getByText('сделай громче').first().textContent()
  await main.screenshot({ path: `${shots}/e2e-home.png` })
} catch (error) {
  errors.push(String(error))
  for (const [index, page] of browser.contexts().flatMap((context) => context.pages()).entries()) {
    await page.screenshot({ path: `${shots}/e2e-error-${index}.png` }).catch(() => {})
  }
} finally {
  await browser.close()
}
console.log(JSON.stringify({ results, errors }))
