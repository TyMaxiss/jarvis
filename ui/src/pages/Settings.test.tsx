import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createMockApi } from '../lib/mock'
import type { AssistantApi } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call, onEvent: () => () => {} }))

import { SettingsPage, engineHint } from './Settings'

let api: AssistantApi

beforeEach(() => {
  api = createMockApi(() => {})
  call.mockReset()
  call.mockImplementation((method: keyof AssistantApi, ...args: unknown[]) =>
    (api[method] as (...a: unknown[]) => Promise<unknown>)(...args),
  )
})
afterEach(cleanup)

async function renderPage() {
  const state = await api.get_state()
  await act(async () => {
    render(<SettingsPage updates={state.update} />)
  })
}

describe('SettingsPage', () => {
  it('включает автозапуск отдельно от config.yaml', async () => {
    await renderPage()
    const toggle = screen.getByRole('switch', { name: 'Запускать вместе с Windows' })
    expect(toggle.getAttribute('aria-checked')).toBe('false')
    await act(async () => {
      fireEvent.click(toggle)
      await new Promise((resolve) => setTimeout(resolve, 10))
    })
    expect(call).toHaveBeenCalledWith('save_settings', { general: { autostart: true } })
    expect(toggle.getAttribute('aria-checked')).toBe('true')
  })

  it('без автозапуска переключателя нет', async () => {
    const original = api.get_settings
    api.get_settings = async () => {
      const settings = await original()
      return { ...settings, general: { ...settings.general, autostart: null } }
    }
    await renderPage()
    expect(screen.queryByRole('switch', { name: 'Запускать вместе с Windows' })).toBeNull()
  })

  it('Silero: свой список голосов, подсказка про загрузку, «Прослушать» с выбранным голосом', async () => {
    await renderPage()
    await act(async () => {
      fireEvent.click(screen.getByRole('radio', { name: 'Silero' }))
      await new Promise((resolve) => setTimeout(resolve, 10))
    })
    expect(call).toHaveBeenCalledWith('save_settings', { tts: { engine: 'silero' } })
    expect(screen.getByText(/скачается около 270 МБ/)).toBeTruthy()
    const voice = screen.getByRole('combobox', { name: 'Голос Silero' }) as HTMLSelectElement
    expect(voice.value).toBe('aidar')
    fireEvent.click(screen.getByRole('button', { name: 'Прослушать' }))
    expect(call).toHaveBeenCalledWith('preview_voice', expect.objectContaining({ engine: 'silero', silero_speaker: 'aidar' }))
  })
})

describe('Обновления в настройках', () => {
  it('показывают версию и сохраняют автоматическую проверку', async () => {
    await renderPage()
    expect(screen.getByText('0.6.0')).toBeTruthy()
    expect(screen.getByText('Ещё не проверяли')).toBeTruthy()
    const toggle = screen.getByRole('switch', { name: 'Проверять обновления' })
    expect(toggle.getAttribute('aria-checked')).toBe('true')
    await act(async () => {
      fireEvent.click(toggle)
      await new Promise((resolve) => setTimeout(resolve, 10))
    })
    expect(call).toHaveBeenCalledWith('save_settings', { general: { check_updates: false } })
    fireEvent.click(screen.getByRole('button', { name: 'Проверить сейчас' }))
    expect(call).toHaveBeenCalledWith('update_check')
  })
})

describe('engineHint', () => {
  it('говорит, что скачивается и нужен ли интернет', () => {
    expect(engineHint('silero', true)).not.toMatch(/скачается/)
    expect(engineHint('edge', false)).toMatch(/через интернет/)
    expect(engineHint('piper', false)).toMatch(/Офлайн/)
  })
})
