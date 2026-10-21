import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createMockApi } from '../lib/mock'
import type { AssistantApi } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call, onEvent: () => () => {} }))

import { SettingsPage } from './Settings'

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
  await act(async () => {
    render(<SettingsPage />)
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
})
