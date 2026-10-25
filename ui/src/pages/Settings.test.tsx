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

describe('Голоса по ключу', () => {
  const tick = () => new Promise((resolve) => setTimeout(resolve, 10))

  it('OpenAI: голос, манера речи в духе Джарвиса, ключ сохраняется кнопкой', async () => {
    await renderPage()
    await act(async () => {
      fireEvent.click(screen.getByRole('radio', { name: 'OpenAI' }))
      await tick()
    })
    expect(call).toHaveBeenCalledWith('save_settings', { tts: { engine: 'openai' } })
    expect((screen.getByRole('combobox', { name: 'Голос OpenAI' }) as HTMLSelectElement).value).toBe('cedar')
    const style = screen.getByRole('textbox', { name: 'Манера речи' }) as HTMLTextAreaElement
    expect(style.value).toMatch(/ИИ-дворецкий/)
    expect(screen.queryByRole('button', { name: 'Как у Джарвиса' })).toBeNull() // и так по умолчанию
    // Своя манера сохраняется, когда уходят с поля; «Как у Джарвиса» возвращает исходную.
    await act(async () => {
      fireEvent.change(style, { target: { value: 'Говори бодро и быстро' } })
      fireEvent.blur(style)
      await tick()
    })
    expect(call).toHaveBeenCalledWith('save_settings', { tts: { openai_style: 'Говори бодро и быстро' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Как у Джарвиса' }))
      await tick()
    })
    expect((call.mock.calls.at(-1) as unknown[])[1]).toEqual({ tts: { openai_style: expect.stringMatching(/ИИ-дворецкий/) } })

    const key = screen.getByLabelText('Ключ голоса') as HTMLInputElement
    expect(key.placeholder).toBe('Вставьте ключ: sk-…')
    expect(screen.getByText(/Тот же ключ, что у ChatGPT/)).toBeTruthy()
    await act(async () => {
      fireEvent.change(key, { target: { value: 'sk-test-1234abcd' } })
      fireEvent.click(screen.getByRole('button', { name: 'Сохранить ключ голоса' }))
      await tick()
    })
    expect(call).toHaveBeenCalledWith('tts_save_key', 'openai', 'sk-test-1234abcd')
    expect(key.value).toBe('')
    expect(key.placeholder).toBe('Сохранён ••••abcd — вставьте новый, чтобы заменить')
    fireEvent.click(screen.getByRole('button', { name: 'Прослушать' }))
    expect(call).toHaveBeenCalledWith('preview_voice', expect.objectContaining({ engine: 'openai', openai_voice: 'cedar' }))
  })

  it('Яндекс: свой список голосов и ключ Yandex Cloud, без манеры речи', async () => {
    await renderPage()
    await act(async () => {
      fireEvent.click(screen.getByRole('radio', { name: 'Яндекс' }))
      await tick()
    })
    expect((screen.getByRole('combobox', { name: 'Голос Яндекса' }) as HTMLSelectElement).value).toBe('kirill:strict')
    expect(screen.queryByRole('textbox', { name: 'Манера речи' })).toBeNull()
    expect((screen.getByLabelText('Ключ голоса') as HTMLInputElement).placeholder).toBe('Вставьте API-ключ Yandex Cloud')
    fireEvent.click(screen.getByRole('button', { name: 'Где взять ключ голоса' }))
    expect(call).toHaveBeenCalledWith('open_tts_key_page', 'yandex')
  })

  it('несохранённый ключ не переходит к другому голосу', async () => {
    await renderPage()
    await act(async () => {
      fireEvent.click(screen.getByRole('radio', { name: 'Яндекс' }))
      await tick()
    })
    fireEvent.change(screen.getByLabelText('Ключ голоса'), { target: { value: 'AQVN-yandex-key' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('radio', { name: 'OpenAI' }))
      await tick()
    })
    // Иначе «Сохранить» записал бы ключ Яндекса вместо ключа OpenAI — общего с ChatGPT.
    expect((screen.getByLabelText('Ключ голоса') as HTMLInputElement).value).toBe('')
    expect((screen.getByRole('button', { name: 'Сохранить ключ голоса' }) as HTMLButtonElement).disabled).toBe(true)
  })
})

describe('Оформление оверлея', () => {
  it('сфера выбирается и сохраняется сразу, без оверлея выбор недоступен', async () => {
    await renderPage()
    const sphere = screen.getByRole('radio', { name: 'Сфера' })
    expect(screen.getByRole('radio', { name: 'Плашка' }).getAttribute('aria-checked')).toBe('true')
    await act(async () => {
      fireEvent.click(sphere)
      await new Promise((resolve) => setTimeout(resolve, 10))
    })
    expect(call).toHaveBeenCalledWith('save_settings', { ui: { overlay_style: 'orb' } })
    expect(sphere.getAttribute('aria-checked')).toBe('true')
    await act(async () => {
      fireEvent.click(screen.getByRole('switch', { name: 'Оверлей' }))
      await new Promise((resolve) => setTimeout(resolve, 10))
    })
    expect((sphere as HTMLButtonElement).disabled).toBe(true)
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
    expect(engineHint('openai', false)).toMatch(/по ключу/)
    expect(engineHint('yandex', false)).toMatch(/работает в России/)
  })
})
