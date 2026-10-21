import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createMockApi } from '../lib/mock'
import type { Settings } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call }))

import { LlmSection, missingField } from './LlmSection'

let settings: Settings

beforeEach(async () => {
  settings = await createMockApi(() => {}).get_settings()
  call.mockReset()
})
afterEach(cleanup)

function renderSection(onSettings = vi.fn()) {
  render(<LlmSection settings={settings} onSettings={onSettings} onEnabled={vi.fn()} />)
  return onSettings
}

describe('LlmSection', () => {
  it('показывает провайдеров и отмечает текущего', () => {
    renderSection()
    const current = screen.getByRole('radio', { name: /DeepSeek/ })
    expect(current.getAttribute('aria-checked')).toBe('true')
    expect(current.textContent).toContain('сейчас')
    expect(screen.getByRole('radio', { name: /ChatGPT/ }).textContent).toContain('нужен ключ')
  })

  it('без ключа нельзя ни проверить, ни сохранить; ключ скрыт', () => {
    renderSection()
    fireEvent.click(screen.getByRole('radio', { name: /ChatGPT/ }))
    expect(screen.getByText('Вставьте ключ API')).toBeTruthy()
    expect((screen.getByRole('button', { name: 'Использовать ChatGPT' }) as HTMLButtonElement).disabled).toBe(true)
    const input = screen.getByLabelText('Ключ API') as HTMLInputElement
    expect(input.type).toBe('password')
    fireEvent.change(input, { target: { value: 'sk-test' } })
    expect((screen.getByRole('button', { name: 'Использовать ChatGPT' }) as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Показать ключ' }))
    expect(input.type).toBe('text')
  })

  it('«Проверить» отправляет профиль, модель и ключ и показывает время ответа', async () => {
    call.mockResolvedValue({ ok: true, text: 'Да.', ms: 840, error: '' })
    renderSection()
    fireEvent.click(screen.getByRole('radio', { name: /ChatGPT/ }))
    fireEvent.change(screen.getByLabelText('Ключ API'), { target: { value: ' sk-test ' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Проверить' }))
    })
    expect(call).toHaveBeenCalledWith('llm_test', { profile: 'openai', model: 'gpt-6-luna', key: 'sk-test', base_url: '' })
    expect(screen.getByRole('status').textContent).toContain('840 мс')
  })

  it('ошибка проверки видна целиком', async () => {
    call.mockResolvedValue({ ok: false, text: '', ms: 0, error: 'ключ API не подошёл (профиль openai)' })
    renderSection()
    fireEvent.click(screen.getByRole('radio', { name: /ChatGPT/ }))
    fireEvent.change(screen.getByLabelText('Ключ API'), { target: { value: 'sk-bad' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Проверить' }))
    })
    expect(screen.getByRole('alert').textContent).toContain('ключ API не подошёл')
  })

  it('сохранение передаёт новые настройки наверх', async () => {
    const saved = structuredClone(settings)
    saved.llm.profile = 'openai'
    call.mockResolvedValue({ ok: true, errors: [], restart: false, settings: saved })
    const onSettings = renderSection()
    fireEvent.click(screen.getByRole('radio', { name: /ChatGPT/ }))
    fireEvent.change(screen.getByLabelText('Ключ API'), { target: { value: 'sk-test' } })
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Использовать ChatGPT' }))
    })
    expect(call).toHaveBeenCalledWith('llm_save', { profile: 'openai', model: 'gpt-6-luna', key: 'sk-test', base_url: '' })
    expect(onSettings).toHaveBeenCalledWith(saved)
  })

  it('свой сервис: адрес и модель вводятся вручную', () => {
    renderSection()
    fireEvent.click(screen.getByRole('radio', { name: /Другой сервис/ }))
    expect(screen.getByText('Укажите адрес API')).toBeTruthy()
    fireEvent.change(screen.getByLabelText('Адрес API'), { target: { value: 'http://localhost:1234/v1' } })
    fireEvent.change(screen.getByLabelText('Модель'), { target: { value: 'local-model' } })
    // Ключ своему сервису не обязателен (например, LM Studio без ключа).
    expect(screen.queryByText('Вставьте ключ API')).toBeNull()
    expect((screen.getByRole('button', { name: 'Использовать Другой сервис' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByLabelText('Ключ API') as HTMLInputElement).placeholder).toBe('Ключ, если сервису он нужен')
  })
})

describe('missingField', () => {
  it('Ollama без ключа готова, если есть адрес и модель', () => {
    const ollama = settings.llm.profiles.find((p) => p.name === 'ollama')!
    expect(missingField(ollama, { model: 'qwen3:8b', key: '', base_url: ollama.base_url, otherModel: false })).toBe('')
    expect(missingField(ollama, { model: '', key: '', base_url: ollama.base_url, otherModel: false })).toBe('Выберите модель')
  })
})
