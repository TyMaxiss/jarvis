import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { UpdateState } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call, onEvent: () => () => {} }))

import { createMockApi, mockRelease } from '../lib/mock'
import { Home } from './Home'
import { UpdateBanner, UpdatesCard, updateStatus } from './Updates'

async function updateState(changes: Partial<UpdateState> = {}): Promise<UpdateState> {
  const state = await createMockApi(() => {}).get_state()
  return { ...state.update, release: mockRelease, checked: true, ...changes }
}

beforeEach(() => call.mockReset())
afterEach(cleanup)

describe('updateStatus', () => {
  it('одной строкой: идёт установка, проверка, ошибка, новая версия или последняя', async () => {
    const found = await updateState()
    expect(updateStatus({ ...found, installing: true, step: 'Скачиваю обновление: 40%' })).toBe('Скачиваю обновление: 40%')
    expect(updateStatus({ ...found, checking: true })).toBe('Проверяю…')
    expect(updateStatus({ ...found, error: 'Не удалось проверить обновления: нет сети' })).toMatch(/нет сети/)
    expect(updateStatus(found)).toBe('Доступна версия 0.7.0')
    expect(updateStatus({ ...found, release: null })).toBe('У вас последняя версия')
    expect(updateStatus({ ...found, release: null, checked: false })).toBe('')
  })
})

describe('Баннер обновления', () => {
  it('на главной — только когда вышла новая версия', async () => {
    const state = await createMockApi(() => {}).get_state()
    const { rerender } = render(<Home state={state} onOpenSettings={() => {}} />)
    expect(screen.queryByRole('status', { name: 'Обновление' })).toBeNull()
    rerender(<Home state={{ ...state, update: await updateState() }} onOpenSettings={() => {}} />)
    expect(screen.getByRole('status', { name: 'Обновление' }).textContent).toMatch(/Доступна новая версия 0.7.0/)
  })

  it('«Обновить» ставит новую версию, «Что нового» открывает релиз', async () => {
    render(<UpdateBanner update={await updateState()} />)
    expect(screen.getByText('У вас 0.6.0. Обновление займёт пару минут.')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Обновить' }))
    expect(call).toHaveBeenCalledWith('update_install')
    fireEvent.click(screen.getByRole('button', { name: 'Что нового' }))
    expect(call).toHaveBeenCalledWith('update_open_page')
  })

  it('пока обновление скачивается — ход и кнопка занята', async () => {
    const step = 'Скачиваю обновление: 40% (72 МБ из 180 МБ)'
    render(<UpdateBanner update={await updateState({ installing: true, step, progress: 0.4 })} />)
    expect(screen.getByText(step)).toBeTruthy()
    expect(screen.getByRole('progressbar', { name: 'Ход обновления' }).getAttribute('aria-valuenow')).toBe('40')
    expect((screen.getByRole('button', { name: 'Обновить' }) as HTMLButtonElement).disabled).toBe(true)
  })

  it('ошибка обновления видна, а из исходников — подсказка вместо кнопки', async () => {
    const { rerender } = render(<UpdateBanner update={await updateState({ error: 'Обновление не удалось: нет сети' })} />)
    expect(screen.getByText('Обновление не удалось: нет сети')).toBeTruthy()
    rerender(<UpdateBanner update={await updateState({ supported: false })} />)
    expect(screen.getByText(/git pull и uv sync/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Обновить' })).toBeNull()
  })
})

describe('Раздел «Обновления»', () => {
  it('после проверки — новая версия с кнопками или «последняя версия»', async () => {
    const props = { checkUpdates: true, onCheckUpdates: () => {} }
    const { rerender } = render(<UpdatesCard update={await updateState()} {...props} />)
    expect(screen.getByText('Доступна версия 0.7.0')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Обновить' })).toBeTruthy()
    rerender(<UpdatesCard update={await updateState({ release: null })} {...props} />)
    expect(screen.getByText('У вас последняя версия')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Проверить сейчас' })).toBeTruthy()
    rerender(<UpdatesCard update={await updateState({ release: null, checking: true })} {...props} />)
    expect((screen.getByRole('button', { name: 'Проверить сейчас' }) as HTMLButtonElement).disabled).toBe(true)
  })
})
