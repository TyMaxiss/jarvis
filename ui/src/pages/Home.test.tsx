import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AlarmItem, AppState } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call, onEvent: () => () => {} }))

import { createMockApi } from '../lib/mock'
import { Home, TimersCard } from './Home'

const NOW = new Date(2026, 9, 21, 17, 0, 0)
const at = (seconds: number) => new Date(NOW.getTime() + seconds * 1000).toISOString()

const timers: AlarmItem[] = [
  { id: 't1', kind: 'timer', due: at(90), text: '', seconds: 300 },
  { id: 'r1', kind: 'reminder', due: at(3600 * 25), text: 'позвонить маме', seconds: 0 },
]

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date', 'setInterval', 'clearInterval'] })
  vi.setSystemTime(NOW)
  call.mockReset()
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('TimersCard', () => {
  it('отсчитывает время и отменяет таймер', () => {
    render(<TimersCard timers={timers} />)
    expect(screen.getByText('Таймер')).toBeTruthy()
    expect(screen.getByText('на 5 минут')).toBeTruthy()
    expect(screen.getByLabelText('Осталось').textContent).toBe('1:30')
    act(() => {
      vi.advanceTimersByTime(1000)
    })
    expect(screen.getByLabelText('Осталось').textContent).toBe('1:29')

    expect(screen.getByText('позвонить маме')).toBeTruthy()
    expect(screen.getByText('завтра в 18:00')).toBeTruthy()
    expect(screen.getAllByLabelText('Осталось')).toHaveLength(1) // дальше суток — без отсчёта

    fireEvent.click(screen.getByRole('button', { name: 'Отменить таймер' }))
    expect(call).toHaveBeenCalledWith('timer_cancel', 't1')
  })
})

describe('Кнопка микрофона', () => {
  async function stateWith(voiceState: AppState['voice']['state'], busy = false, enabled = true): Promise<AppState> {
    const state = await createMockApi(() => {}).get_state()
    return { ...state, busy, voice: { ...state.voice, state: voiceState, enabled } }
  }

  it('в покое — слушать команду', async () => {
    render(<Home state={await stateWith('idle')} onOpenSettings={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Слушать команду' }))
    expect(call).toHaveBeenCalledWith('voice_listen')
  })

  it.each([
    ['слушает', 'listening', false],
    ['распознаёт', 'recognizing', false],
    ['выполняет голосовую команду', 'processing', false],
    ['отвечает на команду из текста', 'idle', true],
  ] as const)('%s — остановить', async (_, voiceState, busy) => {
    render(<Home state={await stateWith(voiceState, busy)} onOpenSettings={() => {}} />)
    fireEvent.click(screen.getByRole('button', { name: 'Остановить' }))
    expect(call).toHaveBeenCalledWith('voice_stop')
    expect(call).not.toHaveBeenCalledWith('voice_listen')
  })

  it('без голоса кнопка недоступна, но задачу из текста остановить можно', async () => {
    const { rerender } = render(<Home state={await stateWith('off', false, false)} onOpenSettings={() => {}} />)
    expect((screen.getByRole('button', { name: 'Слушать команду' }) as HTMLButtonElement).disabled).toBe(true)
    rerender(<Home state={await stateWith('off', true, false)} onOpenSettings={() => {}} />)
    const stop = screen.getByRole('button', { name: 'Остановить' }) as HTMLButtonElement
    expect(stop.disabled).toBe(false)
    fireEvent.click(stop)
    expect(call).toHaveBeenCalledWith('voice_stop')
  })
})

describe('Главная, пока голос загружается', () => {
  async function loadingState(step: string, progress: number | null): Promise<AppState> {
    const state = await createMockApi(() => {}).get_state()
    return { ...state, voice: { ...state.voice, enabled: false, loading: true, step, progress } }
  }

  it('показывает, что скачивается, и полоску хода', async () => {
    const step = 'Скачиваю библиотеки CUDA: 34% (450 МБ из 1,3 ГБ)'
    render(<Home state={await loadingState(step, 0.34)} onOpenSettings={() => {}} />)
    expect(screen.getByText('Запускаю голос…')).toBeTruthy()
    expect(screen.getByText(step)).toBeTruthy()
    expect(screen.getByRole('progressbar', { name: 'Ход загрузки' }).getAttribute('aria-valuenow')).toBe('34')
    expect(screen.getByText(/только при первом запуске/)).toBeTruthy()
  })

  it('без шага — обычная подпись и без полоски', async () => {
    render(<Home state={await loadingState('', null)} onOpenSettings={() => {}} />)
    expect(screen.getByText('Загружаю распознавание речи — это несколько секунд.')).toBeTruthy()
    expect(screen.queryByRole('progressbar')).toBeNull()
    expect(screen.getByText('Команды текстом уже работают.')).toBeTruthy()
  })
})
