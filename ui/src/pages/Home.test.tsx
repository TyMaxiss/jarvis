import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AlarmItem } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call, onEvent: () => () => {} }))

import { TimersCard } from './Home'

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
