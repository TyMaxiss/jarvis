import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { initialOverlay, overlayReduce, type OverlayState } from '../lib/overlay'
import type { AssistantEvent } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call }))

import { Orb } from './Orb'

afterEach(cleanup)

const run = (events: AssistantEvent[]): OverlayState => events.reduce(overlayReduce, initialOverlay)

describe('Orb', () => {
  it('слушает продолжение: «Слушаю…», микрофон и итог прошлой команды', () => {
    const state = run([
      { type: 'recognized', payload: { text: 'открой ютуб' } },
      {
        type: 'result',
        payload: { id: 1, time: '12:00', phrase: 'открой ютуб', route: 'local', reply: 'YouTube открыт', ok: true, ms: 300, source: 'voice' },
      },
      { type: 'voice', payload: { state: 'listening', detail: '' } },
    ])
    const { container } = render(<Orb state={state} level={0.5} present />)
    expect(screen.getByText('Слушаю…')).toBeTruthy()
    expect(screen.getByText('YouTube открыт')).toBeTruthy()
    const stage = container.firstElementChild!
    expect(stage.classList.contains('orb-present')).toBe(true)
    expect(stage.getAttribute('data-mode')).toBe('listening')
    expect(container.querySelectorAll('.orb-layer')).toHaveLength(5)
    expect(container.querySelector('.orb-dots')!.getAttribute('data-state')).toBe('voice')
  })

  it('пока окно не показано, сфера спрятана (страница проиграет появление по событию)', () => {
    const { container } = render(<Orb state={initialOverlay} level={0} present={false} />)
    expect(container.firstElementChild!.classList.contains('orb-present')).toBe(false)
  })

  it('выполнение: фраза, шаг и волна точек', () => {
    const state = run([
      { type: 'recognized', payload: { text: 'открой ютуб' } },
      { type: 'step', payload: { number: 1, total: 2, action: 'open_url', text: 'Открываю YouTube', ok: true } },
    ])
    const { container } = render(<Orb state={state} level={0.9} present />)
    expect(screen.getByText('«открой ютуб»')).toBeTruthy()
    expect(screen.getByText('Шаг 1 из 2 · Открываю YouTube')).toBeTruthy()
    expect(container.querySelector('.orb-dots')!.getAttribute('data-state')).toBe('working')
  })

  it('подтверждение отвечает кнопками', () => {
    const state = run([{ type: 'confirm', payload: { id: 'c1', question: 'Выключить компьютер?' } }])
    render(<Orb state={state} level={0} present />)
    expect(screen.getByText('Выключить компьютер?')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Да' }))
    fireEvent.click(screen.getByRole('button', { name: 'Нет' }))
    expect(call).toHaveBeenCalledWith('confirm_answer', 'c1', true)
    expect(call).toHaveBeenCalledWith('confirm_answer', 'c1', false)
  })
})
