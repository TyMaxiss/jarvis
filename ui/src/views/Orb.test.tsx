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
  it('слушает продолжение: «Слушаю…», микрофон и прошлая команда с итогом', () => {
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
    expect(screen.getByText('Открой ютуб')).toBeTruthy()
    expect(screen.getByText('YouTube открыт')).toBeTruthy()
    const stage = container.firstElementChild!
    expect(stage.classList.contains('orb-present')).toBe(true)
    expect(stage.classList.contains('orb-glass')).toBe(false)
    expect(stage.getAttribute('data-mode')).toBe('listening')
    expect(container.querySelectorAll('.orb-disc')).toHaveLength(1) // внешний слой — неподвижный диск по форме окна
    expect(container.querySelectorAll('.orb-layer')).toHaveLength(4)
    expect(container.querySelector('.orb-dots')!.getAttribute('data-state')).toBe('voice')
    expect(container.querySelector('.orb-caret')).toBeNull() // фраза прошлая — курсора нет
  })

  it('слова по ходу речи — крупно, с курсором', () => {
    const state = run([
      { type: 'voice', payload: { state: 'capturing', detail: '' } },
      { type: 'partial', payload: { text: 'сделай громче' } },
    ])
    const { container } = render(<Orb state={state} level={0.5} present glass />)
    expect(container.querySelector('.orb-phrase')!.textContent).toBe('Сделай громче')
    expect(container.querySelector('.orb-caret')).not.toBeNull()
    expect(container.firstElementChild!.classList.contains('orb-glass')).toBe(true) // полупрозрачные кольцо и карточка
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
    expect(screen.getByText('Выполняю…')).toBeTruthy()
    expect(screen.getByText('Открой ютуб')).toBeTruthy()
    expect(screen.getByText('Шаг 1 из 2 · Открываю YouTube')).toBeTruthy()
    expect(container.querySelector('.orb-dots')!.getAttribute('data-state')).toBe('working')
  })

  it('клик прячет сферу с итогом команды', () => {
    const state = run([
      { type: 'recognized', payload: { text: 'громче' } },
      { type: 'result', payload: { id: 1, time: '12:00', phrase: 'громче', route: 'local', reply: '', ok: true, ms: 300, source: 'voice' } },
    ])
    const { container } = render(<Orb state={state} level={0} present />)
    fireEvent.click(container.querySelector('.orb-card')!)
    expect(call).toHaveBeenCalledWith('overlay_dismiss')
  })

  it('подтверждение отвечает кнопками, а клик мимо них сферу не прячет', () => {
    call.mockClear()
    const state = run([{ type: 'confirm', payload: { id: 'c1', question: 'Выключить компьютер?' } }])
    const { container } = render(<Orb state={state} level={0} present />)
    expect(screen.getByText('Выключить компьютер?')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Да' }))
    fireEvent.click(screen.getByRole('button', { name: 'Нет' }))
    fireEvent.click(container.querySelector('.orb-card')!)
    expect(call).toHaveBeenCalledWith('confirm_answer', 'c1', true)
    expect(call).toHaveBeenCalledWith('confirm_answer', 'c1', false)
    expect(call).not.toHaveBeenCalledWith('overlay_dismiss')
  })
})
