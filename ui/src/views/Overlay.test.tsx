import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createMockApi } from '../lib/mock'
import type { AppState, AssistantEvent } from '../lib/types'

const { call, listeners } = vi.hoisted(() => ({
  call: vi.fn(),
  listeners: new Set<(event: AssistantEvent) => void>(),
}))
vi.mock('../lib/bridge', () => ({
  call,
  onEvent: (listener: (event: AssistantEvent) => void) => {
    listeners.add(listener)
    return () => listeners.delete(listener)
  },
}))

import { Overlay } from './Overlay'

afterEach(cleanup)

const emit = (event: AssistantEvent) => act(() => listeners.forEach((listener) => listener(event)))

describe('Overlay', () => {
  it('меняет оформление, только когда оверлей спрятан', async () => {
    const state: AppState = await createMockApi(() => {}).get_state()
    const withStyle = (overlay_style: 'classic' | 'orb'): AssistantEvent => ({
      type: 'state',
      payload: { ...state, ui: { ...state.ui, overlay_style } },
    })
    call.mockImplementation(async () => withStyle('orb').payload)
    const { container } = render(<Overlay />)
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0))
    })
    expect(container.querySelector('.orb-stage')).not.toBeNull()
    await emit({ type: 'overlay', payload: { visible: true } })
    await emit(withStyle('classic')) // выбрали плашку, пока сфера на экране — окно ещё под сферу
    expect(container.querySelector('.orb-stage')).not.toBeNull()
    await emit({ type: 'overlay', payload: { visible: false } })
    expect(container.querySelector('.orb-stage')).not.toBeNull() // окно сферы ещё уезжает
    await act(() => new Promise((resolve) => setTimeout(resolve, 350)))
    expect(container.querySelector('.orb-stage')).toBeNull()
    expect(container.textContent).toContain('Слушаю…')
  })

  it('сфера полупрозрачная, если так сказал Python при показе; плашка прячется кликом', async () => {
    const state: AppState = await createMockApi(() => {}).get_state()
    const withStyle = (overlay_style: 'classic' | 'orb'): AssistantEvent => ({
      type: 'state',
      payload: { ...state, ui: { ...state.ui, overlay_style } },
    })
    const { container } = render(<Overlay />)
    await emit(withStyle('orb'))
    await emit({ type: 'overlay', payload: { visible: true, glass: true } })
    expect(container.querySelector('.orb-glass')).not.toBeNull()
    await emit({ type: 'overlay', payload: { visible: false, glass: false } }) // уезжая, сфера не меняется
    expect(container.querySelector('.orb-glass')).not.toBeNull()
    await emit({ type: 'overlay', payload: { visible: true, glass: false } }) // «Эффекты прозрачности» выключили
    expect(container.querySelector('.orb-stage')).not.toBeNull()
    expect(container.querySelector('.orb-glass')).toBeNull()
    await emit({ type: 'overlay', payload: { visible: false } })
    cleanup()

    const pill = render(<Overlay />)
    await emit(withStyle('classic'))
    await emit({ type: 'voice', payload: { state: 'capturing', detail: '' } })
    await emit({ type: 'partial', payload: { text: 'сделай громче' } })
    expect(pill.container.textContent).toContain('Сделай громче')
    call.mockClear()
    await act(async () => {
      ;(pill.container.firstElementChild as HTMLElement).click()
    })
    expect(call).toHaveBeenCalledWith('overlay_dismiss')
  })
})
