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
    expect(container.querySelector('.orb-stage')).toBeNull()
    expect(container.textContent).toContain('Слушаю…')
  })
})
