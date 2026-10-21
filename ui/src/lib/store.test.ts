import { describe, expect, it } from 'vitest'
import { createMockApi } from './mock'
import { applyTheme, reduce } from './store'
import type { RecentItem } from './types'

describe('reduce', () => {
  it('событие state заменяет состояние целиком, остальные без состояния игнорируются', async () => {
    const state = await createMockApi(() => {}).get_state()
    expect(reduce(null, { type: 'state', payload: state })).toBe(state)
    expect(reduce(null, { type: 'voice', payload: { state: 'listening', detail: '' } })).toBeNull()
  })

  it('voice меняет состояние голоса, result добавляет команду в начало без повторов', async () => {
    const state = await createMockApi(() => {}).get_state()
    const listening = reduce(state, { type: 'voice', payload: { state: 'listening', detail: '' } })!
    expect(listening.voice.state).toBe('listening')
    const item: RecentItem = { ...state.recent[0], id: 99, phrase: 'новая' }
    const next = reduce(listening, { type: 'result', payload: item })!
    expect(next.recent[0]).toBe(item)
    expect(reduce(next, { type: 'result', payload: item })!.recent.filter((r) => r.id === 99)).toHaveLength(1)
  })
})

describe('applyTheme', () => {
  it('ставит тему и принимает только цвет вида #rrggbb', () => {
    applyTheme({ theme: 'light', accent: '#22b8cf' })
    expect(document.documentElement.dataset.theme).toBe('light')
    expect(document.documentElement.style.getPropertyValue('--accent')).toBe('#22b8cf')
    applyTheme({ theme: 'dark', accent: 'red; background: url(x)' })
    expect(document.documentElement.style.getPropertyValue('--accent')).toBe('#22b8cf')
  })
})
