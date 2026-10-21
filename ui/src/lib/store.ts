// Общее состояние приложения: загружается из Python и обновляется его событиями.

import { useEffect, useRef, useSyncExternalStore } from 'react'
import { call, onEvent } from './bridge'
import type { AppState, AssistantEvent, UiSettings } from './types'

let current: AppState | null = null
const subscribers = new Set<() => void>()
let started = false

function set(next: AppState): void {
  current = next
  applyTheme(next.ui)
  for (const notify of [...subscribers]) notify()
}

/** Применяет тему и акцентный цвет к документу. */
export function applyTheme(ui: Pick<UiSettings, 'theme' | 'accent'>): void {
  const root = document.documentElement
  root.dataset.theme = ui.theme
  if (/^#[0-9a-f]{6}$/i.test(ui.accent)) root.style.setProperty('--accent', ui.accent)
}

/** Как событие меняет состояние (вынесено для тестов). */
export function reduce(state: AppState | null, event: AssistantEvent): AppState | null {
  if (event.type === 'state') return event.payload
  if (!state) return state
  switch (event.type) {
    case 'voice':
      return { ...state, voice: { ...state.voice, state: event.payload.state } }
    case 'result': {
      const recent = [event.payload, ...state.recent.filter((item) => item.id !== event.payload.id)]
      return { ...state, recent: recent.slice(0, 20) }
    }
    default:
      return state
  }
}

function start(): void {
  if (started) return
  started = true
  onEvent((event) => {
    const next = reduce(current, event)
    if (next && next !== current) set(next)
  })
  void call('get_state').then((state) => {
    if (state) set(state)
  })
}

function subscribe(notify: () => void): () => void {
  subscribers.add(notify)
  return () => {
    subscribers.delete(notify)
  }
}

export function useAppState(): AppState | null {
  start()
  return useSyncExternalStore(subscribe, () => current)
}

/** Подписка на события Python на время жизни компонента. */
export function useAssistantEvent(listener: (event: AssistantEvent) => void): void {
  const ref = useRef(listener)
  useEffect(() => {
    ref.current = listener
  })
  useEffect(() => onEvent((event) => ref.current(event)), [])
}
