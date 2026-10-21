// Что показывает оверлей: состояние одной команды от «Слушаю…» до ответа.

import type { AssistantEvent, Step } from './types'

export type OverlayMode = 'listening' | 'recognizing' | 'working' | 'done' | 'error'

export interface OverlayState {
  mode: OverlayMode
  heard: string // распознанная фраза
  step: Step | null // последний шаг
  reply: string
  confirm: { id: string; question: string } | null
}

export const initialOverlay: OverlayState = { mode: 'listening', heard: '', step: null, reply: '', confirm: null }

export function overlayReduce(state: OverlayState, event: AssistantEvent): OverlayState {
  switch (event.type) {
    case 'voice': {
      const voice = event.payload.state
      if (voice === 'listening' || voice === 'capturing' || voice === 'holding') {
        // Новая фраза. Задачу, которая ещё выполняется, не сбрасываем.
        return state.mode === 'working' ? state : initialOverlay
      }
      if (voice === 'recognizing') return { ...initialOverlay, mode: 'recognizing' }
      return state
    }
    case 'recognized':
      return { ...initialOverlay, mode: 'working', heard: event.payload.text }
    case 'step':
      return { ...state, mode: 'working', step: event.payload }
    case 'reply':
      return { ...state, reply: event.payload.text }
    case 'result':
      return {
        ...state,
        mode: event.payload.ok ? 'done' : 'error',
        heard: state.heard || event.payload.phrase,
        reply: event.payload.reply || state.reply,
        confirm: null,
      }
    case 'confirm':
      return { ...state, confirm: event.payload }
    case 'confirm_done':
      return state.confirm?.id === event.payload.id ? { ...state, confirm: null } : state
    default:
      return state
  }
}

/** Подпись шага: «Шаг 2 из 5 · Открываю Блокнот». */
export function stepLabel(step: Step): string {
  const number = step.total ? `Шаг ${step.number} из ${step.total}` : `Шаг ${step.number}`
  return step.text ? `${number} · ${step.text}` : number
}

/** Две строки оверлея: главная и пояснение. */
export function overlayLines(state: OverlayState): [string, string] {
  if (state.confirm) return [state.confirm.question, 'Скажите «да» или «нет»']
  const heard = state.heard ? `«${state.heard}»` : ''
  switch (state.mode) {
    case 'listening':
      return ['Слушаю…', '']
    case 'recognizing':
      return ['Распознаю…', '']
    case 'working':
      return [heard || 'Выполняю…', state.reply || (state.step ? stepLabel(state.step) : 'Выполняю…')]
    case 'done':
      return state.reply ? [state.reply, heard] : [heard || 'Готово', state.step ? stepLabel(state.step) : 'Готово']
    case 'error':
      return [state.reply || 'Не получилось', heard]
  }
}
