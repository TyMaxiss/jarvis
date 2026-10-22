// Что показывает оверлей: состояние одной команды от «Слушаю…» до ответа.

import type { AssistantEvent, Step } from './types'

export type OverlayMode = 'listening' | 'recognizing' | 'working' | 'done' | 'error' | 'stopped'

export interface OverlayState {
  mode: OverlayMode
  heard: string // распознанная фраза
  step: Step | null // последний шаг
  reply: string
  confirm: { id: string; question: string } | null
  last: Outcome | null // итог прошлой команды: сфера показывает его, пока слушает продолжение
}

export const initialOverlay: OverlayState = { mode: 'listening', heard: '', step: null, reply: '', confirm: null, last: null }

export function overlayReduce(state: OverlayState, event: AssistantEvent): OverlayState {
  const next = reduceCommand(state, event)
  const outcome = outcomeOf(next)
  return outcome && next !== state ? { ...next, last: outcome } : next
}

function reduceCommand(state: OverlayState, event: AssistantEvent): OverlayState {
  const fresh = { ...initialOverlay, last: state.last } // новая фраза помнит итог прошлой
  switch (event.type) {
    case 'voice': {
      const voice = event.payload.state
      if (voice === 'listening' || voice === 'capturing' || voice === 'holding') {
        // Новая фраза. Задачу, которая ещё выполняется, не сбрасываем.
        return state.mode === 'working' ? state : fresh
      }
      if (voice === 'recognizing') return { ...fresh, mode: 'recognizing' }
      return state
    }
    case 'overlay':
      // Оверлей показался заново: прошлые команды уже не к месту.
      return event.payload.visible && state.last ? { ...state, last: null } : state
    case 'recognized':
      return { ...fresh, mode: 'working', heard: event.payload.text }
    case 'step':
      return { ...state, mode: 'working', step: event.payload }
    case 'reply':
      return { ...state, reply: event.payload.text }
    case 'stopped':
      return { ...state, mode: 'stopped', confirm: null }
    case 'result':
      if (state.mode === 'stopped') return state // прерванная задача — не ошибка
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

/** Чем кончилась команда: строка результата в карточке сферы. */
export interface Outcome {
  tone: 'ok' | 'error' | 'muted'
  text: string
}

/** Итог команды, если она уже кончилась (сфера показывает его и пока слушает продолжение). */
export function outcomeOf(state: OverlayState): Outcome | null {
  switch (state.mode) {
    case 'done':
      return { tone: 'ok', text: state.reply || 'Готово' }
    case 'error':
      return { tone: 'error', text: state.reply || 'Не получилось' }
    case 'stopped':
      return { tone: 'muted', text: 'Остановлено' }
    default:
      return null
  }
}

/** Карточка сферы: заголовок (что происходит или что услышано) и строка под ним.
 *
 * Пока ассистент слушает, внизу остаётся итог прошлой команды, как «YouTube открыт». */
export function orbCard(state: OverlayState): { title: string; heard: boolean; line: Outcome | null } {
  if (state.confirm) return { title: 'Подтвердите', heard: false, line: { tone: 'muted', text: state.confirm.question } }
  const heard = state.heard ? `«${state.heard}»` : ''
  switch (state.mode) {
    case 'listening':
      return { title: 'Слушаю…', heard: false, line: state.last }
    case 'recognizing':
      return { title: 'Распознаю…', heard: false, line: state.last }
    case 'working': {
      const progress = state.reply || (state.step ? stepLabel(state.step) : '')
      return { title: heard || 'Выполняю…', heard: Boolean(heard), line: progress ? { tone: 'muted', text: progress } : null }
    }
    default:
      return { title: heard || outcomeOf(state)!.text, heard: Boolean(heard), line: heard ? outcomeOf(state) : null }
  }
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
    case 'stopped':
      return ['Остановлено', heard]
  }
}
