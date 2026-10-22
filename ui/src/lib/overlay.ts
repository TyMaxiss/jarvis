// Что показывает оверлей: одна команда от «Слушаю…» до ответа, слова фразы по ходу речи и итог
// прошлой команды — он остаётся на экране, пока не прозвучит новая фраза.

import type { AssistantEvent, Step } from './types'

// idle — голос не слушает: видна прошлая команда (last) или «Ничего не услышал».
export type OverlayMode = 'listening' | 'recognizing' | 'working' | 'done' | 'error' | 'stopped' | 'idle'

/** Кончившаяся команда: что от неё видно до следующей фразы. */
export interface Finished {
  mode: 'done' | 'error' | 'stopped'
  heard: string
  step: Step | null
  reply: string
}

export interface OverlayState {
  mode: OverlayMode
  words: string // слова фразы по ходу речи (Vosk), пока её говорят и распознают
  heard: string // распознанная фраза (Whisper)
  step: Step | null // последний шаг
  reply: string
  confirm: { id: string; question: string } | null
  last: Finished | null // прошлая команда: видна, пока ассистент слушает продолжение и после
}

export const initialOverlay: OverlayState = {
  mode: 'listening',
  words: '',
  heard: '',
  step: null,
  reply: '',
  confirm: null,
  last: null,
}

const finished = (mode: OverlayMode): mode is Finished['mode'] => mode === 'done' || mode === 'error' || mode === 'stopped'

export function overlayReduce(state: OverlayState, event: AssistantEvent): OverlayState {
  const next = reduceCommand(state, event)
  if (next === state || !finished(next.mode)) return next
  return { ...next, last: { mode: next.mode, heard: next.heard, step: next.step, reply: next.reply } }
}

function reduceCommand(state: OverlayState, event: AssistantEvent): OverlayState {
  switch (event.type) {
    case 'voice': {
      const voice = event.payload.state
      if (voice === 'listening' || voice === 'capturing' || voice === 'holding') {
        // Ждёт фразу; прошлая команда видна, пока её не начали говорить. Задачу, которая ещё
        // выполняется, не сбрасываем.
        return state.mode === 'working' ? state : { ...initialOverlay, last: state.last }
      }
      // Новая фраза распознаётся: прошлой команде пора уступить место.
      if (voice === 'recognizing') return { ...initialOverlay, mode: 'recognizing', words: state.words }
      if (voice === 'idle' || voice === 'paused' || voice === 'off') {
        // Не дождался фразы (или Whisper ничего не услышал): снова видна прошлая команда.
        return state.mode === 'listening' || state.mode === 'recognizing' ? { ...state, mode: 'idle', words: '' } : state
      }
      return state
    }
    case 'partial':
      return state.mode === 'listening' ? { ...state, words: event.payload.text } : state
    case 'overlay':
      // Оверлей показался заново: прошлые команды уже не к месту.
      return event.payload.visible && state.last ? { ...state, last: null } : state
    case 'recognized':
      return { ...initialOverlay, mode: 'working', heard: event.payload.text }
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

/** Что показать: в покое — прошлую команду так, будто она только что кончилась. */
export function shown(state: OverlayState): OverlayState {
  return state.mode === 'idle' && state.last ? { ...state, ...state.last } : state
}

/** Фраза для показа: с заглавной буквы и без точки в конце («сделай громче» → «Сделай громче»). */
export function phrase(text: string): string {
  const clean = text.trim().replace(/(?<!\.)\.$/, '')
  return clean.charAt(0).toUpperCase() + clean.slice(1)
}

/** Конец длинной фразы, которую ещё говорят: начало уходит за «…», последние слова видны. */
export function tail(text: string, limit = 34): string {
  if (text.length <= limit) return text
  const end = text.slice(-limit)
  const space = end.indexOf(' ') // первое слово, скорее всего, обрезано — с него не начинаем
  return `…${space >= 0 && space < limit - 8 ? end.slice(space + 1) : end}`
}

/** Подпись шага: «Шаг 2 из 5 · Открываю Блокнот». */
export function stepLabel(step: Step): string {
  const number = step.total ? `Шаг ${step.number} из ${step.total}` : `Шаг ${step.number}`
  return step.text ? `${number} · ${step.text}` : number
}

/** Чем кончилась команда: строка результата. */
export interface Outcome {
  tone: 'ok' | 'error' | 'muted'
  text: string
}

/** Итог команды, если она уже кончилась: ответ ассистента, иначе итог последнего шага. */
export function outcomeOf(state: Pick<OverlayState, 'mode' | 'reply' | 'step'>): Outcome | null {
  switch (state.mode) {
    case 'done':
      return { tone: 'ok', text: state.reply || state.step?.text || 'Готово' }
    case 'error':
      return { tone: 'error', text: state.reply || 'Не получилось' }
    case 'stopped':
      return { tone: 'muted', text: 'Остановлено' }
    default:
      return null
  }
}

const FINISHED_STATUS: Record<Finished['mode'], string> = { done: 'Готово', error: 'Не получилось', stopped: 'Остановлено' }

/** Карточка сферы: значок и строка состояния, крупно — фраза, внизу — ход или итог команды. */
export interface OrbCard {
  badge: OverlayMode | 'confirm'
  status: string // «Слушаю…», «Готово»
  text: string // слова, которые говорят, или фраза команды
  live: boolean // text ещё говорят — за ним курсор
  line: Outcome | null
}

export function orbCard(input: OverlayState): OrbCard {
  if (input.confirm) return { badge: 'confirm', status: 'Подтвердите', text: input.confirm.question, live: false, line: null }
  const state = shown(input)
  switch (state.mode) {
    case 'listening': {
      const last = state.last
      if (state.words || !last) return { badge: 'listening', status: 'Слушаю…', text: tail(phrase(state.words)), live: true, line: null }
      // Слушает продолжение: видна прошлая команда, как «Сделай громче ✓ Громкость 60%».
      return { badge: 'listening', status: 'Слушаю…', text: phrase(last.heard), live: false, line: finishedLine(last) }
    }
    case 'recognizing':
      return { badge: 'recognizing', status: 'Распознаю…', text: tail(phrase(state.words)), live: false, line: null }
    case 'working': {
      const progress = state.reply || (state.step ? stepLabel(state.step) : '')
      return {
        badge: 'working',
        status: 'Выполняю…',
        text: phrase(state.heard),
        live: false,
        line: progress ? { tone: 'muted', text: progress } : null,
      }
    }
    case 'idle':
      return { badge: 'idle', status: 'Ничего не услышал', text: '', live: false, line: null }
    default:
      return { badge: state.mode, status: FINISHED_STATUS[state.mode], text: phrase(state.heard), live: false, line: finishedLine(state) }
  }
}

/** Строка итога под фразой; без неё, если итог повторил бы строку состояния («Готово», «Остановлено»). */
function finishedLine(state: Pick<OverlayState, 'mode' | 'reply' | 'step'>): Outcome | null {
  const outcome = outcomeOf(state)
  return outcome && outcome.text !== FINISHED_STATUS[state.mode as Finished['mode']] ? outcome : null
}

/** Две строки плашки: главная и пояснение. */
export function overlayLines(input: OverlayState): [string, string] {
  if (input.confirm) return [input.confirm.question, 'Скажите «да» или «нет»']
  const state = shown(input)
  const heard = state.heard ? `«${state.heard}»` : ''
  switch (state.mode) {
    case 'listening':
      if (state.words) return [phrase(state.words), 'Слушаю…']
      return ['Слушаю…', state.last ? lastLine(state.last) : '']
    case 'recognizing':
      return state.words ? [phrase(state.words), 'Распознаю…'] : ['Распознаю…', '']
    case 'working':
      return [heard || 'Выполняю…', state.reply || (state.step ? stepLabel(state.step) : 'Выполняю…')]
    case 'done':
      return state.reply ? [state.reply, heard] : [heard || 'Готово', state.step ? stepLabel(state.step) : 'Готово']
    case 'error':
      return [state.reply || 'Не получилось', heard]
    case 'stopped':
      return ['Остановлено', heard]
    case 'idle':
      return ['Ничего не услышал', '']
  }
}

/** Прошлая команда одной строкой: «сделай громче» · Громкость 60%. */
function lastLine(last: Finished): string {
  return [last.heard && `«${last.heard}»`, outcomeOf(last)?.text].filter(Boolean).join(' · ')
}
