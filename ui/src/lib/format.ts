import type { AppState, VoiceState } from './types'

const voiceStates: Record<VoiceState, string> = {
  off: 'Голос выключен',
  idle: 'Жду обращения',
  listening: 'Слушаю…',
  capturing: 'Слушаю…',
  holding: 'Слушаю, пока клавиши зажаты…',
  recognizing: 'Распознаю…',
  processing: 'Выполняю…',
  paused: 'Прослушивание на паузе',
}

/** Короткая подпись состояния голоса для меню и главной. */
export function voiceLabel(state: AppState | null): string {
  if (!state) return ''
  const voice = state.voice
  // Задача из текста или ответ ещё звучит, а голос в покое (или выключен).
  if (state.busy && (!voice.enabled || voice.state === 'idle')) return 'Отвечаю…'
  if (!voice.enabled) {
    if (voice.loading) return 'Запускаю голос…'
    return voice.problem ? 'Голос недоступен' : voiceStates.off
  }
  if (voice.paused) return voiceStates.paused
  if (voice.state === 'idle' && voice.wake_word) return `Жду «${state.name}»`
  return voiceStates[voice.state] ?? voiceStates.idle
}

export function isListening(state: VoiceState): boolean {
  return state === 'listening' || state === 'capturing' || state === 'holding'
}

/** Форма слова для числа: plural(5, ['шаг', 'шага', 'шагов']) → «шагов». */
export function plural(count: number, forms: [string, string, string]): string {
  const tens = Math.abs(count) % 100
  const ones = tens % 10
  if (tens > 10 && tens < 20) return forms[2]
  if (ones === 1) return forms[0]
  if (ones >= 2 && ones <= 4) return forms[1]
  return forms[2]
}

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1).replace('.', ',')} с` : `${Math.round(ms)} мс`
}

/** «5 минут», «1 час 30 минут», «45 секунд» — как в assistant/timers.py (duration_text). */
export function durationText(seconds: number): string {
  const total = Math.max(1, Math.round(seconds))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = total % 60
  const parts: string[] = []
  if (hours) parts.push(`${hours} ${plural(hours, ['час', 'часа', 'часов'])}`)
  if (minutes) parts.push(`${minutes} ${plural(minutes, ['минуту', 'минуты', 'минут'])}`)
  if (secs && !hours) parts.push(`${secs} ${plural(secs, ['секунду', 'секунды', 'секунд'])}`)
  return parts.join(' ')
}

const pad = (value: number) => String(value).padStart(2, '0')

/** Когда сработает: «в 18:30», «завтра в 9:00», «24.10 в 9:00» (moment_text в Python). */
export function momentText(due: Date, now: Date): string {
  const clock = `${due.getHours()}:${pad(due.getMinutes())}`
  const day = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime()
  const days = Math.round((day(due) - day(now)) / 86_400_000)
  if (days === 0) return `в ${clock}`
  if (days === 1) return `завтра в ${clock}`
  return `${pad(due.getDate())}.${pad(due.getMonth() + 1)} в ${clock}`
}

/** Обратный отсчёт: «4:59», «1:04:05». Секунды — вверх: 0:00 появляется в момент срабатывания. */
export function countdown(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000))
  const hours = Math.floor(total / 3600)
  const minutes = Math.floor((total % 3600) / 60)
  const secs = pad(total % 60)
  return hours ? `${hours}:${pad(minutes)}:${secs}` : `${minutes}:${secs}`
}
