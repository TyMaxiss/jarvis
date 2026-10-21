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
  if (!voice.enabled) return voice.problem ? 'Голос недоступен' : voiceStates.off
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
