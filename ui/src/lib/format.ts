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

export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1).replace('.', ',')} с` : `${Math.round(ms)} мс`
}
