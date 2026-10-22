// Оверлей в браузере (?view=overlay&mock): события, которые прислал бы Python.
// Без параметров команды идут по кругу; ?scene=… застывает на одном состоянии (для снимков):
// listening, followup (слушает после «YouTube открыт», как на эскизе), working, done, error, confirm.
// ?level=0.6 — постоянная громкость вместо «речи».

import type { AssistantEvent, RecentItem } from './types'

type Emit = (event: AssistantEvent) => void

const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

function result(phrase: string, reply: string, ok = true): AssistantEvent {
  const item: RecentItem = { id: Date.now(), time: '12:00', phrase, route: 'local', reply, ok, ms: 420, source: 'voice' }
  return { type: 'result', payload: item }
}

/** Громкость «речи»: слоги по 120–260 мс с паузами. */
function speech(started: number, now: number): number {
  const t = (now - started) / 1000
  const syllable = Math.max(0, Math.sin(t * 9.3) * Math.sin(t * 2.1 + 0.6))
  return Math.min(1, syllable * 0.75 + Math.abs(Math.sin(t * 23)) * 0.08)
}

/** Сцены: что прислать, чтобы оверлей застыл в нужном состоянии. */
const scenes: Record<string, AssistantEvent[]> = {
  listening: [{ type: 'voice', payload: { state: 'listening', detail: '' } }],
  followup: [
    { type: 'recognized', payload: { text: 'открой ютуб' } },
    result('открой ютуб', 'YouTube открыт'),
    { type: 'voice', payload: { state: 'listening', detail: '' } },
  ],
  working: [
    { type: 'recognized', payload: { text: 'открой ютуб и включи музыку' } },
    { type: 'step', payload: { number: 1, total: 2, action: 'open_url', text: 'Открываю YouTube', ok: true } },
  ],
  done: [{ type: 'recognized', payload: { text: 'открой ютуб' } }, result('открой ютуб', 'YouTube открыт')],
  error: [{ type: 'recognized', payload: { text: 'сделай кофе' } }, result('сделай кофе', 'Не знаю такой команды', false)],
  confirm: [
    { type: 'recognized', payload: { text: 'выключи компьютер' } },
    { type: 'confirm', payload: { id: 'c1', question: 'Выключить компьютер?' } },
  ],
}

export function startOverlayDemo(emit: Emit, params: URLSearchParams): void {
  const fixed = params.get('level')
  const started = Date.now()
  let speaking = false
  window.setInterval(() => {
    const value = fixed !== null ? Number(fixed) : speaking ? speech(started, Date.now()) : 0.02
    emit({ type: 'level', payload: { value } })
  }, 70)

  const scene = params.get('scene')
  if (scene && scenes[scene]) {
    emit({ type: 'overlay', payload: { visible: true } })
    for (const event of scenes[scene]) emit(event)
    speaking = scene === 'listening' || scene === 'followup'
    return
  }

  void (async () => {
    for (;;) {
      emit({ type: 'voice', payload: { state: 'listening', detail: '' } })
      emit({ type: 'overlay', payload: { visible: true } })
      speaking = true
      await pause(2600)
      speaking = false
      emit({ type: 'voice', payload: { state: 'recognizing', detail: '' } })
      await pause(600)
      emit({ type: 'recognized', payload: { text: 'открой ютуб' } })
      emit({ type: 'step', payload: { number: 1, total: null, action: 'open_url', text: 'Открываю YouTube', ok: true } })
      await pause(1300)
      emit(result('открой ютуб', 'YouTube открыт'))
      await pause(1500)
      emit({ type: 'voice', payload: { state: 'listening', detail: '' } })
      speaking = true
      await pause(2400)
      speaking = false
      emit({ type: 'voice', payload: { state: 'idle', detail: '' } })
      await pause(1200)
      emit({ type: 'overlay', payload: { visible: false } })
      await pause(1400)
    }
  })()
}
