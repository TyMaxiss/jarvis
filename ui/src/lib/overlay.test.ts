import { describe, expect, it } from 'vitest'
import { initialOverlay, overlayLines, overlayReduce, type OverlayState } from './overlay'
import type { AssistantEvent, RecentItem } from './types'

const run = (events: AssistantEvent[], state: OverlayState = initialOverlay) => events.reduce(overlayReduce, state)

const result = (values: Partial<RecentItem>): AssistantEvent => ({
  type: 'result',
  payload: { id: 1, time: '12:00', phrase: 'открой блокнот', route: 'local', reply: '', ok: true, ms: 300, source: 'voice', ...values },
})

describe('overlayReduce', () => {
  it('ведёт команду от «Слушаю…» до ответа', () => {
    let state = run([{ type: 'voice', payload: { state: 'listening', detail: '' } }])
    expect(overlayLines(state)).toEqual(['Слушаю…', ''])
    state = run([{ type: 'voice', payload: { state: 'recognizing', detail: '' } }], state)
    expect(overlayLines(state)[0]).toBe('Распознаю…')
    state = run([{ type: 'recognized', payload: { text: 'открой блокнот' } }], state)
    expect(overlayLines(state)).toEqual(['«открой блокнот»', 'Выполняю…'])
    state = run([{ type: 'step', payload: { number: 1, total: 2, action: 'open_app', text: 'Открываю Блокнот', ok: true } }], state)
    expect(overlayLines(state)[1]).toBe('Шаг 1 из 2 · Открываю Блокнот')
    state = run([result({ reply: 'Готово' })], state)
    expect(state.mode).toBe('done')
    expect(overlayLines(state)).toEqual(['Готово', '«открой блокнот»'])
  })

  it('ошибка показывает ответ и фразу', () => {
    const state = run([{ type: 'recognized', payload: { text: 'сделай кофе' } }, result({ ok: false, reply: 'Не знаю такой команды' })])
    expect(state.mode).toBe('error')
    expect(overlayLines(state)).toEqual(['Не знаю такой команды', '«сделай кофе»'])
  })

  it('подтверждение: вопрос и кнопки, пока не придёт ответ', () => {
    let state = run([
      { type: 'recognized', payload: { text: 'выключи компьютер' } },
      { type: 'confirm', payload: { id: 'c1', question: 'Выключить компьютер?' } },
    ])
    expect(overlayLines(state)).toEqual(['Выключить компьютер?', 'Скажите «да» или «нет»'])
    state = run([{ type: 'confirm_done', payload: { id: 'other' } }], state)
    expect(state.confirm).not.toBeNull()
    state = run([{ type: 'confirm_done', payload: { id: 'c1' } }], state)
    expect(state.confirm).toBeNull()
  })

  it('новая фраза после ответа начинается с чистого листа, а выполняющаяся задача не сбрасывается', () => {
    const done = run([{ type: 'recognized', payload: { text: 'громче' } }, result({ reply: 'Громкость 60' })])
    expect(run([{ type: 'voice', payload: { state: 'listening', detail: '' } }], done)).toEqual(initialOverlay)
    const working = run([{ type: 'recognized', payload: { text: 'громче' } }])
    expect(run([{ type: 'voice', payload: { state: 'capturing', detail: '' } }], working)).toBe(working)
  })
})
