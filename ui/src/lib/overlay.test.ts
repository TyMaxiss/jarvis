import { describe, expect, it } from 'vitest'
import { initialOverlay, orbCard, overlayLines, overlayReduce, phrase, tail, type OverlayState } from './overlay'
import type { AssistantEvent, RecentItem, VoiceState } from './types'

const run = (events: AssistantEvent[], state: OverlayState = initialOverlay) => events.reduce(overlayReduce, state)

const voice = (state: VoiceState): AssistantEvent => ({ type: 'voice', payload: { state, detail: '' } })
const partial = (text: string): AssistantEvent => ({ type: 'partial', payload: { text } })
const result = (values: Partial<RecentItem>): AssistantEvent => ({
  type: 'result',
  payload: { id: 1, time: '12:00', phrase: 'открой блокнот', route: 'local', reply: '', ok: true, ms: 300, source: 'voice', ...values },
})

describe('overlayReduce', () => {
  it('ведёт команду от «Слушаю…» до ответа', () => {
    let state = run([voice('listening')])
    expect(overlayLines(state)).toEqual(['Слушаю…', ''])
    state = run([voice('recognizing')], state)
    expect(overlayLines(state)[0]).toBe('Распознаю…')
    state = run([{ type: 'recognized', payload: { text: 'открой блокнот' } }], state)
    expect(overlayLines(state)).toEqual(['«открой блокнот»', 'Выполняю…'])
    state = run([{ type: 'step', payload: { number: 1, total: 2, action: 'open_app', text: 'Открываю Блокнот', ok: true } }], state)
    expect(overlayLines(state)[1]).toBe('Шаг 1 из 2 · Открываю Блокнот')
    state = run([result({ reply: 'Готово' })], state)
    expect(state.mode).toBe('done')
    expect(overlayLines(state)).toEqual(['Готово', '«открой блокнот»'])
  })

  it('слова по ходу речи видны, пока фразу говорят и распознают', () => {
    let state = run([voice('listening'), voice('capturing'), partial('сделай')])
    expect(overlayLines(state)).toEqual(['Сделай', 'Слушаю…'])
    state = run([partial('сделай громче'), voice('processing'), voice('recognizing')], state)
    expect(state.words).toBe('сделай громче')
    expect(overlayLines(state)).toEqual(['Сделай громче', 'Распознаю…'])
    state = run([{ type: 'recognized', payload: { text: 'Сделай громче.' } }], state)
    expect(state.words).toBe('') // дальше — точный текст Whisper
    expect(overlayLines(state)[0]).toBe('«Сделай громче.»')
  })

  it('шум вместо фразы: слова убираются', () => {
    const state = run([voice('capturing'), partial('сделай'), partial(''), voice('listening')])
    expect(state.words).toBe('')
    expect(overlayLines(state)).toEqual(['Слушаю…', ''])
  })

  it('слова не перебивают выполняющуюся задачу', () => {
    const working = run([{ type: 'recognized', payload: { text: 'расскажи сказку' } }])
    expect(run([voice('holding'), partial('стоп')], working)).toBe(working)
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

  it('«стоп» показывает «Остановлено», и результат прерванной задачи его не меняет', () => {
    let state = run([{ type: 'recognized', payload: { text: 'расскажи сказку' } }, { type: 'stopped', payload: {} }])
    expect(state.mode).toBe('stopped')
    expect(overlayLines(state)).toEqual(['Остановлено', '«расскажи сказку»'])
    state = run([result({ ok: false, reply: 'Жили-были…' })], state)
    expect(overlayLines(state)).toEqual(['Остановлено', '«расскажи сказку»'])
    expect(run([voice('listening')], state)).toEqual({
      ...initialOverlay,
      last: { mode: 'stopped', heard: 'расскажи сказку', step: null, reply: '' },
    })
  })

  it('итог команды держится до следующей фразы, а выполняющаяся задача не сбрасывается', () => {
    const done = run([{ type: 'recognized', payload: { text: 'громче' } }, result({ reply: 'Громкость 60' })])
    let state = run([voice('listening')], done)
    expect(state).toEqual({ ...initialOverlay, last: { mode: 'done', heard: 'громче', step: null, reply: 'Громкость 60' } })
    expect(overlayLines(state)).toEqual(['Слушаю…', '«громче» · Громкость 60'])
    state = run([voice('idle')], state) // продолжения не было — снова виден итог
    expect(state.mode).toBe('idle')
    expect(overlayLines(state)).toEqual(['Громкость 60', '«громче»'])
    state = run([voice('listening'), voice('capturing'), partial('открой')], state) // снова «Джарвис»
    expect(overlayLines(state)).toEqual(['Открой', 'Слушаю…'])
    state = run([voice('recognizing')], state)
    expect(overlayLines(state)).toEqual(['Открой', 'Распознаю…'])
    state = run([voice('idle')], state) // Whisper ничего не услышал (шум) — снова видна прошлая команда
    expect(overlayLines(state)).toEqual(['Громкость 60', '«громче»'])
    state = run([voice('listening'), voice('recognizing'), { type: 'recognized', payload: { text: 'открой ютуб' } }], state)
    expect(state.last).toBeNull() // новая команда — прошлая уступила место
    expect(overlayLines(run([voice('listening'), voice('idle')]))).toEqual(['Ничего не услышал', ''])

    const working = run([{ type: 'recognized', payload: { text: 'громче' } }])
    expect(run([voice('capturing')], working)).toBe(working)
  })
})

describe('orbCard', () => {
  it('слушает: курсор ждёт слов, слова появляются по ходу речи', () => {
    let state = run([voice('listening')])
    expect(orbCard(state)).toEqual({ badge: 'listening', status: 'Слушаю…', text: '', live: true, line: null })
    state = run([voice('capturing'), partial('сделай громче')], state)
    expect(orbCard(state)).toEqual({ badge: 'listening', status: 'Слушаю…', text: 'Сделай громче', live: true, line: null })
    state = run([voice('recognizing')], state)
    expect(orbCard(state)).toEqual({ badge: 'recognizing', status: 'Распознаю…', text: 'Сделай громче', live: false, line: null })
  })

  it('команда: фраза, ход и итог, как «Сделай громче ✓ Громкость увеличена»', () => {
    let state = run([{ type: 'recognized', payload: { text: 'Сделай громче.' } }])
    expect(orbCard(state)).toEqual({ badge: 'working', status: 'Выполняю…', text: 'Сделай громче', live: false, line: null })
    state = run([{ type: 'step', payload: { number: 1, total: 2, action: 'volume_up', text: 'Громкость 60%', ok: true } }], state)
    expect(orbCard(state).line).toEqual({ tone: 'muted', text: 'Шаг 1 из 2 · Громкость 60%' })
    state = run([result({ reply: 'Громкость увеличена' })], state)
    expect(orbCard(state)).toEqual({
      badge: 'done',
      status: 'Готово',
      text: 'Сделай громче',
      live: false,
      line: { tone: 'ok', text: 'Громкость увеличена' },
    })
  })

  it('без ответа итог — последний шаг, а «Готово» не повторяется', () => {
    const step: AssistantEvent = { type: 'step', payload: { number: 1, total: 1, action: 'volume_up', text: 'Громкость 60%', ok: true } }
    expect(orbCard(run([{ type: 'recognized', payload: { text: 'громче' } }, step, result({})])).line).toEqual({ tone: 'ok', text: 'Громкость 60%' })
    expect(orbCard(run([{ type: 'recognized', payload: { text: 'громче' } }, result({})])).line).toBeNull()
    expect(orbCard(run([{ type: 'recognized', payload: { text: 'громче' } }, { type: 'stopped', payload: {} }]))).toEqual({
      badge: 'stopped',
      status: 'Остановлено',
      text: 'Громче',
      live: false,
      line: null,
    })
  })

  it('итог видно и пока слушает продолжение, и после, пока не прозвучала новая фраза', () => {
    let state = run([{ type: 'recognized', payload: { text: 'открой ютуб' } }, result({ reply: 'YouTube открыт' }), voice('listening')])
    expect(orbCard(state)).toEqual({ badge: 'listening', status: 'Слушаю…', text: 'Открой ютуб', live: false, line: { tone: 'ok', text: 'YouTube открыт' } })
    state = run([voice('idle')], state)
    expect(orbCard(state)).toEqual({ badge: 'done', status: 'Готово', text: 'Открой ютуб', live: false, line: { tone: 'ok', text: 'YouTube открыт' } })
    state = run([voice('listening'), voice('capturing'), partial('который')], state)
    expect(orbCard(state)).toEqual({ badge: 'listening', status: 'Слушаю…', text: 'Который', live: true, line: null })
    state = run([partial('')], state) // это был шум — снова видна прошлая команда
    expect(orbCard(state).text).toBe('Открой ютуб')
  })

  it('новый показ оверлея начинается без прошлых итогов', () => {
    const state = run([{ type: 'recognized', payload: { text: 'сделай кофе' } }, result({ ok: false, reply: 'Не знаю такой команды' }), voice('listening')])
    expect(orbCard(state).line).toEqual({ tone: 'error', text: 'Не знаю такой команды' })
    const fresh = run([{ type: 'overlay', payload: { visible: true } }], state)
    expect(orbCard(fresh)).toEqual({ badge: 'listening', status: 'Слушаю…', text: '', live: true, line: null })
    expect(run([{ type: 'overlay', payload: { visible: false } }], state)).toBe(state) // уходя, ничего не меняет
  })

  it('не дождался фразы — «Ничего не услышал»', () => {
    expect(orbCard(run([voice('listening'), voice('idle')]))).toEqual({
      badge: 'idle',
      status: 'Ничего не услышал',
      text: '',
      live: false,
      line: null,
    })
  })

  it('подтверждение: вопрос крупно', () => {
    const state = run([
      { type: 'recognized', payload: { text: 'выключи компьютер' } },
      { type: 'confirm', payload: { id: 'c1', question: 'Выключить компьютер?' } },
    ])
    expect(orbCard(state)).toEqual({ badge: 'confirm', status: 'Подтвердите', text: 'Выключить компьютер?', live: false, line: null })
  })
})

describe('phrase и tail', () => {
  it('фраза с заглавной и без точки в конце', () => {
    expect(phrase('сделай громче')).toBe('Сделай громче')
    expect(phrase(' Который час? ')).toBe('Который час?')
    expect(phrase('Открой ютуб.')).toBe('Открой ютуб')
    expect(phrase('Подожди...')).toBe('Подожди...')
    expect(phrase('')).toBe('')
  })

  it('у длинной фразы видны последние слова', () => {
    expect(tail('сделай громче')).toBe('сделай громче')
    const long = 'открой ютуб и включи там какую-нибудь спокойную музыку'
    expect(tail(long)).toBe('…там какую-нибудь спокойную музыку')
    expect(tail('скажи пожалуйста который сейчас час в Нью-Йорке', 20)).toBe('…час в Нью-Йорке')
    expect(tail(long).length).toBeLessThanOrEqual(35)
  })
})
