import { describe, expect, it } from 'vitest'
import { keyLabel } from '../components/ui'
import { countdown, durationText, formatMs, isListening, momentText, plural, voiceLabel } from './format'
import { createMockApi } from './mock'

describe('formatMs', () => {
  it('миллисекунды до секунды, дальше — секунды с запятой', () => {
    expect(formatMs(412.4)).toBe('412 мс')
    expect(formatMs(2310)).toBe('2,3 с')
  })
})

describe('voiceLabel', () => {
  it('подписи состояний голоса', async () => {
    const state = await createMockApi(() => {}).get_state()
    expect(voiceLabel(state)).toBe('Жду «Джарвис»')
    expect(voiceLabel({ ...state, voice: { ...state.voice, paused: true } })).toBe('Прослушивание на паузе')
    expect(voiceLabel({ ...state, voice: { ...state.voice, enabled: false, problem: 'нет микрофона' } })).toBe('Голос недоступен')
    expect(voiceLabel({ ...state, voice: { ...state.voice, enabled: false, loading: true } })).toBe('Запускаю голос…')
    // Задача из текста или ответ ещё звучит — и с голосом, и без него.
    expect(voiceLabel({ ...state, busy: true })).toBe('Отвечаю…')
    expect(voiceLabel({ ...state, busy: true, voice: { ...state.voice, enabled: false } })).toBe('Отвечаю…')
    expect(voiceLabel({ ...state, busy: true, voice: { ...state.voice, state: 'listening' } })).toBe('Слушаю…')
    expect(isListening('holding')).toBe(true)
    expect(isListening('processing')).toBe(false)
  })
})

describe('plural', () => {
  it('русские формы слова для чисел', () => {
    const forms: [string, string, string] = ['шаг', 'шага', 'шагов']
    expect([1, 2, 4, 5, 11, 12, 21, 22, 25, 111].map((n) => plural(n, forms))).toEqual([
      'шаг', 'шага', 'шага', 'шагов', 'шагов', 'шагов', 'шаг', 'шага', 'шагов', 'шагов',
    ])
  })
})

describe('keyLabel', () => {
  it('названия клавиш keyboard → подписи', () => {
    expect(keyLabel('ctrl')).toBe('Ctrl')
    expect(keyLabel('windows')).toBe('Win')
    expect(keyLabel('space')).toBe('Пробел')
    expect(keyLabel('page up')).toBe('PgUp')
    expect(keyLabel('f5')).toBe('F5')
    expect(keyLabel('q')).toBe('Q')
  })
})

describe('таймеры', () => {
  it('на сколько поставлен — как в Python', () => {
    expect(durationText(60)).toBe('1 минуту')
    expect(durationText(150)).toBe('2 минуты 30 секунд')
    expect(durationText(300)).toBe('5 минут')
    expect(durationText(3600 * 2 + 60 * 21)).toBe('2 часа 21 минуту')
    expect(durationText(45)).toBe('45 секунд')
  })

  it('когда сработает', () => {
    const now = new Date(2026, 9, 21, 17, 0)
    expect(momentText(new Date(2026, 9, 21, 18, 5), now)).toBe('в 18:05')
    expect(momentText(new Date(2026, 9, 22, 9, 0), now)).toBe('завтра в 9:00')
    expect(momentText(new Date(2026, 9, 24, 17, 0), now)).toBe('24.10 в 17:00')
  })

  it('обратный отсчёт', () => {
    expect(countdown(300_000)).toBe('5:00')
    expect(countdown(299_001)).toBe('5:00')
    expect(countdown(59_000)).toBe('0:59')
    expect(countdown(3_845_000)).toBe('1:04:05')
    expect(countdown(-5)).toBe('0:00')
  })
})
