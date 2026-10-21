import { describe, expect, it } from 'vitest'
import { keyLabel } from '../components/ui'
import { formatMs, isListening, plural, voiceLabel } from './format'
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
