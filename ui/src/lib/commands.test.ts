import { describe, expect, it } from 'vitest'
import { draftFromHistory, emptyForm, fieldKind, fieldValue, isDirty, placeholders, rangeHint, slugify, uniqueId } from './commands'
import type { HistoryItem } from './types'

describe('команды', () => {
  it('находит слоты во фразах', () => {
    expect(placeholders(['громкость на {percent}', 'найди {query} на {site}', 'громче'])).toEqual(['percent', 'query', 'site'])
  })

  it('делает id из фразы и не повторяет занятые', () => {
    expect(slugify('Рабочий режим')).toBe('rabochiy_rezhim')
    expect(slugify('громкость на {percent}!')).toBe('gromkost_na_percent')
    expect(slugify('щука и ёж')).toBe('shchuka_i_ezh')
    expect(uniqueId('tea', new Set(['tea', 'tea_2']))).toBe('tea_3')
    expect(uniqueId('tea', new Set())).toBe('tea')
  })

  it('поля по схеме и значения из них', () => {
    expect(fieldKind({ type: 'string' })).toBe('text')
    expect(fieldKind({ type: 'string', maxLength: 5000 })).toBe('longtext')
    expect(fieldKind({ type: 'string', enum: ['a', 'b'] })).toBe('enum')
    expect(fieldKind({ type: 'integer' })).toBe('integer')
    expect(fieldValue('30', 'integer')).toBe(30)
    expect(fieldValue('2,5', 'number')).toBe(2.5)
    expect(fieldValue('{percent}', 'integer')).toBe('{percent}')
    expect(fieldValue(' Привет ', 'text')).toBe(' Привет ')
    expect(fieldValue('', 'integer')).toBe('')
    expect(rangeHint({ minimum: 0, maximum: 100 })).toBe('от 0 до 100')
    expect(rangeHint({ exclusiveMinimum: 0, maximum: 60 })).toBe('от 0 до 60')
    expect(rangeHint({ exclusiveMinimum: 0 })).toBe('больше 0')
  })

  it('не считает изменением пустые поля и тип слота по умолчанию', () => {
    const saved = { ...emptyForm(), id: 'a', phrases: ['x {n}'], steps: [{ action: 'say', args: { text: 'hi' } }] }
    expect(isDirty({ ...saved, slots: { n: 'str' }, steps: [{ action: 'say', args: { text: 'hi', extra: '' } }] }, saved)).toBe(false)
    expect(isDirty({ ...saved, slots: { n: 'int' } }, saved)).toBe(true)
  })

  it('черновик из задачи нейросети берёт только выполненные шаги', () => {
    const item: HistoryItem = {
      id: 'r1',
      time: '2026-10-21T10:00:00+03:00',
      phrase: 'создай папку Проекты',
      source: 'voice',
      route: 'llm',
      commands: [],
      reply: 'Готово',
      ok: true,
      error: '',
      duration_ms: 10,
      actions: [
        { action: 'create_folder', args: { path: 'Проекты' }, status: 'done', result: '', duration_ms: 1 },
        { action: 'open_folder', args: { path: 'Проекты' }, status: 'failed', result: '', duration_ms: 1 },
      ],
    }
    const draft = draftFromHistory(item)
    expect(draft.id).toBe('sozday_papku_proekty')
    expect(draft.phrases).toEqual(['создай папку Проекты'])
    expect(draft.steps).toEqual([{ action: 'create_folder', args: { path: 'Проекты' } }])
  })
})
