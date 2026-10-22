import { describe, expect, it } from 'vitest'
import type { AssistantEvent } from '../lib/types'
import { chatReduce, type Exchange } from './QuickChat'

const running: Exchange = { id: 1, text: 'открой блокнот', steps: [], reply: '', status: 'running', ms: null }

describe('chatReduce', () => {
  it('шаги, ответ и итог относятся к выполняющейся команде', () => {
    const events: AssistantEvent[] = [
      { type: 'step', payload: { number: 1, total: null, action: 'open_app', text: 'Открываю Блокнот', ok: true } },
      { type: 'reply', payload: { text: 'Открываю' } },
      { type: 'reply', payload: { text: 'Открываю Блокнот.' } },
      {
        type: 'result',
        payload: { id: 5, time: '12:00', phrase: 'открой блокнот', route: 'llm', reply: '', ok: true, ms: 1200, source: 'text' },
      },
    ]
    const [exchange] = events.reduce(chatReduce, [running])
    expect(exchange.steps).toHaveLength(1)
    expect(exchange.reply).toBe('Открываю Блокнот.')
    expect(exchange.status).toBe('ok')
    expect(exchange.ms).toBe(1200)
  })

  it('остановленная команда — «Остановлено», итог прерванной задачи её не меняет', () => {
    let list = chatReduce([running], { type: 'stopped', payload: {} })
    expect(list[0].status).toBe('stopped')
    list = chatReduce(list, {
      type: 'result',
      payload: { id: 6, time: '12:00', phrase: 'открой блокнот', route: 'llm', reply: '', ok: false, ms: 900, source: 'text' },
    })
    expect(list[0].status).toBe('stopped')
  })

  it('повтор шага с тем же номером заменяет его, а законченную команду события не меняют', () => {
    const step = { number: 1, total: null, action: 'open_app', text: 'Открываю', ok: true }
    let list = chatReduce([running], { type: 'step', payload: step })
    list = chatReduce(list, { type: 'step', payload: { ...step, ok: false } })
    expect(list[0].steps).toEqual([{ ...step, ok: false }])
    const done: Exchange[] = [{ ...running, status: 'ok' }]
    expect(chatReduce(done, { type: 'reply', payload: { text: 'чужой ответ' } })).toBe(done)
  })
})
