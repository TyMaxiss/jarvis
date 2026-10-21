import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { AssistantEvent, HistoryItem } from '../lib/types'

const { call, listeners } = vi.hoisted(() => ({
  call: vi.fn(),
  listeners: [] as Array<(event: AssistantEvent) => void>,
}))
vi.mock('../lib/bridge', () => ({
  call,
  onEvent: (listener: (event: AssistantEvent) => void) => {
    listeners.push(listener)
    return () => listeners.splice(listeners.indexOf(listener), 1)
  },
}))

import { dayLabel, formatArgs, groupByDay, HistoryPage } from './History'

function item(fields: Partial<HistoryItem> & Pick<HistoryItem, 'id' | 'phrase'>): HistoryItem {
  return {
    time: new Date().toISOString(),
    source: 'voice',
    route: 'local',
    commands: [],
    reply: '',
    ok: true,
    error: '',
    duration_ms: 420,
    actions: [],
    ...fields,
  }
}

const agentTask = item({
  id: 'a1',
  phrase: 'создай папку Проекты и открой её',
  route: 'llm',
  reply: 'Готово',
  actions: [
    {
      action: 'create_folder',
      args: { path: 'Проекты' },
      status: 'done',
      result: 'Папка создана',
      duration_ms: 30,
    },
    {
      action: 'open_folder',
      args: { path: 'Проекты' },
      status: 'failed',
      result: 'Нет доступа',
      duration_ms: 5,
    },
  ],
})
const failed = item({
  id: 'f1',
  phrase: 'открой телеграм',
  ok: false,
  error: 'Не нашёл приложение «телеграм»',
})

async function renderPage(onSaveAsCommand = vi.fn()) {
  await act(async () => {
    render(<HistoryPage onSaveAsCommand={onSaveAsCommand} />)
  })
  return onSaveAsCommand
}

beforeEach(() => {
  call.mockReset()
  call.mockResolvedValue({ items: [agentTask, failed], more: false })
})
afterEach(cleanup)

describe('HistoryPage', () => {
  it('показывает фразы с итогом, а по щелчку — действия и кнопки', async () => {
    const onSave = await renderPage()
    expect(call).toHaveBeenCalledWith('history', {
      filter: 'all',
      search: '',
      limit: 50,
      before: undefined,
      before_time: undefined,
    })
    expect(screen.getByText('Сегодня')).toBeTruthy()
    expect(screen.getByText('Не нашёл приложение «телеграм»')).toBeTruthy()
    fireEvent.click(screen.getByText('создай папку Проекты и открой её'))
    expect(screen.getByText('create_folder(path="Проекты")')).toBeTruthy()
    expect(screen.getByLabelText('не получилось')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Повторить' }))
    expect(call).toHaveBeenCalledWith('send_text', 'создай папку Проекты и открой её')
    fireEvent.click(screen.getByRole('button', { name: 'Сохранить как команду' }))
    expect(onSave).toHaveBeenCalledWith(agentTask)
  })

  it('задачу без выполненных действий в команду не сохранить', async () => {
    call.mockResolvedValue({
      items: [
        item({
          id: 'x',
          phrase: 'расскажи анекдот',
          route: 'llm',
          reply: 'Шутка',
        }),
      ],
      more: false,
    })
    await renderPage()
    fireEvent.click(screen.getByText('расскажи анекдот'))
    expect(screen.queryByRole('button', { name: 'Сохранить как команду' })).toBeNull()
  })

  it('фильтр и «Показать ещё» запрашивают нужную страницу', async () => {
    call.mockResolvedValue({ items: [agentTask, failed], more: true })
    await renderPage()
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Показать ещё' }))
    })
    expect(call).toHaveBeenLastCalledWith('history', {
      filter: 'all',
      search: '',
      limit: 50,
      before: 'f1',
      before_time: failed.time,
    })
    expect(screen.getAllByText('открой телеграм')).toHaveLength(2) // вторая страница добавилась к первой
    await act(async () => {
      fireEvent.click(screen.getByRole('radio', { name: 'Ошибки' }))
    })
    expect(call).toHaveBeenLastCalledWith('history', expect.objectContaining({ filter: 'errors', before: undefined }))
  })

  it('новая выполненная фраза перечитывает историю', async () => {
    await renderPage()
    call.mockClear()
    await act(async () => {
      for (const listener of listeners)
        listener({
          type: 'result',
          payload: {
            id: 1,
            time: '12:00',
            phrase: 'x',
            route: 'local',
            reply: '',
            ok: true,
            ms: 1,
            source: 'voice',
          },
        })
    })
    expect(call).toHaveBeenCalledWith('history', expect.objectContaining({ filter: 'all' }))
  })

  it('пустая история объясняет, что здесь будет', async () => {
    call.mockResolvedValue({ items: [], more: false })
    await renderPage()
    expect(screen.getByText(/Здесь появятся ваши команды/)).toBeTruthy()
  })
})

describe('даты и аргументы', () => {
  it('подписывает дни', () => {
    const now = new Date(2026, 9, 21, 12, 0)
    expect(dayLabel(new Date(2026, 9, 21, 0, 5), now)).toBe('Сегодня')
    expect(dayLabel(new Date(2026, 9, 20, 23, 59), now)).toBe('Вчера')
    expect(dayLabel(new Date(2026, 9, 3), now)).toBe('3 октября')
    expect(dayLabel(new Date(2025, 11, 31), now)).toBe('31 декабря 2025 г.')
    const groups = groupByDay([
      item({ id: '1', phrase: 'a' }),
      item({ id: '2', phrase: 'b', time: '2020-01-01T10:00:00+03:00' }),
    ])
    expect(groups.map(([day, items]) => [day, items.length])).toEqual([
      ['Сегодня', 1],
      ['1 января 2020 г.', 1],
    ])
  })

  it('коротко показывает аргументы', () => {
    expect(formatArgs({ name: 'Telegram', percent: 30, empty: '', missing: null })).toBe('name="Telegram", percent=30')
    expect(formatArgs({ path: 'C:\\Users\\Проекты', flags: ['a'] })).toBe('path="C:\\Users\\Проекты", flags=["a"]')
  })
})
