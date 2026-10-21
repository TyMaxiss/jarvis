import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createMockCommands } from '../lib/mockCommands'
import type { CommandForm } from '../lib/types'

const { call } = vi.hoisted(() => ({ call: vi.fn() }))
vi.mock('../lib/bridge', () => ({ call, onEvent: () => () => {} }))

import { emptyForm } from '../lib/commands'
import { CommandsPage } from './Commands'

let api: ReturnType<typeof createMockCommands>
const ran: string[] = []

beforeEach(() => {
  ran.length = 0
  api = createMockCommands((phrase) => ran.push(phrase))
  call.mockReset()
  call.mockImplementation((method: keyof typeof api, ...args: unknown[]) =>
    (api[method] as (...a: unknown[]) => Promise<unknown>)(...args),
  )
  vi.spyOn(window, 'confirm').mockReturnValue(true)
})
afterEach(cleanup)

async function renderPage(draft: CommandForm | null = null) {
  await act(async () => {
    render(<CommandsPage draft={draft} onDraftUsed={() => {}} />)
  })
}

async function click(element: HTMLElement) {
  await act(async () => {
    fireEvent.click(element)
  })
}

function savedCall() {
  return call.mock.calls.find(([method]) => method === 'command_save')
}

describe('CommandsPage', () => {
  it('показывает паки и правит свою команду', async () => {
    await renderPage()
    const tree = screen.getByRole('navigation', { name: 'Паки команд' })
    expect(within(tree).getByText('Мои команды · my.yaml')).toBeTruthy()
    await click(within(tree).getByRole('button', { name: /рабочий режим/ }))
    expect(screen.getByRole('heading', { name: 'рабочий режим' })).toBeTruthy()
    const save = screen.getByRole('button', { name: 'Сохранить' }) as HTMLButtonElement
    expect(save.disabled).toBe(true) // ничего не меняли

    const phrase = screen.getByLabelText('Новая фраза')
    fireEvent.change(phrase, { target: { value: 'поехали работать' } })
    fireEvent.keyDown(phrase, { key: 'Enter' })
    expect(screen.getByText('не сохранено')).toBeTruthy()
    await click(save)
    const [, file, originalId, form] = savedCall()!
    expect([file, originalId]).toEqual(['my.yaml', 'work_mode'])
    expect(form.phrases).toEqual(['рабочий режим', 'начинаем работу', 'поехали работать'])
    expect(screen.queryByText('не сохранено')).toBeNull()
  })

  it('встроенная команда сохраняется копией, слот подставляется в шаг', async () => {
    await renderPage()
    await click(screen.getByRole('button', { name: /громкость на \{percent\}/ }))
    expect(screen.getByText(/Встроенные паки не меняются/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Удалить' })).toBeNull()
    expect((screen.getByLabelText('Тип слота percent') as HTMLSelectElement).value).toBe('int')
    expect((screen.getByLabelText('percent') as HTMLInputElement).value).toBe('{percent}')

    fireEvent.change(screen.getByLabelText('Проверочная фраза'), { target: { value: 'громкость на 40' } })
    await click(screen.getByRole('button', { name: 'Проверить' }))
    expect(screen.getByLabelText('Результат проверки').textContent).toBe('Сработает эта команда (оценка 100; percent = 40)')
    await click(screen.getByRole('button', { name: 'Выполнить' }))
    expect(ran).toEqual(['громкость на 40'])
  })

  it('новая команда: id из фразы, шаг с полями по схеме', async () => {
    await renderPage()
    await click(screen.getAllByRole('button', { name: 'Новая команда' })[0])
    const phrase = screen.getByLabelText('Новая фраза')
    fireEvent.change(phrase, { target: { value: 'ночной режим' } })
    fireEvent.keyDown(phrase, { key: 'Enter' })
    expect((screen.getByLabelText('id команды') as HTMLInputElement).value).toBe('nochnoy_rezhim')

    await click(screen.getByRole('button', { name: 'Действие' }))
    fireEvent.change(screen.getByLabelText('Действие'), { target: { value: 'volume_set' } })
    fireEvent.change(screen.getByLabelText('percent'), { target: { value: '20' } })
    await click(screen.getByRole('button', { name: 'Сохранить' }))
    const [, , , form] = savedCall()!
    expect(form).toMatchObject({ id: 'nochnoy_rezhim', phrases: ['ночной режим'], steps: [{ action: 'volume_set', args: { percent: 20 } }] })
    // После сохранения открыта сохранённая команда из своего пака.
    expect(screen.getByText('nochnoy_rezhim')).toBeTruthy()
    expect(within(screen.getByRole('navigation', { name: 'Паки команд' })).getByRole('button', { name: /ночной режим/ })).toBeTruthy()
  })

  it('ошибки сохранения видны над формой', async () => {
    await renderPage()
    await click(screen.getAllByRole('button', { name: 'Новая команда' })[0])
    await click(screen.getByRole('button', { name: 'Сохранить' }))
    expect(screen.getByText('Укажите id команды')).toBeTruthy()
    expect(screen.getByText('Добавьте хотя бы одно действие')).toBeTruthy()
  })

  it('черновик из Истории открывается новой командой', async () => {
    const draft = { ...emptyForm(), id: 'sozday_papku', phrases: ['создай папку'], steps: [{ action: 'say', args: { text: 'Готово' } }] }
    await renderPage(draft)
    expect(screen.getByRole('heading', { name: 'создай папку' })).toBeTruthy()
    expect(screen.getByText('новая')).toBeTruthy()
    expect((screen.getByLabelText('text') as HTMLInputElement).value).toBe('Готово')
  })

  it('удаление своей команды', async () => {
    await renderPage()
    await click(screen.getByRole('button', { name: /рабочий режим/ }))
    await click(screen.getByRole('button', { name: 'Удалить' }))
    expect(call).toHaveBeenCalledWith('command_delete', 'my.yaml', 'work_mode')
    expect(screen.getByText(/Пока пусто/)).toBeTruthy()
  })
})

