// Редактор команд в ?mock: паки в памяти, проверка фразы — простым сравнением с фразами.

import { placeholders } from './commands'
import { mockActions } from './mockActions'
import type { AssistantApi, CommandForm, CommandsTree, PhraseTest } from './types'

type CommandsApi = Pick<
  AssistantApi,
  'commands_tree' | 'actions' | 'command_get' | 'command_save' | 'command_delete' | 'command_test' | 'command_run'
>

const USER_FILE = 'my.yaml'
const FILES = [USER_FILE, 'browser.yaml', 'media.yaml', 'windows.yaml']

function command(id: string, phrases: string[], steps: CommandForm['steps'], extra: Partial<CommandForm> = {}): CommandForm {
  return {
    id,
    phrases,
    slots: {},
    when: { active_window_contains: [], active_process: [] },
    confirm: false,
    reply: '',
    hotkey: '',
    steps,
    ...extra,
  }
}

export function createMockCommands(run: (phrase: string) => void): CommandsApi {
  const packs: Record<string, CommandForm[]> = {
    [USER_FILE]: [
      command(
        'work_mode',
        ['рабочий режим', 'начинаем работу'],
        [
          { action: 'open_app', args: { name: 'Telegram' } },
          { action: 'open_app', args: { name: 'Visual Studio Code' } },
          { action: 'volume_set', args: { percent: 30 } },
        ],
        { reply: 'Все системы готовы', hotkey: 'ctrl+alt+1' },
      ),
    ],
    'browser.yaml': [
      command('open_youtube', ['открой ютуб', 'запусти ютуб'], [{ action: 'open_url', args: { url: 'https://www.youtube.com' } }]),
      command('web_search', ['найди {query}', 'загугли {query}'], [{ action: 'web_search', args: { query: '{query}' } }], {
        slots: { query: 'text' },
      }),
    ],
    'media.yaml': [
      command('volume_up', ['сделай громче', 'громче'], [{ action: 'volume_change', args: { delta: 10 } }]),
      command('volume_set', ['громкость на {percent}', 'громкость {percent}'], [{ action: 'volume_set', args: { percent: '{percent}' } }], {
        slots: { percent: 'int' },
        reply: 'Громкость {percent}',
      }),
      command('youtube_pause', ['пауза', 'стоп'], [{ action: 'hotkey', args: { keys: 'k' } }], {
        when: { active_window_contains: ['YouTube'], active_process: [] },
      }),
    ],
    'windows.yaml': [
      command('minimize_all', ['сверни все окна', 'покажи рабочий стол'], [{ action: 'hotkey', args: { keys: 'win+d' } }]),
      command('shutdown', ['выключи компьютер'], [{ action: 'shutdown', args: {} }]),
    ],
  }

  const tree = (): CommandsTree => {
    const mine = new Set(packs[USER_FILE].map((c) => c.id))
    return structuredClone({
      user_file: USER_FILE,
      packs: FILES.map((file) => ({
        file,
        user: file === USER_FILE,
        error: '',
        commands: packs[file].map((c, index) => ({
          id: c.id,
          title: c.phrases[0] ?? c.id,
          phrases: c.phrases.length,
          steps: c.steps.length,
          hotkey: c.hotkey,
          line: index * 6 + 1,
          active: file === USER_FILE || !mine.has(c.id),
          replaced_by: file !== USER_FILE && mine.has(c.id) ? USER_FILE : '',
          problem: '',
        })),
      })),
    })
  }

  const validate = (form: CommandForm): string[] => {
    const errors: string[] = []
    if (!form.id.trim()) errors.push('Укажите id команды')
    if (!form.steps.length) errors.push('Добавьте хотя бы одно действие')
    form.steps.forEach((step, index) => {
      if (!mockActions.some((action) => action.name === step.action)) errors.push(`Шаг ${index + 1}: выберите действие`)
    })
    if (!form.phrases.length && !form.hotkey) errors.push('Нет ни фраз, ни горячей клавиши — команду нечем запустить')
    return errors
  }

  const active = (): CommandForm[] => {
    const mine = packs[USER_FILE]
    const ids = new Set(mine.map((c) => c.id))
    return [...mine, ...FILES.slice(1).flatMap((file) => packs[file].filter((c) => !ids.has(c.id)))]
  }

  return {
    async commands_tree() {
      return tree()
    },
    async actions() {
      return structuredClone(mockActions)
    },
    async command_get(file: string, id: string) {
      const found = packs[file]?.find((c) => c.id === id)
      if (!found) return { ok: false, errors: [`В ${file} нет команды «${id}»`] }
      const alsoIn = FILES.filter((other) => other !== file && packs[other].some((c) => c.id === id))
      return { ok: true, errors: [], file, user: file === USER_FILE, command: structuredClone(found), problems: [], also_in: alsoIn }
    },
    async command_save(file: string, originalId: string, form: CommandForm) {
      const errors = validate(form)
      if (errors.length) return { ...tree(), ok: false, errors }
      const mine = packs[USER_FILE]
      const index = file === USER_FILE ? mine.findIndex((c) => c.id === originalId) : -1
      const clash = mine.findIndex((c) => c.id === form.id)
      if (clash >= 0 && clash !== index) {
        return { ...tree(), ok: false, errors: [`В ${USER_FILE} уже есть команда «${form.id}» — выберите другой id`] }
      }
      if (index >= 0) mine[index] = structuredClone(form)
      else mine.push(structuredClone(form))
      return { ...tree(), ok: true, errors: [], id: form.id }
    },
    async command_delete(file: string, id: string) {
      packs[file] = packs[file].filter((c) => c.id !== id)
      return { ...tree(), ok: true, errors: [] }
    },
    async command_test(text: string, form: CommandForm | null, originalId: string) {
      const errors = form ? validate(form) : []
      const draft = form && !errors.length ? form : null
      const commands = draft ? [...active().filter((c) => c.id !== draft.id && c.id !== originalId), draft] : active()
      const said = text.trim().toLowerCase()
      const found: PhraseTest['matches'] = []
      for (const candidate of commands) {
        for (const phrase of candidate.phrases) {
          const names = placeholders([phrase])
          const pattern = new RegExp(`^${phrase.toLowerCase().replace(/\{\w+\}/g, '(.+)')}$`)
          const match = said.match(pattern)
          if (!match) continue
          const slots = Object.fromEntries(names.map((name, i) => [name, candidate.slots[name] === 'int' ? Number(match[i + 1]) || match[i + 1] : match[i + 1]]))
          found.push({ id: candidate.id, score: 100, phrase, text, slots, draft: candidate === draft })
          break
        }
      }
      const result: PhraseTest = {
        status: found.length === 1 ? 'matched' : found.length > 1 ? 'ambiguous' : said ? 'not_found' : 'empty',
        reason: '',
        normalized: said,
        matches: found.length === 1 ? found : [],
        candidates: found.map(({ id, score, phrase }) => ({ id, score, phrase })),
        errors,
      }
      return result
    },
    async command_run(form: CommandForm, phrase: string) {
      const errors = validate(form)
      if (!errors.length) run(phrase.trim() || form.phrases[0] || form.id)
      return { ok: !errors.length, errors }
    },
  }
}
