// Логика редактора команд без React: слоты во фразах, id из фразы, поля по JSON-схеме.

import type { CommandForm, HistoryItem, JsonSchema, SlotType } from './types'

export const SLOT_TYPES: Array<{ value: SlotType; title: string }> = [
  { value: 'str', title: 'слово' },
  { value: 'text', title: 'текст' },
  { value: 'int', title: 'число' },
  { value: 'float', title: 'дробное' },
]

const PLACEHOLDER = /\{(\w+)\}/g

/** Слоты, которые встречаются во фразах: «громкость на {percent}» → percent. */
export function placeholders(phrases: string[]): string[] {
  const names = new Set<string>()
  for (const phrase of phrases) for (const match of phrase.matchAll(PLACEHOLDER)) names.add(match[1])
  return [...names]
}

export function emptyForm(): CommandForm {
  return {
    id: '',
    phrases: [],
    slots: {},
    when: { active_window_contains: [], active_process: [] },
    confirm: false,
    reply: '',
    hotkey: '',
    steps: [],
  }
}

/** Черновик команды из задачи нейросети: её фраза и шаги, которые выполнились. */
export function draftFromHistory(item: HistoryItem): CommandForm {
  return {
    ...emptyForm(),
    id: slugify(item.phrase),
    phrases: item.phrase ? [item.phrase] : [],
    steps: item.actions
      .filter((action) => action.status === 'done' || action.status === 'dry_run')
      .map((action) => ({ action: action.action, args: structuredClone(action.args) })),
  }
}

/** Изменилась ли команда относительно сохранённой. */
export function isDirty(form: CommandForm, saved: CommandForm): boolean {
  return JSON.stringify(normalized(form)) !== JSON.stringify(normalized(saved))
}

function normalized(form: CommandForm): CommandForm {
  const used = placeholders(form.phrases)
  return {
    ...form,
    slots: Object.fromEntries(used.map((name) => [name, form.slots[name] ?? 'str'])),
    steps: form.steps.map((step) => ({
      action: step.action,
      args: Object.fromEntries(Object.entries(step.args).filter(([, value]) => value !== '' && value !== null && value !== undefined)),
    })),
  }
}

const TRANSLIT: Record<string, string> = {
  а: 'a', б: 'b', в: 'v', г: 'g', д: 'd', е: 'e', ё: 'e', ж: 'zh', з: 'z', и: 'i', й: 'y', к: 'k', л: 'l', м: 'm',
  н: 'n', о: 'o', п: 'p', р: 'r', с: 's', т: 't', у: 'u', ф: 'f', х: 'kh', ц: 'ts', ч: 'ch', ш: 'sh', щ: 'shch',
  ъ: '', ы: 'y', ь: '', э: 'e', ю: 'yu', я: 'ya',
}

/** id для новой команды из её первой фразы: «рабочий режим» → rabochiy_rezhim. */
export function slugify(phrase: string): string {
  const latin = [...phrase.toLowerCase().replace(PLACEHOLDER, '$1')].map((char) => TRANSLIT[char] ?? char).join('')
  return latin.replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 40).replace(/_+$/, '')
}

/** Свободный id: если такой уже есть — с номером (tea, tea_2…). */
export function uniqueId(base: string, taken: Set<string>): string {
  if (!base) return base
  if (!taken.has(base)) return base
  let number = 2
  while (taken.has(`${base}_${number}`)) number++
  return `${base}_${number}`
}

export type FieldKind = 'text' | 'longtext' | 'integer' | 'number' | 'boolean' | 'enum' | 'json'

/** Как показывать параметр действия в форме шага. */
export function fieldKind(schema: JsonSchema): FieldKind {
  if (schema.enum) return 'enum'
  switch (schema.type) {
    case 'string':
      return (schema.maxLength ?? 0) > 200 ? 'longtext' : 'text'
    case 'integer':
      return 'integer'
    case 'number':
      return 'number'
    case 'boolean':
      return 'boolean'
    default:
      return 'json'
  }
}

/** Значение параметра → текст поля ввода. */
export function fieldText(value: unknown): string {
  if (value === undefined || value === null) return ''
  return typeof value === 'string' ? value : JSON.stringify(value)
}

/** Текст поля → значение параметра: числа числами, слоты «{x}» и всё прочее строкой. */
export function fieldValue(text: string, kind: FieldKind): unknown {
  const value = text.trim()
  if (value === '') return ''
  if ((kind === 'integer' || kind === 'number') && /^-?\d+([.,]\d+)?$/.test(value)) return Number(value.replace(',', '.'))
  if (kind === 'json') {
    try {
      return JSON.parse(value)
    } catch {
      return value
    }
  }
  return kind === 'text' || kind === 'longtext' ? text : value
}

/** Подсказка о допустимых значениях числа: «от 0 до 100». */
export function rangeHint(schema: JsonSchema): string {
  const low = schema.minimum ?? schema.exclusiveMinimum
  const high = schema.maximum
  if (low !== undefined && high !== undefined) return `от ${low} до ${high}`
  if (low !== undefined) return schema.exclusiveMinimum !== undefined ? `больше ${low}` : `от ${low}`
  if (high !== undefined) return `до ${high}`
  return ''
}
