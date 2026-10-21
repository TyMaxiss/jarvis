// Запись сочетания клавиш в формате библиотеки keyboard: «ctrl+alt+space».
// Клавиша берётся по физическому коду (event.code), поэтому раскладка не важна.

const modifierCodes: Record<string, string> = {
  ControlLeft: 'ctrl',
  ControlRight: 'ctrl',
  AltLeft: 'alt',
  AltRight: 'alt',
  ShiftLeft: 'shift',
  ShiftRight: 'shift',
  MetaLeft: 'windows',
  MetaRight: 'windows',
}

const namedCodes: Record<string, string> = {
  Space: 'space',
  Enter: 'enter',
  Tab: 'tab',
  Backspace: 'backspace',
  Insert: 'insert',
  Delete: 'delete',
  Home: 'home',
  End: 'end',
  PageUp: 'page up',
  PageDown: 'page down',
  ArrowUp: 'up',
  ArrowDown: 'down',
  ArrowLeft: 'left',
  ArrowRight: 'right',
  Pause: 'pause',
  ScrollLock: 'scroll lock',
}

const modifierOrder = ['ctrl', 'alt', 'shift', 'windows']

export type HotkeyResult =
  | { kind: 'pending'; modifiers: string[] } // зажаты только модификаторы
  | { kind: 'done'; combo: string }
  | { kind: 'cancel' }
  | { kind: 'error'; message: string }

type KeyEventLike = Pick<KeyboardEvent, 'code' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'metaKey'>

/** Какие модификаторы сейчас зажаты, в порядке «ctrl, alt, shift, windows». */
export function heldModifiers(event: Omit<KeyEventLike, 'code'>): string[] {
  return modifierOrder.filter(
    (name) =>
      (name === 'ctrl' && event.ctrlKey) ||
      (name === 'alt' && event.altKey) ||
      (name === 'shift' && event.shiftKey) ||
      (name === 'windows' && event.metaKey),
  )
}

function mainKey(code: string): string | null {
  if (/^Key[A-Z]$/.test(code)) return code.slice(3).toLowerCase()
  if (/^Digit[0-9]$/.test(code)) return code.slice(5)
  if (/^F([1-9]|1[0-9]|2[0-4])$/.test(code)) return code.toLowerCase()
  return namedCodes[code] ?? null
}

/** Разбирает нажатие: готовое сочетание, ожидание основной клавиши, отмена или ошибка. */
export function comboFromEvent(event: KeyEventLike): HotkeyResult {
  const modifiers = heldModifiers(event)
  if (event.code in modifierCodes) {
    const held = new Set([...modifiers, modifierCodes[event.code]])
    return { kind: 'pending', modifiers: modifierOrder.filter((name) => held.has(name)) }
  }
  if (event.code === 'Escape' && modifiers.length === 0) return { kind: 'cancel' }
  const key = mainKey(event.code)
  if (!key) return { kind: 'error', message: 'Эта клавиша не подходит: выберите букву, цифру или F1–F24' }
  const functionKey = /^f\d+$/.test(key) || key === 'pause' || key === 'scroll lock'
  if (!functionKey && !modifiers.some((name) => name !== 'shift')) {
    return { kind: 'error', message: 'Добавьте Ctrl, Alt или Win, иначе сочетание будет мешать печатать' }
  }
  return { kind: 'done', combo: [...modifiers, key].join('+') }
}
