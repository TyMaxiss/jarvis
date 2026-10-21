import { describe, expect, it } from 'vitest'
import { comboFromEvent, heldModifiers } from './hotkey'

const key = (code: string, mods: { ctrl?: boolean; alt?: boolean; shift?: boolean; win?: boolean } = {}) => ({
  code,
  ctrlKey: Boolean(mods.ctrl),
  altKey: Boolean(mods.alt),
  shiftKey: Boolean(mods.shift),
  metaKey: Boolean(mods.win),
})

describe('comboFromEvent', () => {
  it('собирает сочетание в формате keyboard: модификаторы по порядку, затем клавиша', () => {
    expect(comboFromEvent(key('Space', { ctrl: true, alt: true }))).toEqual({ kind: 'done', combo: 'ctrl+alt+space' })
    expect(comboFromEvent(key('KeyQ', { alt: true, ctrl: true }))).toEqual({ kind: 'done', combo: 'ctrl+alt+q' })
    expect(comboFromEvent(key('Digit5', { win: true, shift: true }))).toEqual({ kind: 'done', combo: 'shift+windows+5' })
    expect(comboFromEvent(key('PageUp', { ctrl: true }))).toEqual({ kind: 'done', combo: 'ctrl+page up' })
  })

  it('берёт физическую клавишу: в русской раскладке Ctrl+Й записывается как ctrl+q', () => {
    expect(comboFromEvent(key('KeyQ', { ctrl: true }))).toEqual({ kind: 'done', combo: 'ctrl+q' })
  })

  it('функциональные клавиши можно без модификаторов', () => {
    expect(comboFromEvent(key('F9'))).toEqual({ kind: 'done', combo: 'f9' })
    expect(comboFromEvent(key('F24'))).toEqual({ kind: 'done', combo: 'f24' })
  })

  it('буква без Ctrl, Alt или Win мешала бы печатать', () => {
    expect(comboFromEvent(key('KeyA')).kind).toBe('error')
    expect(comboFromEvent(key('KeyA', { shift: true })).kind).toBe('error')
  })

  it('пока зажаты только модификаторы — ждёт основную клавишу', () => {
    expect(comboFromEvent(key('ControlLeft', { ctrl: true }))).toEqual({ kind: 'pending', modifiers: ['ctrl'] })
    expect(comboFromEvent(key('AltRight', { ctrl: true }))).toEqual({ kind: 'pending', modifiers: ['ctrl', 'alt'] })
  })

  it('Esc без модификаторов отменяет запись', () => {
    expect(comboFromEvent(key('Escape'))).toEqual({ kind: 'cancel' })
  })

  it('неподходящая клавиша — понятная ошибка', () => {
    const result = comboFromEvent(key('Comma', { ctrl: true }))
    expect(result.kind).toBe('error')
  })
})

describe('heldModifiers', () => {
  it('перечисляет зажатые модификаторы в каноническом порядке', () => {
    expect(heldModifiers(key('', { win: true, ctrl: true }))).toEqual(['ctrl', 'windows'])
    expect(heldModifiers(key(''))).toEqual([])
  })
})
