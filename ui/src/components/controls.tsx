// Элементы настроек: запись сочетания клавиш, переключатель вариантов, уровень микрофона, выбор цвета.

import { Check, Keyboard } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { call } from '../lib/bridge'
import { comboFromEvent, heldModifiers } from '../lib/hotkey'
import { useAssistantEvent } from '../lib/store'
import { Keys, cx } from './ui'

/** Кнопка с текущим сочетанием; по щелчку ждёт нажатия нового. Esc — отмена. */
export function HotkeyInput({
  value,
  onChange,
  label,
  disabled,
}: {
  value: string
  onChange: (combo: string) => void
  label: string
  disabled?: boolean
}) {
  const [recording, setRecording] = useState(false)
  const [held, setHeld] = useState<string[]>([])
  const [error, setError] = useState('')
  const changeRef = useRef(onChange)
  useEffect(() => {
    changeRef.current = onChange
  })

  useEffect(() => {
    if (!recording) return
    // Пока идёт запись, глобальные сочетания ассистента не должны срабатывать.
    void call('pause_hotkeys', true)
    const finish = () => {
      setRecording(false)
      setHeld([])
    }
    const onKeyDown = (event: KeyboardEvent) => {
      event.preventDefault()
      event.stopPropagation()
      const result = comboFromEvent(event)
      if (result.kind === 'pending') {
        setHeld(result.modifiers)
        setError('')
      } else if (result.kind === 'cancel') {
        setError('')
        finish()
      } else if (result.kind === 'error') {
        setError(result.message)
      } else {
        setError('')
        finish()
        changeRef.current(result.combo)
      }
    }
    const onKeyUp = (event: KeyboardEvent) => {
      event.preventDefault()
      setHeld(heldModifiers(event))
    }
    const onBlur = () => finish()
    window.addEventListener('keydown', onKeyDown, true)
    window.addEventListener('keyup', onKeyUp, true)
    window.addEventListener('blur', onBlur)
    return () => {
      window.removeEventListener('keydown', onKeyDown, true)
      window.removeEventListener('keyup', onKeyUp, true)
      window.removeEventListener('blur', onBlur)
      void call('pause_hotkeys', false)
    }
  }, [recording])

  return (
    <div className="flex flex-col items-end gap-1">
      <button
        type="button"
        aria-label={`${label}: ${recording ? 'нажмите новое сочетание' : value}`}
        disabled={disabled}
        onClick={() => {
          setError('')
          setRecording(!recording)
        }}
        className={cx(
          'flex h-9 min-w-40 items-center justify-between gap-3 rounded-xl border px-3 text-[13px] transition',
          recording ? 'border-accent/70 bg-accent/10' : 'border-line bg-surface-2 hover:bg-surface-3',
          'disabled:opacity-50',
        )}
      >
        {recording ? (
          held.length ? (
            <Keys value={held.join('+')} />
          ) : (
            <span className="text-muted">Нажмите сочетание…</span>
          )
        ) : (
          <Keys value={value} />
        )}
        <Keyboard size={15} className={recording ? 'text-accent' : 'text-faint'} />
      </button>
      {recording && !error && <span className="text-[11.5px] text-faint">Esc — отмена</span>}
      {error && <span className="max-w-64 text-right text-[11.5px] leading-snug text-warn">{error}</span>}
    </div>
  )
}

/** Выбор одного из нескольких вариантов кнопками. */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
  disabled,
}: {
  value: T
  options: Array<{ value: T; title: string }>
  onChange: (value: T) => void
  label: string
  disabled?: boolean
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-xl border border-line bg-surface-2 p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={option.value === value}
          disabled={disabled}
          onClick={() => onChange(option.value)}
          className={cx(
            'h-8 rounded-[10px] px-3 text-[12.5px] transition duration-150 disabled:opacity-50',
            option.value === value ? 'bg-surface-3 text-text shadow-sm' : 'text-muted hover:text-text',
          )}
        >
          {option.title}
        </button>
      ))}
    </div>
  )
}

/** Уровень сигнала микрофона: Python присылает события level, пока индикатор на экране. */
export function LevelMeter() {
  const [level, setLevel] = useState(0)
  useEffect(() => {
    void call('level_meter', true)
    return () => {
      void call('level_meter', false)
    }
  }, [])
  useAssistantEvent((event) => {
    if (event.type === 'level') setLevel(Math.max(0, Math.min(1, event.payload.value)))
  })
  return (
    <div
      role="meter"
      aria-label="Уровень микрофона"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(level * 100)}
      className="h-1.5 w-full overflow-hidden rounded-full bg-surface-3"
    >
      <div
        className="h-full rounded-full bg-accent transition-[width] duration-100 ease-out"
        style={{ width: `${Math.round(level * 100)}%` }}
      />
    </div>
  )
}

export const accentColors = ['#7c6cff', '#4f8cff', '#22b8cf', '#3ecf8e', '#f2a541', '#f06292']

/** Выбор акцентного цвета из готовых вариантов. */
export function Swatches({ value, onChange }: { value: string; onChange: (color: string) => void }) {
  return (
    <div role="radiogroup" aria-label="Акцентный цвет" className="flex items-center gap-2">
      {accentColors.map((color) => (
        <button
          key={color}
          type="button"
          role="radio"
          aria-checked={color === value.toLowerCase()}
          aria-label={color}
          onClick={() => onChange(color)}
          className="flex h-6 w-6 items-center justify-center rounded-full transition-transform duration-150 hover:scale-110"
          style={{ background: color }}
        >
          {color === value.toLowerCase() && <Check size={13} strokeWidth={3} className="text-white" />}
        </button>
      ))}
    </div>
  )
}
