// Список строк «чипами»: фразы команды, части заголовка окна, программы.

import { X } from 'lucide-react'
import { useState } from 'react'
import { cx } from './ui'

/** Текст со слотами: «громкость на {percent}» — слот выделен цветом. */
export function SlotText({ text }: { text: string }) {
  const parts = text.split(/(\{\w+\})/)
  return (
    <>
      {parts.map((part, index) =>
        /^\{\w+\}$/.test(part) ? (
          <span key={index} className="font-medium text-accent">
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </>
  )
}

/** Enter добавляет, × удаляет, щелчок по чипу возвращает его в поле для правки. */
export function ChipsInput({
  values,
  onChange,
  label,
  placeholder,
  slots = false,
}: {
  values: string[]
  onChange: (values: string[]) => void
  label: string
  placeholder?: string
  slots?: boolean
}) {
  const [text, setText] = useState('')
  const add = () => {
    const value = text.split(/\s+/).filter(Boolean).join(' ')
    if (value && !values.includes(value)) onChange([...values, value])
    setText('')
  }
  return (
    <div
      className={cx(
        'flex min-h-10 flex-wrap items-center gap-1.5 rounded-xl border border-line bg-surface-2 px-2 py-1.5 transition',
        'focus-within:border-accent/60',
      )}
    >
      {values.map((value, index) => (
        <span key={value} className="inline-flex h-7 max-w-full items-center gap-0.5 rounded-lg bg-surface-3 pl-2.5 text-[13px]">
          <button
            type="button"
            title="Изменить"
            className="truncate text-left"
            onClick={() => {
              add()
              setText(value)
              onChange(values.filter((_, i) => i !== index))
            }}
          >
            {slots ? <SlotText text={value} /> : value}
          </button>
          <button
            type="button"
            aria-label={`Убрать «${value}»`}
            onClick={() => onChange(values.filter((_, i) => i !== index))}
            className="flex h-7 w-6 shrink-0 items-center justify-center text-faint hover:text-text"
          >
            <X size={12} />
          </button>
        </span>
      ))}
      <input
        aria-label={label}
        value={text}
        placeholder={values.length ? 'ещё…' : placeholder}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === 'Enter') {
            event.preventDefault()
            add()
          } else if (event.key === 'Backspace' && !text && values.length) {
            onChange(values.slice(0, -1))
          }
        }}
        onBlur={add}
        className="h-7 min-w-36 flex-1 bg-transparent px-1 text-[13px] outline-none placeholder:text-faint"
      />
    </div>
  )
}
