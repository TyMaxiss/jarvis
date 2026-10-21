// Базовые элементы интерфейса: карточки, кнопки, переключатели, поля.

import { Loader2 } from 'lucide-react'
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode } from 'react'

export function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ')
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <section className={cx('rounded-card border border-line bg-surface p-5', className)}>{children}</section>
  )
}

export function CardTitle({ icon, title, action }: { icon?: ReactNode; title: string; action?: ReactNode }) {
  return (
    <div className="card-title mb-4 flex items-center gap-2.5">
      {icon && <span className="text-muted">{icon}</span>}
      <h3 className="flex-1 text-[13px] font-medium tracking-wide text-muted">{title}</h3>
      {action}
    </div>
  )
}

type Variant = 'primary' | 'subtle' | 'ghost' | 'danger'

const variants: Record<Variant, string> = {
  primary: 'bg-accent text-white hover:brightness-110',
  subtle: 'bg-surface-2 text-text hover:bg-surface-3 border border-line',
  ghost: 'text-muted hover:text-text hover:bg-surface-2',
  danger: 'bg-danger/15 text-danger hover:bg-danger/25',
}

export function Button({
  variant = 'subtle',
  busy = false,
  icon,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; busy?: boolean; icon?: ReactNode }) {
  return (
    <button
      type="button"
      {...props}
      disabled={props.disabled || busy}
      className={cx(
        'inline-flex h-9 items-center justify-center gap-2 rounded-xl px-3.5 text-[13px] font-medium',
        'transition duration-150 ease-out disabled:cursor-default disabled:opacity-50',
        variants[variant],
        className,
      )}
    >
      {busy ? <Loader2 size={15} className="animate-spin-slow" /> : icon}
      {children}
    </button>
  )
}

export function IconButton({
  label,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      {...props}
      className={cx(
        'inline-flex h-8 w-8 items-center justify-center rounded-lg text-muted transition duration-150',
        'hover:bg-surface-2 hover:text-text disabled:opacity-40',
        className,
      )}
    >
      {children}
    </button>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  disabled,
}: {
  checked: boolean
  onChange: (value: boolean) => void
  label: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx(
        'relative h-[22px] w-[38px] shrink-0 rounded-full transition-colors duration-200 ease-out',
        checked ? 'bg-accent' : 'bg-surface-3',
        disabled && 'opacity-50',
      )}
    >
      <span
        className={cx(
          'absolute top-[3px] left-[3px] h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200 ease-out',
          checked && 'translate-x-4',
        )}
      />
    </button>
  )
}

/** Строка настройки: подпись и пояснение слева, элемент управления справа. */
export function Field({
  label,
  hint,
  children,
  stacked = false,
}: {
  label: string
  hint?: ReactNode
  children: ReactNode
  stacked?: boolean
}) {
  return (
    <div
      className={cx(
        'flex gap-4 border-t border-line py-3.5 first:border-t-0 first:pt-0 last:pb-0',
        '[.card-title+&]:border-t-0 [.card-title+&]:pt-0',
        stacked ? 'flex-col' : 'items-center',
      )}
    >
      <div className="min-w-0 flex-1">
        <div className="text-[13.5px]">{label}</div>
        {hint && <div className="mt-0.5 text-[12.5px] leading-snug text-muted">{hint}</div>}
      </div>
      <div className={cx('flex items-center gap-2', stacked ? 'w-full' : 'shrink-0')}>{children}</div>
    </div>
  )
}

export const inputClass =
  'h-9 rounded-xl border border-line bg-surface-2 px-3 text-[13px] text-text outline-none transition ' +
  'placeholder:text-faint focus:border-accent/60 disabled:opacity-50'

export function Select({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: string
  onChange: (value: string) => void
  options: Array<{ value: string; title: string }>
  label: string
  className?: string
}) {
  return (
    <select
      aria-label={label}
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className={cx(inputClass, 'min-w-48 cursor-pointer pr-8', className)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.title}
        </option>
      ))}
    </select>
  )
}

export function TextInput({
  className,
  ...props
}: InputHTMLAttributes<HTMLInputElement> & { className?: string }) {
  return <input {...props} className={cx(inputClass, className)} />
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  label,
  className,
}: {
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  label: string
  className?: string
}) {
  const percent = ((value - min) / (max - min)) * 100
  return (
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      className={cx('h-1.5 w-full cursor-pointer appearance-none rounded-full accent-[var(--accent)]', className)}
      style={{
        background: `linear-gradient(to right, var(--accent) ${percent}%, var(--surface-3) ${percent}%)`,
      }}
    />
  )
}

export function Badge({ tone = 'neutral', children }: { tone?: 'neutral' | 'ok' | 'warn' | 'danger' | 'accent'; children: ReactNode }) {
  const tones = {
    neutral: 'bg-surface-3 text-muted',
    ok: 'bg-ok/12 text-ok',
    warn: 'bg-warn/12 text-warn',
    danger: 'bg-danger/12 text-danger',
    accent: 'bg-accent/15 text-accent',
  }
  return (
    <span className={cx('inline-flex h-5 items-center rounded-md px-1.5 text-[11.5px] font-medium', tones[tone])}>
      {children}
    </span>
  )
}

const keyNames: Record<string, string> = {
  ctrl: 'Ctrl',
  alt: 'Alt',
  shift: 'Shift',
  win: 'Win',
  windows: 'Win',
  space: 'Пробел',
  enter: 'Enter',
  esc: 'Esc',
  tab: 'Tab',
  backspace: 'Backspace',
  insert: 'Ins',
  delete: 'Del',
  home: 'Home',
  end: 'End',
  'page up': 'PgUp',
  'page down': 'PgDn',
  up: '↑',
  down: '↓',
  left: '←',
  right: '→',
  pause: 'Pause',
  'scroll lock': 'ScrLk',
}

/** Подпись клавиши: «ctrl» → Ctrl, «q» → Q, «page up» → PgUp. */
export function keyLabel(key: string): string {
  const name = key.trim().toLowerCase()
  return keyNames[name] ?? name.toUpperCase()
}

/** Сочетание клавиш: «ctrl+alt+space» → Ctrl Alt Пробел. */
export function Keys({ value }: { value: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      {value
        .split('+')
        .filter((key) => key.trim())
        .map((key) => (
          <kbd
            key={key}
            className="rounded-md border border-line-strong bg-surface-2 px-1.5 py-0.5 font-sans text-[11.5px] text-muted"
          >
            {keyLabel(key)}
          </kbd>
        ))}
    </span>
  )
}
