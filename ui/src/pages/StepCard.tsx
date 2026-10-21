// Шаг команды: действие из реестра и поля его параметров по JSON-схеме.

import { ArrowDown, ArrowUp, Trash2 } from 'lucide-react'
import { Segmented } from '../components/controls'
import { Badge, IconButton, inputClass, cx } from '../components/ui'
import { fieldKind, fieldText, fieldValue, rangeHint } from '../lib/commands'
import type { ActionInfo, CommandStep, JsonSchema } from '../lib/types'

export function StepCard({
  index,
  count,
  step,
  actions,
  slots,
  onChange,
  onMove,
  onRemove,
}: {
  index: number
  count: number
  step: CommandStep
  actions: ActionInfo[]
  slots: string[]
  onChange: (step: CommandStep) => void
  onMove: (delta: number) => void
  onRemove: () => void
}) {
  const info = actions.find((action) => action.name === step.action)
  const required = new Set(info?.schema.required ?? [])
  const properties = Object.entries(info?.schema.properties ?? {})
  return (
    <li className="rounded-xl border border-line bg-surface-2/40 p-3.5" aria-label={`Шаг ${index + 1}`}>
      <div className="flex items-center gap-2.5">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-surface-3 text-[12px] text-muted tabular-nums">
          {index + 1}
        </span>
        <ActionSelect value={step.action} actions={actions} onChange={(action) => onChange({ action, args: {} })} />
        {info?.risk === 'confirm' && <Badge tone="warn">спросит подтверждение</Badge>}
        <IconButton label="Шаг выше" disabled={index === 0} onClick={() => onMove(-1)}>
          <ArrowUp size={14} />
        </IconButton>
        <IconButton label="Шаг ниже" disabled={index === count - 1} onClick={() => onMove(1)}>
          <ArrowDown size={14} />
        </IconButton>
        <IconButton label="Удалить шаг" onClick={onRemove}>
          <Trash2 size={14} />
        </IconButton>
      </div>
      {step.action && !info && (
        <p className="mt-2 text-[12.5px] text-warn">Действия {step.action} нет — выберите другое.</p>
      )}
      {properties.length > 0 && (
        <div className="mt-3 flex flex-col gap-3 border-t border-line pt-3">
          {properties.map(([name, schema]) => (
            <ArgField
              key={name}
              name={name}
              schema={schema}
              required={required.has(name)}
              value={step.args[name]}
              slots={slots}
              onChange={(value) => onChange({ ...step, args: { ...step.args, [name]: value } })}
            />
          ))}
        </div>
      )}
    </li>
  )
}

/** Выбор действия: группы, в подписи — что делает, рядом — имя для YAML. */
function ActionSelect({ value, actions, onChange }: { value: string; actions: ActionInfo[]; onChange: (name: string) => void }) {
  const groups = [...new Set(actions.map((action) => action.group))]
  return (
    <div className="flex min-w-0 flex-1 items-center gap-2">
      <select
        aria-label="Действие"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className={cx(inputClass, 'min-w-0 flex-1 cursor-pointer pr-8')}
      >
        <option value="">Выберите действие…</option>
        {groups.map((group) => (
          <optgroup key={group} label={group}>
            {actions
              .filter((action) => action.group === group)
              .map((action) => (
                <option key={action.name} value={action.name}>
                  {shortDescription(action.description)}
                </option>
              ))}
          </optgroup>
        ))}
      </select>
      {value && <code className="truncate font-mono text-[11.5px] text-faint">{value}</code>}
    </div>
  )
}

/** Первое предложение описания: в списке длинные пояснения не нужны. */
function shortDescription(text: string): string {
  return text.split(/(?<=[.!?])\s|\s—\s|\.\s*$/)[0].replace(/\.$/, '')
}

function ArgField({
  name,
  schema,
  required,
  value,
  slots,
  onChange,
}: {
  name: string
  schema: JsonSchema
  required: boolean
  value: unknown
  slots: string[]
  onChange: (value: unknown) => void
}) {
  const kind = fieldKind(schema)
  const hint = rangeHint(schema)
  const insertSlot = (slot: string) =>
    onChange(kind === 'text' || kind === 'longtext' ? `${fieldText(value)}{${slot}}` : `{${slot}}`)
  return (
    <label className="flex flex-col gap-1.5">
      <span className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[12.5px]">
        <code className="font-mono text-[12px] text-text">{name}</code>
        <span className="text-muted">{schema.description}</span>
        {required && <span className="text-[11.5px] text-warn">обязательно</span>}
        {hint && <span className="text-[11.5px] text-faint">{hint}</span>}
      </span>
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <Control kind={kind} schema={schema} required={required} value={value} name={name} onChange={onChange} />
        </div>
        {slots.length > 0 && kind !== 'boolean' && (
          <div className="flex shrink-0 flex-wrap gap-1 pt-1.5">
            {slots.map((slot) => (
              <button
                key={slot}
                type="button"
                title={`Подставить значение слота ${slot}`}
                onClick={(event) => {
                  event.preventDefault()
                  insertSlot(slot)
                }}
                className="h-6 rounded-md border border-accent/30 px-1.5 text-[11.5px] text-accent hover:bg-accent/10"
              >
                {`{${slot}}`}
              </button>
            ))}
          </div>
        )}
      </div>
    </label>
  )
}

function Control({
  kind,
  schema,
  required,
  value,
  name,
  onChange,
}: {
  kind: ReturnType<typeof fieldKind>
  schema: JsonSchema
  required: boolean
  value: unknown
  name: string
  onChange: (value: unknown) => void
}) {
  if (kind === 'boolean') {
    const current = value === true ? 'true' : value === false ? 'false' : ''
    const options = [
      ...(required ? [] : [{ value: '' as const, title: 'Не указано' }]),
      { value: 'true' as const, title: 'Да' },
      { value: 'false' as const, title: 'Нет' },
    ]
    return (
      <Segmented
        label={name}
        value={current}
        options={options}
        onChange={(next) => onChange(next === '' ? '' : next === 'true')}
      />
    )
  }
  if (kind === 'enum') {
    return (
      <select
        aria-label={name}
        value={fieldText(value)}
        onChange={(event) => onChange(event.target.value)}
        className={cx(inputClass, 'w-full cursor-pointer pr-8')}
      >
        <option value="">{required ? 'Выберите…' : 'Не указано'}</option>
        {(schema.enum ?? []).map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
        {typeof value === 'string' && value.startsWith('{') && <option value={value}>{value}</option>}
      </select>
    )
  }
  if (kind === 'longtext') {
    return (
      <textarea
        aria-label={name}
        value={fieldText(value)}
        rows={3}
        onChange={(event) => onChange(fieldValue(event.target.value, kind))}
        className={cx(inputClass, 'h-auto w-full resize-y py-2 leading-relaxed')}
      />
    )
  }
  const example = schema.default !== undefined && schema.default !== null ? `по умолчанию ${fieldText(schema.default)}` : ''
  return (
    <input
      aria-label={name}
      value={fieldText(value)}
      inputMode={kind === 'integer' || kind === 'number' ? 'decimal' : undefined}
      placeholder={example}
      spellCheck={false}
      onChange={(event) => onChange(fieldValue(event.target.value, kind))}
      className={cx(inputClass, 'w-full', kind === 'json' && 'font-mono')}
    />
  )
}
