// Редактор одной команды: фразы, слоты, шаги, ответ, условие, горячая клавиша; проверка и запуск.

import { CircleAlert, CircleCheck, FlaskConical, Info, Play, Plus, Save, Trash2, X } from 'lucide-react'
import { useEffect, useState, type ReactNode } from 'react'
import { ChipsInput, SlotText } from '../components/Chips'
import { HotkeyInput } from '../components/controls'
import { Badge, Button, Card, CardTitle, Field, IconButton, TextInput, Toggle, cx } from '../components/ui'
import { call } from '../lib/bridge'
import { SLOT_TYPES, isDirty, placeholders, slugify, uniqueId } from '../lib/commands'
import type { ActionInfo, CommandForm, CommandSaveResult, PhraseTest, SlotType } from '../lib/types'
import { StepCard } from './StepCard'

export interface EditorTarget {
  file: string // откуда открыта ('' — новая)
  originalId: string
  user: boolean // свой пак: правится на месте
  form: CommandForm
  problems: string[]
  alsoIn: string[]
}

export function CommandEditor({
  target,
  actions,
  userFile,
  takenIds,
  onSaved,
  onDeleted,
  onDirty,
}: {
  target: EditorTarget
  actions: ActionInfo[]
  userFile: string
  takenIds: Set<string>
  onSaved: (result: CommandSaveResult) => void
  onDeleted: (result: CommandSaveResult) => void
  onDirty: (dirty: boolean) => void
}) {
  const [form, setForm] = useState(target.form)
  const [idTouched, setIdTouched] = useState(Boolean(target.originalId))
  const [errors, setErrors] = useState<string[]>([])
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState<'save' | 'run' | 'delete' | 'test' | null>(null)
  const [testText, setTestText] = useState('')
  const [test, setTest] = useState<PhraseTest | null>(null)
  const dirty = isDirty(form, target.form)
  const slots = placeholders(form.phrases)
  const isNew = !target.originalId

  useEffect(() => onDirty(dirty), [dirty, onDirty])

  const update = (patch: Partial<CommandForm>) => {
    setNotice('')
    setForm((current) => {
      const next = { ...current, ...patch }
      // У новой команды id подбирается по первой фразе, пока его не поменяли вручную.
      if (!idTouched && patch.phrases) next.id = uniqueId(slugify(next.phrases[0] ?? ''), takenIds)
      return next
    })
  }

  const save = async () => {
    setBusy('save')
    const result = await call('command_save', target.file, target.originalId, form)
    setBusy(null)
    if (!result) return
    setErrors(result.errors)
    if (result.ok) onSaved(result)
  }

  const remove = async () => {
    if (!window.confirm(`Удалить команду «${target.originalId}»?`)) return
    setBusy('delete')
    const result = await call('command_delete', target.file, target.originalId)
    setBusy(null)
    if (!result) return
    setErrors(result.errors)
    if (result.ok) onDeleted(result)
  }

  const check = async () => {
    if (!testText.trim()) return
    setBusy('test')
    setTest((await call('command_test', testText, form, target.originalId)) ?? null)
    setBusy(null)
  }

  const run = async () => {
    setBusy('run')
    const result = await call('command_run', form, testText)
    setBusy(null)
    if (!result) return
    setErrors(result.errors)
    if (result.ok) setNotice('Выполняется — итог появится в Истории и на Главной')
  }

  const title = form.phrases[0] || form.id || 'Новая команда'
  return (
    <div className="flex min-h-full flex-col">
      <div className="flex flex-1 flex-col gap-5 p-7 pb-4">
        <div className="flex items-start gap-3">
          <div className="min-w-0 flex-1">
            <h1 className="truncate font-display text-[22px] font-semibold tracking-tight">
              <SlotText text={title} />
            </h1>
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-[12.5px] text-muted">
              {isNew ? <Badge tone="accent">новая</Badge> : <code className="font-mono text-[12px]">{target.originalId}</code>}
              <span>·</span>
              <span>{target.user || isNew ? userFile : `встроенная, ${target.file}`}</span>
              {dirty && <Badge tone="warn">не сохранено</Badge>}
            </div>
          </div>
          {target.user && !isNew && (
            <Button variant="ghost" icon={<Trash2 size={14} />} busy={busy === 'delete'} onClick={() => void remove()}>
              Удалить
            </Button>
          )}
        </div>

        {!target.user && !isNew && (
          <Note icon={<Info size={15} />}>
            Встроенные паки не меняются: изменения сохранятся копией в {userFile} и заменят эту команду.
            {target.alsoIn.includes(userFile) && ' Копия уже есть — сейчас работает она.'}
          </Note>
        )}
        {target.problems.length > 0 && errors.length === 0 && (
          <Note tone="warn" icon={<CircleAlert size={15} />}>
            Команда не загружается: {target.problems.join('; ')}
          </Note>
        )}
        {errors.length > 0 && (
          <Note tone="danger" icon={<CircleAlert size={15} />}>
            <ul className="flex flex-col gap-0.5">
              {errors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </Note>
        )}

        <Card>
          <CardTitle title="Фразы" />
          <ChipsInput
            label="Новая фраза"
            values={form.phrases}
            slots
            placeholder="Как вы скажете команду, например «рабочий режим»; слот — {в скобках}"
            onChange={(phrases) => update({ phrases })}
          />
          {slots.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-[12.5px]">
              <span className="text-muted">Слоты:</span>
              {slots.map((slot) => (
                <label key={slot} className="flex items-center gap-1.5">
                  <code className="font-mono text-accent">{`{${slot}}`}</code>
                  <select
                    aria-label={`Тип слота ${slot}`}
                    value={form.slots[slot] ?? 'str'}
                    onChange={(event) => update({ slots: { ...form.slots, [slot]: event.target.value as SlotType } })}
                    className="h-7 cursor-pointer rounded-lg border border-line bg-surface-2 px-1.5 text-[12.5px]"
                  >
                    {SLOT_TYPES.map((type) => (
                      <option key={type.value} value={type.value}>
                        {type.title}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          )}
        </Card>

        <Card>
          <CardTitle title="Действия" />
          {form.steps.length > 0 && (
            <ol className="mb-3 flex flex-col gap-2.5">
              {form.steps.map((_, index) => (
                <StepCardSlot
                  key={index}
                  index={index}
                  form={form}
                  actions={actions}
                  slots={slots}
                  onSteps={(steps) => update({ steps })}
                />
              ))}
            </ol>
          )}
          <Button icon={<Plus size={14} />} onClick={() => update({ steps: [...form.steps, { action: '', args: {} }] })}>
            Действие
          </Button>
        </Card>

        <Card>
          <CardTitle title="Дополнительно" />
          <Field label="id" hint="Имя команды в YAML; одинаковый id в my.yaml заменяет встроенную">
            <TextInput
              aria-label="id команды"
              className="w-64 font-mono"
              spellCheck={false}
              value={form.id}
              onChange={(event) => {
                setIdTouched(true)
                update({ id: event.target.value })
              }}
            />
          </Field>
          <Field label="Ответ" hint="Что сказать после выполнения: {result} — итог последнего шага, {слот} — его значение">
            <TextInput aria-label="Ответ" className="w-80" value={form.reply} onChange={(event) => update({ reply: event.target.value })} />
          </Field>
          <Field label="Спрашивать подтверждение" hint="Перед выполнением: «Выполнить …?»">
            <Toggle label="Спрашивать подтверждение" checked={form.confirm} onChange={(confirm) => update({ confirm })} />
          </Field>
          <Field label="Горячая клавиша" hint="Запуск без голоса">
            {form.hotkey && (
              <IconButton label="Убрать горячую клавишу" onClick={() => update({ hotkey: '' })}>
                <X size={15} />
              </IconButton>
            )}
            <HotkeyInput
              label="Горячая клавиша команды"
              placeholder="не назначена"
              value={form.hotkey}
              onChange={(hotkey) => update({ hotkey })}
            />
          </Field>
          <Field label="Только когда активно окно" hint="Часть заголовка (например, YouTube); пусто — всегда" stacked>
            <ChipsInput
              label="Часть заголовка окна"
              values={form.when.active_window_contains}
              placeholder="Часть заголовка окна"
              onChange={(values) => update({ when: { ...form.when, active_window_contains: values } })}
            />
          </Field>
          <Field label="Только в программе" hint="Имя процесса, например chrome.exe; пусто — в любой" stacked>
            <ChipsInput
              label="Программа"
              values={form.when.active_process}
              placeholder="Например, chrome.exe"
              onChange={(values) => update({ when: { ...form.when, active_process: values } })}
            />
          </Field>
        </Card>
      </div>

      <div className="sticky bottom-0 z-10 border-t border-line bg-bg/95 px-7 py-3 backdrop-blur">
        {test && <TestResult test={test} />}
        <div className="flex items-center gap-2">
          <div className="relative min-w-0 flex-1">
            <TextInput
              aria-label="Проверочная фраза"
              placeholder="Проверить фразу, например «громкость на пятьдесят»"
              className="w-full pr-10"
              value={testText}
              onChange={(event) => setTestText(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && void check()}
            />
            <IconButton
              label="Проверить"
              className="absolute top-0.5 right-0.5"
              disabled={!testText.trim() || busy === 'test'}
              onClick={() => void check()}
            >
              <FlaskConical size={15} />
            </IconButton>
          </div>
          <Button icon={<Play size={14} />} busy={busy === 'run'} onClick={() => void run()}>
            Выполнить
          </Button>
          <Button variant="primary" icon={<Save size={14} />} busy={busy === 'save'} disabled={!dirty && !isNew} onClick={() => void save()}>
            Сохранить
          </Button>
        </div>
        {notice && (
          <p role="status" className="mt-2 flex items-center gap-1.5 text-[12.5px] text-ok">
            <CircleCheck size={14} /> {notice}
          </p>
        )}
      </div>
    </div>
  )
}

/** Шаг в списке: правка, перестановка и удаление меняют весь список шагов. */
function StepCardSlot({
  index,
  form,
  actions,
  slots,
  onSteps,
}: {
  index: number
  form: CommandForm
  actions: ActionInfo[]
  slots: string[]
  onSteps: (steps: CommandForm['steps']) => void
}) {
  const steps = form.steps
  return (
    <StepCard
      index={index}
      count={steps.length}
      step={steps[index]}
      actions={actions}
      slots={slots}
      onChange={(step) => onSteps(steps.map((current, i) => (i === index ? step : current)))}
      onMove={(delta) => {
        const next = [...steps]
        const [moved] = next.splice(index, 1)
        next.splice(index + delta, 0, moved)
        onSteps(next)
      }}
      onRemove={() => onSteps(steps.filter((_, i) => i !== index))}
    />
  )
}

function TestResult({ test }: { test: PhraseTest }) {
  const top = test.matches[0]
  let text: ReactNode
  let tone: 'ok' | 'warn' | 'muted' = 'muted'
  if (test.status === 'matched' && top) {
    tone = top.draft ? 'ok' : 'warn'
    const values = Object.entries(top.slots).map(([name, value]) => `${name} = ${JSON.stringify(value)}`)
    text = (
      <>
        {top.draft ? 'Сработает эта команда' : <>Сработает другая команда: <code className="font-mono">{top.id}</code></>}
        {` (оценка ${top.score}${values.length ? `; ${values.join(', ')}` : ''})`}
        {test.matches.length > 1 && ` и ещё ${test.matches.length - 1} — фраза составная`}
      </>
    )
  } else if (test.status === 'ambiguous') {
    tone = 'warn'
    text = `Неоднозначно: ${test.candidates.slice(0, 2).map((c) => c.id).join(' или ')} — уточните фразу`
  } else if (test.status === 'empty') {
    text = 'Пустая фраза'
  } else {
    text = `Ни одна команда не подходит — фразу получит нейросеть${test.candidates[0] ? ` (ближе всех ${test.candidates[0].id}, ${test.candidates[0].score})` : ''}`
  }
  return (
    <p
      role="status"
      aria-label="Результат проверки"
      className={cx(
        'mb-2 text-[12.5px]',
        tone === 'ok' ? 'text-ok' : tone === 'warn' ? 'text-warn' : 'text-muted',
      )}
    >
      {text}
      {test.errors.length > 0 && <span className="text-muted"> (черновик с ошибками — проверено без него)</span>}
    </p>
  )
}

function Note({
  icon,
  tone = 'muted',
  children,
}: {
  icon: ReactNode
  tone?: 'muted' | 'warn' | 'danger'
  children: ReactNode
}) {
  return (
    <div
      className={cx(
        'flex items-start gap-2.5 rounded-card border px-4 py-3 text-[13px] leading-relaxed',
        tone === 'danger' ? 'border-danger/30 bg-danger/8 text-danger' : tone === 'warn' ? 'border-warn/30 bg-warn/8 text-warn' : 'border-line bg-surface text-muted',
      )}
    >
      <span className="mt-0.5 shrink-0">{icon}</span>
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}

