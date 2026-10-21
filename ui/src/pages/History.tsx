// Страница «История»: фразы из журнала — что сказано, куда ушло, какие действия, чем кончилось.

import {
  Ban,
  BookmarkPlus,
  ChevronDown,
  CircleCheck,
  CircleSlash,
  CircleX,
  FileText,
  FlaskConical,
  History as HistoryIcon,
  Keyboard,
  Mic,
  RotateCcw,
  Search,
  SquareTerminal,
  Type,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Segmented } from '../components/controls'
import { Badge, Button, IconButton, TextInput, cx } from '../components/ui'
import { call } from '../lib/bridge'
import { formatMs } from '../lib/format'
import { useAssistantEvent } from '../lib/store'
import type { HistoryAction, HistoryFilter, HistoryItem } from '../lib/types'

const PAGE = 50

const filters: Array<{ value: HistoryFilter; title: string }> = [
  { value: 'all', title: 'Все' },
  { value: 'errors', title: 'Ошибки' },
  { value: 'llm', title: 'Нейросеть' },
  { value: 'local', title: 'Команды' },
]

export function HistoryPage({ onSaveAsCommand }: { onSaveAsCommand?: (item: HistoryItem) => void }) {
  const [filter, setFilter] = useState<HistoryFilter>('all')
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [more, setMore] = useState(false)
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const generation = useRef(0)

  useEffect(() => {
    const timer = window.setTimeout(() => setQuery(search), 250)
    return () => window.clearTimeout(timer)
  }, [search])

  const load = useCallback(
    async (after?: HistoryItem) => {
      const id = ++generation.current
      setLoading(true)
      const page = await call('history', {
        filter,
        search: query,
        limit: PAGE,
        before: after?.id,
        before_time: after?.time,
      })
      if (id !== generation.current) return // пока ждали, фильтр сменился
      setLoading(false)
      if (!page) return
      setItems((current) => (after && current ? [...current, ...page.items] : page.items))
      setMore(page.more)
    },
    [filter, query],
  )

  useEffect(() => {
    void load()
  }, [load])
  // Выполнена новая фраза — она появляется сверху.
  useAssistantEvent((event) => {
    if (event.type === 'result') void load()
  })

  const groups = groupByDay(items ?? [])
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-5 p-7">
      <div className="flex items-center gap-3">
        <h1 className="flex-1 font-display text-[22px] font-semibold tracking-tight">История</h1>
        <Segmented label="Что показать" value={filter} options={filters} onChange={setFilter} />
        <div className="relative">
          <Search size={14} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-faint" />
          <TextInput
            aria-label="Поиск по истории"
            placeholder="Поиск"
            className="w-52 pl-8"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
          />
        </div>
        <IconButton label="Открыть журнал целиком" onClick={() => void call('open_path', 'journal')}>
          <FileText size={16} />
        </IconButton>
      </div>

      {items === null ? null : items.length === 0 ? (
        <Empty filtered={filter !== 'all' || query.trim() !== ''} />
      ) : (
        groups.map(([day, dayItems]) => (
          <section key={day} aria-label={day}>
            <h2 className="mb-2 px-1 text-[12.5px] font-medium text-muted">{day}</h2>
            <ul className="overflow-hidden rounded-card border border-line bg-surface">
              {dayItems.map((item) => (
                <Row
                  key={item.id}
                  item={item}
                  open={open === item.id}
                  onToggle={() => setOpen(open === item.id ? null : item.id)}
                  onSaveAsCommand={onSaveAsCommand}
                />
              ))}
            </ul>
          </section>
        ))
      )}

      {more && (
        <div className="flex justify-center">
          <Button busy={loading} onClick={() => void load(items?.at(-1))}>
            Показать ещё
          </Button>
        </div>
      )}
    </div>
  )
}

function Row({
  item,
  open,
  onToggle,
  onSaveAsCommand,
}: {
  item: HistoryItem
  open: boolean
  onToggle: () => void
  onSaveAsCommand?: (item: HistoryItem) => void
}) {
  const reusable = item.route === 'llm' && item.actions.some((action) => okStatus(action.status))
  return (
    <li className="border-t border-line first:border-t-0">
      <button
        type="button"
        aria-expanded={open}
        onClick={onToggle}
        className="flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-surface-2/60"
      >
        <span className="mt-0.5 w-11 shrink-0 text-[12px] text-faint tabular-nums">{clock(item.time)}</span>
        <span className="mt-0.5 w-20 shrink-0">
          <Badge tone={!item.ok ? 'danger' : item.route === 'llm' ? 'accent' : 'neutral'}>
            {!item.ok ? 'ошибка' : item.route === 'llm' ? 'нейросеть' : 'команда'}
          </Badge>
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-2 text-[13.5px]">
            <SourceIcon source={item.source} />
            <span className="truncate">{item.phrase || 'Без фразы'}</span>
          </span>
          {(item.reply || item.error) && (
            <span className={cx('mt-0.5 block truncate text-[12.5px]', item.ok ? 'text-muted' : 'text-danger')}>
              {item.ok ? item.reply : item.error}
            </span>
          )}
        </span>
        <span className="mt-0.5 shrink-0 text-[11.5px] text-faint tabular-nums">
          {item.actions.length > 0 && `${item.actions.length} ${actionWord(item.actions.length)} · `}
          {formatMs(item.duration_ms)}
        </span>
        <ChevronDown
          size={15}
          className={cx('mt-0.5 shrink-0 text-faint transition-transform duration-200', open && 'rotate-180')}
        />
      </button>
      {open && (
        <div className="animate-fade-up flex flex-col gap-3 border-t border-line bg-bg/40 px-4 py-3 pl-[156px]">
          {item.reply && <Detail title="Ответ">{item.reply}</Detail>}
          {item.error && item.error !== item.reply && <Detail title="Что пошло не так">{item.error}</Detail>}
          {item.commands.length > 0 && <Detail title="Команда">{item.commands.join(', ')}</Detail>}
          {item.actions.length > 0 && (
            <div>
              <div className="mb-1.5 text-[11.5px] font-medium tracking-wide text-faint">Действия</div>
              <ol className="flex flex-col gap-1.5">
                {item.actions.map((action, index) => (
                  <ActionLine key={index} action={action} />
                ))}
              </ol>
            </div>
          )}
          <div className="flex gap-2">
            {item.phrase && (
              <Button icon={<RotateCcw size={14} />} onClick={() => void call('send_text', item.phrase)}>
                Повторить
              </Button>
            )}
            {reusable && onSaveAsCommand && (
              <Button icon={<BookmarkPlus size={14} />} onClick={() => onSaveAsCommand(item)}>
                Сохранить как команду
              </Button>
            )}
          </div>
        </div>
      )}
    </li>
  )
}

function ActionLine({ action }: { action: HistoryAction }) {
  return (
    <li className="flex items-start gap-2 text-[12.5px]">
      <StatusIcon status={action.status} />
      <div className="min-w-0 flex-1 select-text">
        <code className="font-mono text-[12px] break-all text-text">
          {action.action}({formatArgs(action.args)})
        </code>
        {action.result && <div className="mt-0.5 leading-snug break-words text-muted">{action.result}</div>}
      </div>
      <span className="shrink-0 text-[11.5px] text-faint tabular-nums">{formatMs(action.duration_ms)}</span>
    </li>
  )
}

function Detail({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <div className="mb-0.5 text-[11.5px] font-medium tracking-wide text-faint">{title}</div>
      <div className="text-[13px] leading-relaxed break-words whitespace-pre-wrap select-text">{children}</div>
    </div>
  )
}

function StatusIcon({ status }: { status: string }) {
  const icons: Record<string, [ReactNode, string]> = {
    done: [<CircleCheck size={14} className="text-ok" />, 'выполнено'],
    dry_run: [<FlaskConical size={14} className="text-warn" />, 'пробный режим: не выполнялось'],
    failed: [<CircleX size={14} className="text-danger" />, 'не получилось'],
    forbidden: [<Ban size={14} className="text-danger" />, 'запрещено'],
    cancelled: [<CircleSlash size={14} className="text-muted" />, 'не подтверждено'],
  }
  const [icon, title] = icons[status] ?? icons.failed
  return (
    <span className="mt-0.5 shrink-0" title={title} aria-label={title}>
      {icon}
    </span>
  )
}

function SourceIcon({ source }: { source: string }) {
  const icons: Record<string, [ReactNode, string]> = {
    voice: [<Mic size={13} />, 'голосом'],
    text: [<Type size={13} />, 'текстом'],
    hotkey: [<Keyboard size={13} />, 'горячей клавишей'],
    local: [<SquareTerminal size={13} />, 'в консоли'],
  }
  const known = icons[source]
  if (!known) return null
  return (
    <span className="shrink-0 text-faint" title={known[1]} aria-label={known[1]}>
      {known[0]}
    </span>
  )
}

function Empty({ filtered }: { filtered: boolean }) {
  return (
    <div className="flex flex-col items-center gap-3 py-16 text-center">
      <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-surface-2 text-muted">
        <HistoryIcon size={22} />
      </span>
      <p className="max-w-sm text-[13px] leading-relaxed text-muted">
        {filtered
          ? 'Ничего не нашлось — попробуйте другой фильтр или запрос.'
          : 'Здесь появятся ваши команды: что сказано, что сделано и чем закончилось.'}
      </p>
    </div>
  )
}

export function okStatus(status: string): boolean {
  return status === 'done' || status === 'dry_run'
}

/** Аргументы действия коротко: name="Telegram", percent=30; пути — без экранирования «\\». */
export function formatArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .filter(([, value]) => value !== null && value !== '' && value !== undefined)
    .map(([key, value]) => `${key}=${typeof value === 'string' ? `"${value}"` : JSON.stringify(value)}`)
    .join(', ')
}

function actionWord(count: number): string {
  const tens = count % 100
  const ones = tens % 10
  if (tens > 10 && tens < 20) return 'действий'
  if (ones === 1) return 'действие'
  if (ones >= 2 && ones <= 4) return 'действия'
  return 'действий'
}

function clock(time: string): string {
  const date = new Date(time)
  return Number.isNaN(date.getTime()) ? '' : date.toTimeString().slice(0, 5)
}

/** Фразы по дням: «Сегодня», «Вчера», «20 октября» (с годом, если не текущий). */
export function groupByDay(items: HistoryItem[], now = new Date()): Array<[string, HistoryItem[]]> {
  const groups = new Map<string, HistoryItem[]>()
  for (const item of items) {
    const label = dayLabel(new Date(item.time), now)
    groups.set(label, [...(groups.get(label) ?? []), item])
  }
  return [...groups]
}

export function dayLabel(date: Date, now = new Date()): string {
  if (Number.isNaN(date.getTime())) return 'Без даты'
  const day = (value: Date) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime()
  const diff = Math.round((day(now) - day(date)) / 86_400_000)
  if (diff === 0) return 'Сегодня'
  if (diff === 1) return 'Вчера'
  const sameYear = date.getFullYear() === now.getFullYear()
  return date.toLocaleDateString(
    'ru-RU',
    sameYear ? { day: 'numeric', month: 'long' } : { day: 'numeric', month: 'long', year: 'numeric' },
  )
}
