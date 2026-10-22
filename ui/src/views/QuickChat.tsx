// Быстрый чат: окно с полем ввода поверх всех окон (Ctrl+Alt+Enter).
// Текст уходит в тот же конвейер, что и голос; Esc прячет окно.

import { CircleAlert, CircleCheck, CornerDownLeft, Sparkles, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Keys, cx } from '../components/ui'
import { call } from '../lib/bridge'
import { formatMs } from '../lib/format'
import { stepLabel } from '../lib/overlay'
import { useAssistantEvent } from '../lib/store'
import type { AssistantEvent, Step } from '../lib/types'

export interface Exchange {
  id: number
  text: string
  steps: Step[]
  reply: string
  status: 'running' | 'ok' | 'error' | 'stopped'
  ms: number | null
}

/** Как событие меняет разговор: шаги и ответ относятся к последней команде, пока она выполняется. */
export function chatReduce(exchanges: Exchange[], event: AssistantEvent): Exchange[] {
  const last = exchanges.at(-1)
  if (!last || last.status !== 'running') return exchanges
  const update = (values: Partial<Exchange>) => [...exchanges.slice(0, -1), { ...last, ...values }]
  switch (event.type) {
    case 'step':
      return update({ steps: [...last.steps.filter((step) => step.number !== event.payload.number), event.payload] })
    case 'reply':
      return update({ reply: event.payload.text })
    case 'stopped':
      return update({ status: 'stopped' }) // итог прерванной задачи придёт следом — он уже не нужен
    case 'result':
      return update({
        status: event.payload.ok ? 'ok' : 'error',
        reply: event.payload.reply || last.reply,
        ms: event.payload.ms,
      })
    default:
      return exchanges
  }
}

export function QuickChat({ framed = false }: { framed?: boolean }) {
  const [text, setText] = useState('')
  const [exchanges, setExchanges] = useState<Exchange[]>([])
  const [confirm, setConfirm] = useState<{ id: string; question: string } | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const nextId = useRef(1)
  const running = exchanges.at(-1)?.status === 'running'

  useAssistantEvent((event) => {
    if (event.type === 'chat_shown') {
      input.current?.focus()
      input.current?.select()
    } else if (event.type === 'confirm') {
      setConfirm(event.payload)
    } else if (event.type === 'confirm_done') {
      setConfirm((current) => (current?.id === event.payload.id ? null : current))
    } else {
      setExchanges((current) => chatReduce(current, event))
    }
  })

  useEffect(() => {
    input.current?.focus()
  }, [])
  useEffect(() => {
    list.current?.scrollTo({ top: list.current.scrollHeight, behavior: 'smooth' })
  }, [exchanges])

  const send = () => {
    const phrase = text.trim()
    if (!phrase || running) return
    const exchange: Exchange = { id: nextId.current++, text: phrase, steps: [], reply: '', status: 'running', ms: null }
    setExchanges((current) => [...current, exchange].slice(-20))
    setText('')
    void call('send_text', phrase)
  }

  return (
    <div
      className={cx(
        'flex flex-col bg-surface',
        framed ? 'h-[400px] w-[620px] overflow-hidden rounded-2xl border border-line-strong shadow-2xl' : 'h-full w-full',
      )}
    >
      <div className="pywebview-drag-region flex h-14 shrink-0 items-center gap-3 border-b border-line px-4">
        <Sparkles size={18} className="shrink-0 text-accent" />
        <input
          ref={input}
          aria-label="Команда"
          placeholder={running ? 'Выполняю…' : 'Напишите команду или вопрос'}
          className="h-full min-w-0 flex-1 bg-transparent text-[15px] text-text outline-none placeholder:text-faint focus-visible:outline-none"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault()
              send()
            } else if (event.key === 'Escape') {
              event.preventDefault()
              void call('window_action', 'chat', 'hide')
            } else if (event.key === 'ArrowUp' && !text) {
              const previous = exchanges.at(-1)?.text
              if (previous) {
                event.preventDefault()
                setText(previous)
              }
            }
          }}
        />
        {running ? (
          <button
            type="button"
            aria-label="Остановить"
            title="Остановить"
            onClick={() => void call('voice_stop')}
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-surface-3 text-text transition hover:bg-surface-2"
          >
            <Square size={13} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            aria-label="Выполнить"
            disabled={!text.trim()}
            onClick={send}
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-white transition hover:brightness-110 disabled:bg-surface-3 disabled:text-faint"
          >
            <CornerDownLeft size={15} />
          </button>
        )}
      </div>

      <div ref={list} className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        {exchanges.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-1.5 text-center text-[13px] text-muted">
            <span>Например: «открой блокнот», «громкость 40» или «создай на рабочем столе папку Проекты»</span>
            <span className="text-[12px] text-faint">Команды выполняются так же, как голосовые</span>
          </div>
        ) : (
          <ul className="flex flex-col gap-3">
            {exchanges.map((exchange) => (
              <ExchangeView key={exchange.id} exchange={exchange} />
            ))}
          </ul>
        )}
      </div>

      {confirm ? (
        <div className="animate-fade-up flex shrink-0 items-center gap-3 border-t border-line bg-accent/10 px-4 py-2.5">
          <span className="min-w-0 flex-1 text-[13px]">{confirm.question}</span>
          <button
            type="button"
            onClick={() => void call('confirm_answer', confirm.id, true)}
            className="h-8 rounded-lg bg-accent px-3.5 text-[13px] font-medium text-white transition hover:brightness-110"
          >
            Да
          </button>
          <button
            type="button"
            onClick={() => void call('confirm_answer', confirm.id, false)}
            className="h-8 rounded-lg bg-surface-3 px-3.5 text-[13px] font-medium transition hover:bg-surface-2"
          >
            Нет
          </button>
        </div>
      ) : (
        <div className="flex h-9 shrink-0 items-center gap-4 border-t border-line px-4 text-[11.5px] text-faint">
          <span className="flex items-center gap-1.5">
            <Keys value="enter" /> выполнить
          </span>
          <span className="flex items-center gap-1.5">
            <Keys value="esc" /> закрыть
          </span>
          <span className="flex items-center gap-1.5">
            <Keys value="up" /> прошлая команда
          </span>
        </div>
      )}
    </div>
  )
}

const exchangeStatus = { ok: 'Готово', error: 'Не получилось', stopped: 'Остановлено' }

function ExchangeView({ exchange }: { exchange: Exchange }) {
  return (
    <li className="animate-fade-up flex flex-col gap-1.5">
      <div className="self-end rounded-2xl rounded-br-md bg-accent/15 px-3 py-1.5 text-[13.5px] select-text">{exchange.text}</div>
      {exchange.steps.map((step) => (
        <div key={step.number} className="flex items-center gap-2 text-[12.5px] text-muted">
          {step.ok ? <CircleCheck size={13} className="shrink-0 text-ok" /> : <CircleAlert size={13} className="shrink-0 text-danger" />}
          <span className="truncate">{stepLabel(step)}</span>
        </div>
      ))}
      {exchange.reply && <div className="text-[13.5px] leading-relaxed whitespace-pre-wrap select-text">{exchange.reply}</div>}
      {exchange.status === 'running' && !exchange.reply && (
        <div className="flex items-center gap-2 text-[12.5px] text-muted">
          <span className="animate-spin-slow h-3 w-3 rounded-full border-2 border-transparent border-t-accent" />
          Выполняю…
        </div>
      )}
      {exchange.status !== 'running' && (
        <div className={cx('text-[11.5px]', exchange.status === 'error' ? 'text-danger' : 'text-faint')}>
          {exchangeStatus[exchange.status]}
          {exchange.ms !== null && ` · ${formatMs(exchange.ms)}`}
        </div>
      )}
    </li>
  )
}
