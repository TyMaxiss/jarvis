// Главное окно: своя шапка, боковое меню и разделы.

import { BellRing, CircleAlert, Info, ShieldQuestion, X } from 'lucide-react'
import { useCallback, useRef, useState } from 'react'
import { Sidebar, type Page } from '../components/Sidebar'
import { ResizeGrip, TitleBar } from '../components/TitleBar'
import { Button, IconButton } from '../components/ui'
import { call } from '../lib/bridge'
import { draftFromHistory } from '../lib/commands'
import { useAppState, useAssistantEvent } from '../lib/store'
import type { CommandForm } from '../lib/types'
import { CommandsPage } from '../pages/Commands'
import { HistoryPage } from '../pages/History'
import { Home } from '../pages/Home'
import { SettingsPage } from '../pages/Settings'

export function MainWindow() {
  const state = useAppState()
  const [page, setPage] = useState<Page>('home')
  const [draft, setDraft] = useState<CommandForm | null>(null)
  const draftUsed = useCallback(() => setDraft(null), [])

  return (
    <div className="flex h-full flex-col bg-bg">
      <TitleBar title={state?.name ?? 'Джарвис'} dryRun={state?.dry_run ?? false} />
      <div className="flex min-h-0 flex-1">
        <Sidebar page={page} onPage={setPage} state={state} />
        <main className="relative min-w-0 flex-1 overflow-y-auto">
          <ConfirmBanner />
          {!state ? null : page === 'home' ? (
            <Home state={state} onOpenSettings={() => setPage('settings')} />
          ) : page === 'settings' ? (
            <SettingsPage />
          ) : page === 'history' ? (
            <HistoryPage
              onSaveAsCommand={(item) => {
                setDraft(draftFromHistory(item))
                setPage('commands')
              }}
            />
          ) : (
            <CommandsPage draft={draft} onDraftUsed={draftUsed} />
          )}
        </main>
      </div>
      <Toasts />
      <ResizeGrip />
    </div>
  )
}

/** Подтверждение рискованного действия, если главное окно открыто. */
function ConfirmBanner() {
  const [confirm, setConfirm] = useState<{ id: string; question: string } | null>(null)
  useAssistantEvent((event) => {
    if (event.type === 'confirm') setConfirm(event.payload)
    if (event.type === 'confirm_done') setConfirm((current) => (current?.id === event.payload.id ? null : current))
  })
  if (!confirm) return null
  return (
    <div className="sticky top-0 z-10 px-7 pt-5">
      <div
        role="alertdialog"
        aria-label="Подтверждение"
        className="animate-fade-up mx-auto flex max-w-5xl items-center gap-3 rounded-card border border-accent/40 bg-surface px-4 py-3 shadow-lg"
      >
        <ShieldQuestion size={18} className="shrink-0 text-accent" />
        <span className="flex-1 text-[13.5px]">{confirm.question}</span>
        <Button variant="primary" onClick={() => void call('confirm_answer', confirm.id, true)}>
          Да
        </Button>
        <Button onClick={() => void call('confirm_answer', confirm.id, false)}>Нет</Button>
      </div>
    </div>
  )
}

interface Toast {
  id: number
  kind: 'info' | 'error' | 'alarm'
  text: string
}

const toastIcons = {
  info: <Info size={16} className="mt-px shrink-0 text-accent" />,
  error: <CircleAlert size={16} className="mt-px shrink-0 text-danger" />,
  alarm: <BellRing size={16} className="mt-px shrink-0 text-accent" />,
}

/** Уведомления: перезагрузка команд, ошибки сохранения; сработавший таймер висит, пока не закроют. */
function Toasts() {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(1)
  const push = (kind: Toast['kind'], text: string) => {
    const id = nextId.current++
    // На экране не больше трёх, но сработавший таймер не вытесняется.
    setToasts((current) =>
      [...current, { id, kind, text }].filter((toast, index, all) => toast.kind === 'alarm' || index >= all.length - 3),
    )
    if (kind === 'alarm') return
    window.setTimeout(() => setToasts((current) => current.filter((toast) => toast.id !== id)), kind === 'error' ? 8000 : 4000)
  }
  useAssistantEvent((event) => {
    if (event.type === 'toast') push(event.payload.kind, event.payload.text)
    if (event.type === 'commands') {
      if (event.payload.issues.length) push('error', `Команды: ${event.payload.issues.join('; ')}`)
      else push('info', `Команды обновлены: ${event.payload.count}`)
    }
  })
  return (
    <div className="pointer-events-none fixed right-5 bottom-5 z-40 flex w-80 flex-col gap-2" aria-live="polite">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className="animate-fade-up pointer-events-auto flex items-start gap-2.5 rounded-xl border border-line-strong bg-surface-2 px-3.5 py-3 text-[13px] shadow-lg"
        >
          {toastIcons[toast.kind]}
          <span className="min-w-0 flex-1 leading-snug break-words">{toast.text}</span>
          <IconButton
            label="Закрыть"
            className="-my-1 -mr-1.5 h-6 w-6"
            onClick={() => setToasts((current) => current.filter((item) => item.id !== toast.id))}
          >
            <X size={13} />
          </IconButton>
        </div>
      ))}
    </div>
  )
}
