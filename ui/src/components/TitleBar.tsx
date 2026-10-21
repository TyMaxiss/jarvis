// Своя шапка окна без рамки: область перетаскивания и кнопки «свернуть / развернуть / закрыть».
// Окно тянется за элементы с классом pywebview-drag-region (только за них самих: кнопки не тянут).

import { Copy, Minus, Square, X } from 'lucide-react'
import { useState, type PointerEvent as ReactPointerEvent } from 'react'
import { call } from '../lib/bridge'
import { cx } from './ui'

const drag = 'pywebview-drag-region'
let resizeSeq = 0 // номер запроса: Python применяет только самый свежий размер

export function TitleBar({ title, dryRun }: { title: string; dryRun: boolean }) {
  const [maximized, setMaximized] = useState(false)
  const toggleMaximized = () => {
    setMaximized(!maximized)
    void call('window_action', 'main', 'maximize')
  }
  const buttonClass =
    'flex h-10 w-12 items-center justify-center text-muted transition-colors duration-150 hover:bg-surface-2 hover:text-text'

  return (
    <header
      className={cx(drag, 'flex h-10 shrink-0 items-center border-b border-line pl-4')}
      onDoubleClick={(event) => {
        if ((event.target as HTMLElement).classList.contains(drag)) toggleMaximized()
      }}
    >
      <span className={cx(drag, 'mr-2.5 h-2 w-2 rounded-full bg-accent shadow-[0_0_10px_var(--accent)]')} />
      <span className={cx(drag, 'text-[12.5px] font-medium tracking-wide text-muted')}>{title}</span>
      {dryRun && (
        <span className="ml-3 rounded-md bg-warn/12 px-1.5 py-0.5 text-[11px] font-medium text-warn">dry-run</span>
      )}
      <div className={cx(drag, 'h-full flex-1')} />
      <button type="button" aria-label="Свернуть" className={buttonClass} onClick={() => void call('window_action', 'main', 'minimize')}>
        <Minus size={15} />
      </button>
      <button type="button" aria-label={maximized ? 'Восстановить' : 'Развернуть'} className={buttonClass} onClick={toggleMaximized}>
        {maximized ? <Copy size={13} /> : <Square size={13} />}
      </button>
      <button
        type="button"
        aria-label="Закрыть (свернуть в трей)"
        className={cx(buttonClass, 'hover:bg-danger hover:text-white')}
        onClick={() => void call('window_action', 'main', 'close')}
      >
        <X size={16} />
      </button>
    </header>
  )
}

/** Уголок для изменения размера окна без рамки: у таких окон нет системных краёв. */
export function ResizeGrip() {
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    const grip = event.currentTarget
    grip.setPointerCapture(event.pointerId)
    const start = { x: event.screenX, y: event.screenY, width: window.innerWidth, height: window.innerHeight }
    let frame = 0
    const move = (moved: PointerEvent) => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(() => {
        resizeSeq += 1
        const width = Math.round(start.width + moved.screenX - start.x)
        const height = Math.round(start.height + moved.screenY - start.y)
        void call('window_resize', width, height, resizeSeq)
      })
    }
    const stop = () => {
      grip.removeEventListener('pointermove', move)
      grip.removeEventListener('pointerup', stop)
      grip.removeEventListener('pointercancel', stop)
    }
    grip.addEventListener('pointermove', move)
    grip.addEventListener('pointerup', stop)
    grip.addEventListener('pointercancel', stop)
  }
  return (
    <div
      aria-hidden="true"
      onPointerDown={onPointerDown}
      className="fixed right-0 bottom-0 z-50 h-4 w-4 cursor-nwse-resize"
      style={{
        background: 'linear-gradient(135deg, transparent 50%, var(--line-strong) 50%, var(--line-strong) 60%, transparent 60%, transparent 70%, var(--line-strong) 70%, var(--line-strong) 80%, transparent 80%)',
      }}
    />
  )
}
