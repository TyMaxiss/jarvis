// Оверлей — маленькое окно-«пилюля» внизу экрана. Окно показывает и прячет Python;
// здесь только отрисовка событий: «Слушаю…» с волной, распознанный текст, шаги, ответ, «Да / Нет».

import { CircleAlert, CircleCheck, CircleStop } from 'lucide-react'
import { useReducer, useState } from 'react'
import { cx } from '../components/ui'
import { call } from '../lib/bridge'
import { initialOverlay, overlayLines, overlayReduce, type OverlayState } from '../lib/overlay'
import { useAssistantEvent } from '../lib/store'

export function Overlay({ framed = false }: { framed?: boolean }) {
  const [state, dispatch] = useReducer(overlayReduce, initialOverlay)
  const [level, setLevel] = useState(0)
  useAssistantEvent((event) => {
    if (event.type === 'level') setLevel(event.payload.value)
    else dispatch(event)
  })
  const [main, sub] = overlayLines(state)

  return (
    <div
      className={cx(
        'flex items-center gap-3.5 bg-surface px-4',
        // В браузере (?mock) окна нет — рисуем его границы сами.
        framed ? 'h-16 w-[480px] rounded-2xl border border-line-strong shadow-2xl' : 'h-full w-full',
      )}
    >
      <StatusIcon state={state} level={level} />
      <div className="min-w-0 flex-1" aria-live="polite">
        <div className="truncate text-[14px] leading-5 font-medium">{main}</div>
        {sub && <div className="truncate text-[12px] leading-4 text-muted">{sub}</div>}
      </div>
      {state.confirm && (
        <div className="flex shrink-0 gap-1.5">
          <button
            type="button"
            onClick={() => void call('confirm_answer', state.confirm!.id, true)}
            className="h-8 rounded-lg bg-accent px-3.5 text-[13px] font-medium text-white transition hover:brightness-110"
          >
            Да
          </button>
          <button
            type="button"
            onClick={() => void call('confirm_answer', state.confirm!.id, false)}
            className="h-8 rounded-lg bg-surface-3 px-3.5 text-[13px] font-medium transition hover:bg-surface-2"
          >
            Нет
          </button>
        </div>
      )}
    </div>
  )
}

function StatusIcon({ state, level }: { state: OverlayState; level: number }) {
  if (state.mode === 'done' && !state.confirm) return <CircleCheck size={22} className="shrink-0 text-ok" />
  if (state.mode === 'stopped') return <CircleStop size={22} className="shrink-0 text-muted" />
  if (state.mode === 'error' && !state.confirm) return <CircleAlert size={22} className="shrink-0 text-danger" />
  if (state.mode === 'listening') return <Wave level={level} />
  return (
    <span className="relative h-[22px] w-[22px] shrink-0">
      <span className="absolute inset-0 rounded-full border-2 border-surface-3" />
      <span className="animate-spin-slow absolute inset-0 rounded-full border-2 border-transparent border-t-accent" />
    </span>
  )
}

const bars = [0.55, 0.85, 1, 0.75, 0.5]

/** Волна из пяти столбиков: высота следует за громкостью микрофона. */
function Wave({ level }: { level: number }) {
  const loud = Math.min(1, level * 1.6)
  return (
    <span className="flex h-[22px] w-[22px] shrink-0 items-center justify-between" aria-hidden="true">
      {bars.map((weight, index) => (
        <span
          key={index}
          className="w-[3px] rounded-full bg-accent transition-[height] duration-100 ease-out"
          style={{ height: `${Math.round(4 + 18 * loud * weight)}px` }}
        />
      ))}
    </span>
  )
}
