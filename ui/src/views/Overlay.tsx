// Оверлей внизу экрана в одном из двух оформлений: плашка («пилюля») или сфера (Orb.tsx).
// Окно показывает и прячет Python; здесь только отрисовка событий: «Слушаю…» с волной,
// распознанный текст, шаги, ответ, «Да / Нет».

import { CircleAlert, CircleCheck, CircleStop } from 'lucide-react'
import { useEffect, useReducer, useState } from 'react'
import { cx } from '../components/ui'
import { call } from '../lib/bridge'
import { initialOverlay, overlayLines, overlayReduce, type OverlayState } from '../lib/overlay'
import { useAppState, useAssistantEvent } from '../lib/store'
import { Orb } from './Orb'

export function Overlay({ framed = false }: { framed?: boolean }) {
  const app = useAppState()
  const [state, dispatch] = useReducer(overlayReduce, initialOverlay)
  const [level, setLevel] = useState(0)
  const [present, setPresent] = useState(false)
  useAssistantEvent((event) => {
    if (event.type === 'level') setLevel(event.payload.value)
    else {
      if (event.type === 'overlay') setPresent(event.payload.visible)
      dispatch(event)
    }
  })
  // Оформление меняется, только пока оверлей спрятан: окно под новое оформление Python
  // перестроит при следующем показе.
  const wanted = app?.ui.overlay_style ?? 'classic'
  const [style, setStyle] = useState(wanted)
  useEffect(() => {
    if (!present) setStyle(wanted)
  }, [wanted, present])
  if (style === 'orb') return <Orb state={state} level={level} present={present} framed={framed} />
  return <Pill state={state} level={level} framed={framed} />
}

/** Плашка: значок состояния, две строки и кнопки «Да / Нет». */
function Pill({ state, level, framed }: { state: OverlayState; level: number; framed: boolean }) {
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
