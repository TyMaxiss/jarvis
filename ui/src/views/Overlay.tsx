// Оверлей внизу экрана в одном из двух оформлений: плашка («пилюля») или сфера (Orb.tsx).
// Окно показывает и прячет Python; здесь только отрисовка событий: «Слушаю…» с волной, слова
// по ходу речи, распознанный текст, шаги, ответ, «Да / Нет». Клик по оверлею его прячет.

import { CircleAlert, CircleCheck, CircleStop, MicOff } from 'lucide-react'
import { useEffect, useReducer, useRef, useState } from 'react'
import { cx } from '../components/ui'
import { call } from '../lib/bridge'
import { initialOverlay, overlayLines, overlayReduce, shown, type OverlayState } from '../lib/overlay'
import { useAppState, useAssistantEvent } from '../lib/store'
import { Orb } from './Orb'

const STYLE_SWITCH_MS = 300 // дольше, чем оверлей уезжает (slide.py, HIDE_TIME)

export function Overlay({ framed = false }: { framed?: boolean }) {
  const app = useAppState()
  const [state, dispatch] = useReducer(overlayReduce, initialOverlay)
  const [level, setLevel] = useState(0)
  const [present, setPresent] = useState(false)
  // Прозрачность окна сферы решает Python: показывая — какая она сейчас, пряча — какой будет при
  // следующем показе. Спрятанная сфера заранее становится такой, какой появится.
  const [glass, setGlass] = useState(false)
  const [nextGlass, setNextGlass] = useState(false)
  const [snap, setSnap] = useState(false) // сфера мгновенно стала невидимой и ждёт кадра, чтобы раскрыться
  const turn = useRef(0) // номер события overlay: отложенное появление не переживает следующего
  const showing = useRef(false) // Python показал оверлей
  const hiddenAt = useRef(0) // когда показанный оверлей начал уезжать
  useAssistantEvent((event) => {
    if (event.type === 'level') setLevel(event.payload.value)
    else {
      if (event.type === 'overlay') {
        const { visible, glass: next = false } = event.payload
        const mine = ++turn.current
        if (showing.current && !visible) hiddenAt.current = Date.now()
        showing.current = visible
        setNextGlass(next)
        const switching = visible && next !== glass
        setSnap(switching)
        if (switching) {
          // Прозрачность сменилась, пока сфера была спрятана: сначала она сразу невидимая в новом
          // виде (orb-snap), и лишь кадром позже раскрывается — иначе переходу (index.css, .orb-pop)
          // не от чего начаться.
          setGlass(next)
          afterFrames(2, () => {
            if (turn.current !== mine) return
            setSnap(false)
            setPresent(true)
          })
        } else setPresent(visible)
      }
      dispatch(event)
    }
  })
  // Оформление меняется, только пока оверлей спрятан: окно под новое оформление Python
  // перестроит при следующем показе. Уезжающее окно ещё прежней формы — ждём, пока оно уедет.
  const wanted = app?.ui.overlay_style ?? 'classic'
  const [style, setStyle] = useState(wanted)
  useEffect(() => {
    if (present || (style === wanted && glass === nextGlass)) return
    const apply = () => {
      setStyle(wanted)
      setGlass(nextGlass)
    }
    const wait = hiddenAt.current + STYLE_SWITCH_MS - Date.now()
    if (wait <= 0) {
      apply()
      return
    }
    const timer = window.setTimeout(apply, wait)
    return () => window.clearTimeout(timer)
  }, [wanted, nextGlass, present, style, glass])
  if (style === 'orb') return <Orb state={state} level={level} present={present} glass={glass} snap={snap} framed={framed} />
  return <Pill state={state} level={level} framed={framed} />
}

/** Вызывает action через count кадров: к тому времени браузер успел отрисовать прежние изменения. */
function afterFrames(count: number, action: () => void) {
  window.requestAnimationFrame(() => (count > 1 ? afterFrames(count - 1, action) : action()))
}

/** Плашка: значок состояния, две строки и кнопки «Да / Нет». */
function Pill({ state, level, framed }: { state: OverlayState; level: number; framed: boolean }) {
  const [main, sub] = overlayLines(state)

  return (
    <div
      // Клик прячет плашку (и итог команды, который она держит), если она не ждёт «Да / Нет».
      onClick={state.confirm ? undefined : () => void call('overlay_dismiss')}
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

function StatusIcon({ state: input, level }: { state: OverlayState; level: number }) {
  const state = shown(input)
  if (state.mode === 'done' && !state.confirm) return <CircleCheck size={22} className="shrink-0 text-ok" />
  if (state.mode === 'stopped') return <CircleStop size={22} className="shrink-0 text-muted" />
  if (state.mode === 'error' && !state.confirm) return <CircleAlert size={22} className="shrink-0 text-danger" />
  if (state.mode === 'idle') return <MicOff size={22} className="shrink-0 text-muted" />
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
