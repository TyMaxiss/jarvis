// Оверлей «Сфера»: шар из полупрозрачных синих слоёв дышит с голосом, под ним — карточка:
// что происходит, что услышано и чем кончилось. Окно прозрачное, появление и уход рисует страница
// по событию overlay; окно Python прячет, когда уход доигран.

import { Check, CircleAlert, LoaderCircle, Mic, ShieldQuestionMark, Square, X } from 'lucide-react'
import { useEffect, useRef, type CSSProperties, type ReactNode, type RefObject } from 'react'
import { cx } from '../components/ui'
import { call } from '../lib/bridge'
import { orbCard, type Outcome, type OverlayState } from '../lib/overlay'

// Слои шара снаружи внутрь: размер (доля шара), непрозрачность синего, насколько слой растёт с голосом,
// период перетекания (с) и сдвиг от центра (пиксели) — слои не концентричны, как на эскизе.
const LAYERS = [
  { size: 1, alpha: 0.26, gain: 0.03, period: 23, x: 0, y: 0 },
  { size: 0.82, alpha: 0.42, gain: 0.09, period: 19, x: -5, y: 3, reverse: true },
  { size: 0.64, alpha: 0.64, gain: 0.14, period: 16, x: 6, y: 1 },
  { size: 0.46, alpha: 0.8, gain: 0.21, period: 13, x: 9, y: -3, reverse: true },
  { size: 0.34, alpha: 1, gain: 0.3, period: 11, x: 5, y: -6 },
]
const DOTS = 12

interface Props {
  state: OverlayState
  level: number // громкость микрофона 0..1
  present: boolean // окно показано (событие overlay)
  framed?: boolean // в браузере (?mock) — границы окна пунктиром
}

export function Orb({ state, level, present, framed = false }: Props) {
  const stage = useRef<HTMLDivElement>(null)
  const voice = !state.confirm && (state.mode === 'listening' || state.mode === 'recognizing')
  useSmoothLevel(stage, voice ? Math.min(1, level * 1.6) : 0)

  const card = orbCard(state)
  const mode = state.confirm ? 'confirm' : state.mode
  return (
    <div
      ref={stage}
      className={cx('orb-stage', present && 'orb-present', framed && 'orb-framed')}
      data-mode={mode}
    >
      <div className="orb" aria-hidden="true">
        <div className="orb-breath">
          {LAYERS.map((layer, index) => (
            <span
              key={index}
              className="orb-layer"
              style={
                {
                  '--size': `${layer.size * 100}%`,
                  '--alpha': layer.alpha,
                  '--gain': layer.gain,
                  '--x': `${layer.x}px`,
                  '--y': `${layer.y}px`,
                  animationDuration: `${layer.period}s`,
                  animationDelay: `${-index * 3.7}s`, // слои не в одной фазе с первого кадра
                  animationDirection: layer.reverse ? 'reverse' : 'normal',
                } as CSSProperties
              }
            />
          ))}
        </div>
      </div>
      <div className="orb-card">
        <div className="flex items-center gap-3">
          <Badge mode={mode} />
          <div aria-live="polite" className={cx('min-w-0 flex-1 truncate text-[14px]', card.heard ? 'font-medium text-text' : 'text-muted')}>
            {card.title}
          </div>
          <Dots state={voice ? 'voice' : mode === 'working' ? 'working' : 'still'} />
        </div>
        {state.confirm ? (
          <div className="flex items-end gap-3">
            <div className="line-clamp-2 min-w-0 flex-1 text-[14px] leading-5 font-medium text-text">{state.confirm.question}</div>
            <div className="flex shrink-0 gap-1.5">
              <button
                type="button"
                onClick={() => void call('confirm_answer', state.confirm!.id, true)}
                className="h-8 rounded-lg bg-[#4486ff] px-3.5 text-[13px] font-medium text-white transition hover:brightness-110"
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
          </div>
        ) : (
          card.line && <OutcomeLine outcome={card.line} />
        )}
      </div>
    </div>
  )
}

/** Синий кружок слева: микрофон, пока слушает; затем ход и итог команды. */
function Badge({ mode }: { mode: string }) {
  const icons: Record<string, ReactNode> = {
    listening: <Mic size={15} strokeWidth={2.2} />,
    recognizing: <Mic size={15} strokeWidth={2.2} />,
    working: <LoaderCircle size={15} strokeWidth={2.4} className="animate-spin-slow" />,
    done: <Check size={15} strokeWidth={2.6} />,
    error: <X size={15} strokeWidth={2.6} />,
    stopped: <Square size={11} strokeWidth={0} fill="currentColor" />,
    confirm: <ShieldQuestionMark size={15} strokeWidth={2.2} />,
  }
  return (
    <span className="orb-badge" data-tone={mode === 'error' ? 'error' : mode === 'stopped' ? 'muted' : 'blue'}>
      {icons[mode]}
    </span>
  )
}

/** Ряд точек справа: пока говорят — пляшут под голос, пока выполняется — бегут волной. */
function Dots({ state }: { state: 'voice' | 'working' | 'still' }) {
  return (
    <span className="orb-dots" data-state={state} aria-hidden="true">
      {Array.from({ length: DOTS }, (_, index) => (
        <i key={index} style={{ '--i': index } as CSSProperties} />
      ))}
    </span>
  )
}

function OutcomeLine({ outcome }: { outcome: Outcome }) {
  const icon =
    outcome.tone === 'ok' ? (
      <Check size={17} strokeWidth={2.4} />
    ) : outcome.tone === 'error' ? (
      <CircleAlert size={16} strokeWidth={2.2} />
    ) : null
  return (
    <div
      className={cx(
        'flex items-start gap-2.5 text-[14.5px] leading-5',
        outcome.tone === 'ok' ? 'text-ok' : outcome.tone === 'error' ? 'text-danger' : 'text-muted',
      )}
    >
      {icon && <span className="mt-[1.5px] shrink-0">{icon}</span>}
      <span className="line-clamp-3 min-w-0">{outcome.text}</span>
    </div>
  )
}

/** Громкость — в CSS-переменную --level каждый кадр: события приходят 14 раз в секунду, а шар
 * должен двигаться плавно — быстро растёт со звуком и мягко опадает. */
function useSmoothLevel(element: RefObject<HTMLDivElement | null>, target: number) {
  const goal = useRef(target)
  useEffect(() => {
    goal.current = target
  })
  useEffect(() => {
    if (typeof requestAnimationFrame !== 'function') return
    let current = 0
    let frame = 0
    const step = () => {
      const delta = goal.current - current
      const next = Math.abs(delta) < 0.002 ? goal.current : current + delta * (delta > 0 ? 0.35 : 0.1)
      if (next !== current) {
        current = next
        element.current?.style.setProperty('--level', current.toFixed(3))
      }
      frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [element])
}
