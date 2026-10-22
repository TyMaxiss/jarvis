// Оверлей «Сфера»: шар из синих слоёв дышит с голосом, под ним — карточка: что происходит, какие
// слова слышно, что услышано и чем кончилось. Окно обрезано по кругу шара и карточке (windows.py,
// ORB_SHAPE) и появляется на своём месте, раскрываясь из центра шара; внутренние слои при этом
// «распускаются» по событию overlay. С «Эффектами прозрачности» Windows (glass) внешнее кольцо и
// карточка полупрозрачные, а растёт из центра сама страница (.orb-pop); без них сферу кругом из
// центра открывает окно.

import { Check, CircleAlert, LoaderCircle, Mic, MicOff, ShieldQuestionMark, Square, X } from 'lucide-react'
import { useEffect, useRef, type CSSProperties, type ReactNode, type RefObject } from 'react'
import { cx } from '../components/ui'
import { call } from '../lib/bridge'
import { orbCard, type OrbCard, type Outcome, type OverlayState } from '../lib/overlay'

// Внешний слой — неподвижный диск по форме окна (с glass — полупрозрачный). Внутренние слои снаружи
// внутрь: размер (доля шара), цвет, насколько слой растёт с голосом, период перетекания (с) и сдвиг от
// центра (пиксели) — слои не концентричны, как на эскизе. Цвета непрозрачные: это синий #4486ff
// с непрозрачностью 0.42, 0.64, 0.8 и 1 поверх диска #1e2f51 — сквозь кольцо видно рабочий стол,
// сквозь внутренние слои — нет.
const LAYERS = [
  { size: 0.82, color: 'rgb(46 84 154)', gain: 0.09, period: 19, x: -5, y: 3, reverse: true },
  { size: 0.64, color: 'rgb(60 116 219)', gain: 0.14, period: 16, x: 6, y: 1 },
  { size: 0.46, color: 'rgb(66 130 248)', gain: 0.21, period: 13, x: 9, y: -3, reverse: true },
  { size: 0.34, color: 'rgb(68 134 255)', gain: 0.3, period: 11, x: 5, y: -6 },
]
const DOTS = 12

interface Props {
  state: OverlayState
  level: number // громкость микрофона 0..1
  present: boolean // окно показано (событие overlay)
  glass?: boolean // окно смешивается с рабочим столом: кольцо и карточка полупрозрачные
  snap?: boolean // сфера сразу, без перехода, невидимая — сейчас начнёт раскрываться
  framed?: boolean // в браузере (?mock) — границы окна пунктиром
}

export function Orb({ state, level, present, glass = false, snap = false, framed = false }: Props) {
  const stage = useRef<HTMLDivElement>(null)
  const voice = !state.confirm && (state.mode === 'listening' || state.mode === 'recognizing')
  useSmoothLevel(stage, voice ? Math.min(1, level * 1.6) : 0)

  const card = orbCard(state)
  return (
    <div
      ref={stage}
      className={cx(
        'orb-stage',
        present && 'orb-present',
        glass && 'orb-glass',
        snap && 'orb-snap',
        framed && 'orb-framed',
      )}
      data-mode={card.badge}
      // Клик прячет сферу (и итог команды, который она держит), если она не ждёт «Да / Нет».
      onClick={state.confirm ? undefined : () => void call('overlay_dismiss')}
    >
      <div className="orb-pop">
        <div className="orb" aria-hidden="true">
          <span className="orb-disc" />
          <div className="orb-bloom">
            <div className="orb-breath">
              {LAYERS.map((layer, index) => (
                <span
                  key={index}
                  className="orb-layer"
                  style={
                    {
                      '--size': `${layer.size * 100}%`,
                      '--color': layer.color,
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
        </div>
        <div className="orb-card">
          <div className="flex items-center gap-3">
            <Badge mode={card.badge} />
            <div className="min-w-0 flex-1 truncate text-[14px] text-muted">{card.status}</div>
            <Dots state={voice ? 'voice' : card.badge === 'working' ? 'working' : 'still'} />
          </div>
          {state.confirm ? (
            <div className="mt-auto flex items-end gap-3">
              <div aria-live="polite" className="line-clamp-2 min-w-0 flex-1 text-[15px] leading-5 font-semibold text-text">
                {card.text}
              </div>
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
            <>
              <Phrase card={card} />
              {card.line && <OutcomeLine outcome={card.line} />}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

/** Крупно — фраза: слова, которые ещё говорят (за ними мигает курсор), или фраза команды. */
function Phrase({ card }: { card: OrbCard }) {
  return (
    <div aria-live="polite" className="orb-phrase">
      <span className="truncate">{card.text}</span>
      {card.live && <span className="orb-caret" aria-hidden="true" />}
    </div>
  )
}

/** Синий кружок слева: микрофон, пока слушает; затем ход и итог команды. */
function Badge({ mode }: { mode: OrbCard['badge'] }) {
  const icons: Record<OrbCard['badge'], ReactNode> = {
    listening: <Mic size={15} strokeWidth={2.2} />,
    recognizing: <Mic size={15} strokeWidth={2.2} />,
    working: <LoaderCircle size={15} strokeWidth={2.4} className="animate-spin-slow" />,
    done: <Check size={15} strokeWidth={2.6} />,
    error: <X size={15} strokeWidth={2.6} />,
    stopped: <Square size={11} strokeWidth={0} fill="currentColor" />,
    idle: <MicOff size={15} strokeWidth={2.2} />,
    confirm: <ShieldQuestionMark size={15} strokeWidth={2.2} />,
  }
  const tone = mode === 'error' ? 'error' : mode === 'stopped' || mode === 'idle' ? 'muted' : 'blue'
  return (
    <span className="orb-badge" data-tone={tone}>
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
        'mt-auto flex items-start gap-2.5 text-[14.5px] leading-5',
        outcome.tone === 'ok' ? 'text-ok' : outcome.tone === 'error' ? 'text-danger' : 'text-muted',
      )}
    >
      {icon && <span className="mt-[1.5px] shrink-0">{icon}</span>}
      <span className="min-w-0 truncate">{outcome.text}</span>
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
