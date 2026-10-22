import {
  Bell,
  CircleAlert,
  History,
  Mic,
  MicOff,
  Pause,
  Play,
  Sparkles,
  Square,
  Timer,
  Volume2,
  VolumeX,
  X,
  Zap,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Badge, Button, Card, CardTitle, IconButton, Keys, ProgressBar, Slider, cx } from '../components/ui'
import { call } from '../lib/bridge'
import { countdown, durationText, formatMs, isListening, momentText, plural, voiceLabel } from '../lib/format'
import type { AlarmItem, AppState, RecentItem } from '../lib/types'
import { UpdateBanner } from './Updates'

export function Home({ state, onOpenSettings }: { state: AppState; onOpenSettings: () => void }) {
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-5 p-7">
      {state.update.release && <UpdateBanner update={state.update} />}
      <Hero state={state} />
      {state.timers.length > 0 && <TimersCard timers={state.timers} />}
      <div className="grid grid-cols-3 gap-5">
        <VolumeCard state={state} />
        <LlmCard state={state} onOpenSettings={onOpenSettings} />
        <ScenariosCard state={state} />
      </div>
      <RecentCard items={state.recent} />
    </div>
  )
}

function Hero({ state }: { state: AppState }) {
  const voice = state.voice
  const listening = isListening(voice.state)
  const working = voice.state === 'recognizing' || voice.state === 'processing'
  const busy = voice.loading || working
  // Пока ассистент слушает, выполняет задачу или говорит, кнопка его останавливает (как Ctrl+Alt+Q).
  const stoppable = listening || working || state.busy
  const label = stoppable ? 'Остановить' : 'Слушать команду'
  return (
    <Card className="flex items-center gap-7 p-7">
      <button
        type="button"
        aria-label={label}
        title={label}
        disabled={!stoppable && (!voice.enabled || voice.paused)}
        onClick={() => void call(stoppable ? 'voice_stop' : 'voice_listen')}
        className={cx(
          'relative flex h-[88px] w-[88px] shrink-0 items-center justify-center rounded-full transition duration-200 ease-out',
          listening ? 'bg-accent text-white shadow-[0_0_40px_-6px_var(--accent)]' : 'bg-surface-2 text-text hover:bg-surface-3',
          'disabled:cursor-default disabled:opacity-60',
        )}
      >
        {listening && <span className="animate-pulse-ring absolute inset-0 rounded-full border-2 border-accent" />}
        {busy && <span className="animate-spin-slow absolute inset-1 rounded-full border-2 border-transparent border-t-accent" />}
        {stoppable ? (
          <Square size={24} fill="currentColor" />
        ) : voice.enabled && !voice.paused ? (
          <Mic size={30} />
        ) : (
          <MicOff size={30} />
        )}
      </button>
      <div className="min-w-0 flex-1">
        <h1 className="font-display text-[22px] font-semibold tracking-tight">{voiceLabel(state)}</h1>
        {voice.enabled ? (
          <p className="mt-1.5 text-[13px] leading-relaxed text-muted">
            {voice.wake_word && `Скажите «${state.name}, …»`}
            {voice.wake_word && voice.push_to_talk_key && ' или нажмите '}
            {!voice.wake_word && voice.push_to_talk_key && 'Нажмите '}
            {voice.push_to_talk_key && <Keys value={voice.push_to_talk_key} />}. Остановить — этой же кнопкой,{' '}
            <Keys value={voice.stop_key} /> или «стоп».
          </p>
        ) : voice.loading ? (
          <LoadingNote step={voice.step} progress={voice.progress} />
        ) : (
          <p className="mt-1.5 flex items-center gap-2 text-[13px] text-warn">
            <CircleAlert size={15} />
            {voice.problem || 'Голос выключен в настройках'}
          </p>
        )}
      </div>
      {voice.enabled && (
        <Button
          variant="subtle"
          icon={voice.paused ? <Play size={15} /> : <Pause size={15} />}
          onClick={() => void call('voice_pause', !voice.paused)}
        >
          {voice.paused ? 'Продолжить' : 'Пауза'}
        </Button>
      )}
    </Card>
  )
}

/** Что делается, пока голос загружается: при первом запуске модели скачиваются минутами. */
function LoadingNote({ step, progress }: { step: string; progress: number | null }) {
  const percent = progress === null ? null : Math.round(progress * 100)
  return (
    <div className="mt-1.5 max-w-xl text-[13px] text-muted">
      <p className="truncate">{step || 'Загружаю распознавание речи — это несколько секунд.'}</p>
      {percent !== null && <ProgressBar label="Ход загрузки" value={percent} className="mt-2" />}
      <p className="mt-1.5 text-faint">
        {percent !== null && 'Это нужно только при первом запуске. '}Команды текстом уже работают.
      </p>
    </div>
  )
}

function VolumeCard({ state }: { state: AppState }) {
  const [value, setValue] = useState(state.volume.percent ?? 0)
  const timer = useRef<number | undefined>(undefined)
  useEffect(() => setValue(state.volume.percent ?? 0), [state.volume.percent])
  const available = state.volume.percent !== null
  const change = (percent: number) => {
    setValue(percent)
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => void call('set_volume', percent), 120)
  }
  return (
    <Card>
      <CardTitle
        icon={<Volume2 size={16} />}
        title="Громкость"
        action={
          available && (
            <IconButton label={state.volume.muted ? 'Включить звук' : 'Выключить звук'} onClick={() => void call('toggle_mute')}>
              {state.volume.muted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            </IconButton>
          )
        }
      />
      {available ? (
        <>
          <div className={cx('font-display text-[30px] font-semibold tabular-nums', state.volume.muted && 'text-faint')}>
            {value}
            <span className="ml-0.5 text-[16px] text-muted">%</span>
          </div>
          <Slider className="mt-4" label="Громкость" min={0} max={100} value={value} onChange={change} />
        </>
      ) : (
        <p className="text-[13px] text-muted">Недоступна в этой системе</p>
      )}
    </Card>
  )
}

function LlmCard({ state, onOpenSettings }: { state: AppState; onOpenSettings: () => void }) {
  const llm = state.llm
  return (
    <Card className="flex flex-col">
      <CardTitle icon={<Sparkles size={16} />} title="Нейросеть" />
      {llm.enabled ? (
        <>
          <div className="font-display text-[19px] font-semibold">{llm.title}</div>
          <div className="mt-0.5 truncate text-[12.5px] text-muted">{llm.model}</div>
          <div className="mt-3 flex items-center gap-2 text-[12.5px]">
            <span className={cx('h-1.5 w-1.5 shrink-0 rounded-full', llm.ready ? 'bg-ok' : 'bg-warn')} />
            <span className={cx('truncate', llm.ready ? 'text-muted' : 'text-warn')}>{llm.ready ? 'Готова' : llm.problem}</span>
          </div>
        </>
      ) : (
        <p className="text-[13px] text-muted">Выключена: работают только команды</p>
      )}
      <div className="flex-1" />
      <Button className="mt-4 self-start" variant="ghost" onClick={onOpenSettings}>
        Настроить
      </Button>
    </Card>
  )
}

function ScenariosCard({ state }: { state: AppState }) {
  return (
    <Card className="flex flex-col">
      <CardTitle icon={<Zap size={16} />} title="Быстрые сценарии" />
      {state.scenarios.length ? (
        <ul className="-mx-2 flex flex-col gap-0.5">
          {state.scenarios.slice(0, 4).map((scenario) => (
            <li key={scenario.id}>
              <button
                type="button"
                onClick={() => void call('run_scenario', scenario.id)}
                className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] transition-colors hover:bg-surface-2"
              >
                <Play size={13} className="shrink-0 text-accent" />
                <span className="flex-1 truncate">{scenario.title}</span>
                {scenario.steps > 1 && (
                  <span className="text-[11.5px] text-faint">
                    {scenario.steps} {plural(scenario.steps, ['шаг', 'шага', 'шагов'])}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[12.5px] leading-snug text-muted">
          Сценарий — команда из нескольких шагов. Добавьте свой в commands/my.yaml.
        </p>
      )}
    </Card>
  )
}

/** Текущее время, обновляется раз в interval мс. */
function useNow(interval: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), interval)
    return () => window.clearInterval(timer)
  }, [interval])
  return now
}

export function TimersCard({ timers }: { timers: AlarmItem[] }) {
  const now = useNow(1000)
  return (
    <Card>
      <CardTitle icon={<Timer size={16} />} title="Таймеры и напоминания" />
      <ul className="-mx-2 flex flex-col gap-0.5">
        {timers.map((alarm) => (
          <TimerRow key={alarm.id} alarm={alarm} now={now} />
        ))}
      </ul>
    </Card>
  )
}

function TimerRow({ alarm, now }: { alarm: AlarmItem; now: number }) {
  const due = Date.parse(alarm.due)
  const left = Math.max(0, due - now)
  const timer = alarm.kind === 'timer'
  const title = timer ? alarm.text || 'Таймер' : alarm.text
  const detail = timer ? `на ${durationText(alarm.seconds)}` : momentText(new Date(due), new Date(now))
  const done = timer && alarm.seconds > 0 ? Math.min(1, 1 - left / (alarm.seconds * 1000)) : 0
  return (
    <li className="animate-fade-up flex items-center gap-3 rounded-lg px-2 py-2 hover:bg-surface-2/60">
      <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-surface-2 text-accent">
        {timer ? <Timer size={15} /> : <Bell size={15} />}
      </span>
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span className="truncate text-[13.5px]">{title}</span>
          <span className="shrink-0 text-[12px] text-muted">{detail}</span>
        </div>
        {timer && (
          <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-surface-3">
            <div
              className="h-full rounded-full bg-accent transition-[width] duration-1000 ease-linear"
              style={{ width: `${done * 100}%` }}
            />
          </div>
        )}
      </div>
      {left < 86_400_000 && (
        <span className="shrink-0 font-display text-[17px] font-semibold tabular-nums" aria-label="Осталось">
          {countdown(left)}
        </span>
      )}
      <IconButton
        label={timer ? 'Отменить таймер' : 'Отменить напоминание'}
        onClick={() => void call('timer_cancel', alarm.id)}
      >
        <X size={15} />
      </IconButton>
    </li>
  )
}

function RecentCard({ items }: { items: RecentItem[] }) {
  return (
    <Card>
      <CardTitle icon={<History size={16} />} title="Последние команды" />
      {items.length ? (
        <ul className="-mx-2">
          {items.slice(0, 8).map((item) => (
            <li key={item.id} className="animate-fade-up flex items-start gap-3 rounded-lg px-2 py-2 hover:bg-surface-2/60">
              <span className="mt-0.5 w-20 shrink-0">
                <Badge tone={!item.ok ? 'danger' : item.route === 'llm' ? 'accent' : 'neutral'}>
                  {!item.ok ? 'ошибка' : item.route === 'llm' ? 'нейросеть' : 'локально'}
                </Badge>
              </span>
              <div className="min-w-0 flex-1">
                <div className="truncate text-[13.5px]">{item.phrase}</div>
                {item.reply && <div className="mt-0.5 truncate text-[12.5px] text-muted">{item.reply}</div>}
              </div>
              <div className="shrink-0 text-right text-[11.5px] leading-5 text-faint tabular-nums">
                {item.time}
                <br />
                {formatMs(item.ms)}
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-[13px] text-muted">Здесь появятся ваши команды.</p>
      )}
    </Card>
  )
}
