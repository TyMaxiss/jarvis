import {
  AudioLines,
  ExternalLink,
  Eye,
  EyeOff,
  FileText,
  FolderOpen,
  Keyboard,
  Mic,
  Monitor,
  Play,
  RotateCw,
  ShieldCheck,
  Sparkles,
  Trash2,
  Volume2,
} from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { HotkeyInput, LevelMeter, Segmented, Swatches } from '../components/controls'
import { Button, Card, CardTitle, cx, Field, IconButton, inputClass, Select, Slider, TextInput, Toggle } from '../components/ui'
import { call } from '../lib/bridge'
import type {
  KeyedEngine,
  OverlayStyle,
  SaveResult,
  Settings,
  SettingsPatch,
  TtsEngine,
  TtsKey,
  UpdateState,
} from '../lib/types'
import { LlmSection } from './LlmSection'
import { UpdatesCard } from './Updates'

type Section = 'general' | 'voice' | 'stt' | 'tts' | 'ui'

/** Настройки с сохранением каждого изменения; ползунки сохраняются с задержкой. */
function useSettings() {
  const [settings, setSettings] = useState<Settings | null>(null)
  const [restart, setRestart] = useState(false)
  const [errors, setErrors] = useState<string[]>([])
  const timers = useRef(new Map<string, number>())

  useEffect(() => {
    void call('get_settings').then((loaded) => {
      if (!loaded) return
      setSettings(loaded)
      if (loaded.restart_pending) setRestart(true)
    })
  }, [])

  const apply = useCallback((result: SaveResult | undefined) => {
    if (!result) return
    if (result.ok) {
      setErrors([])
      if (result.restart) setRestart(true)
    } else {
      setErrors(result.errors)
      if (result.settings) setSettings(result.settings)
    }
  }, [])

  const update = useCallback(
    <K extends Section>(section: K, values: Partial<Settings[K]>, delay = 0) => {
      setSettings((prev) => prev && { ...prev, [section]: { ...prev[section], ...values } })
      const key = `${section}:${Object.keys(values).join(',')}`
      window.clearTimeout(timers.current.get(key))
      timers.current.set(
        key,
        window.setTimeout(() => {
          void call('save_settings', { [section]: values } as SettingsPatch).then(apply)
        }, delay),
      )
    },
    [apply],
  )

  return { settings, setSettings, update, apply, restart, errors }
}

/** Настройки; updates — состояние обновлений из AppState (раздел «Обновления»). */
export function SettingsPage({ updates }: { updates: UpdateState }) {
  const { settings, setSettings, update, apply, restart, errors } = useSettings()
  if (!settings) return null
  const { voice, stt, tts, ui, general } = settings
  const seconds = (value: number) => `${value.toLocaleString('ru-RU', { maximumFractionDigits: 1 })} с`

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-5 p-7">
      <h1 className="font-display text-[22px] font-semibold tracking-tight">Настройки</h1>

      {restart && (
        <div className="animate-fade-up flex items-center gap-3 rounded-card border border-accent/30 bg-accent/10 px-4 py-3 text-[13px]">
          <RotateCw size={16} className="shrink-0 text-accent" />
          <span className="flex-1">Часть изменений вступит в силу после перезапуска ассистента.</span>
          <Button variant="primary" onClick={() => void call('restart')}>
            Перезапустить
          </Button>
        </div>
      )}
      {errors.length > 0 && (
        <div role="alert" className="rounded-card border border-danger/30 bg-danger/10 px-4 py-3 text-[13px] text-danger">
          {errors.map((error) => (
            <div key={error}>{error}</div>
          ))}
        </div>
      )}

      <LlmSection
        settings={settings}
        onSettings={setSettings}
        onEnabled={(enabled) => {
          setSettings({ ...settings, llm: { ...settings.llm, enabled } })
          void call('save_settings', { llm: { enabled } }).then(apply)
        }}
      />

      <Card>
        <CardTitle icon={<Mic size={16} />} title="Голос" />
        <Field label="Голосовое управление" hint="Выключите, если нужен только ввод текстом">
          <Toggle label="Голосовое управление" checked={voice.enabled} onChange={(enabled) => update('voice', { enabled })} />
        </Field>
        <Field label="Микрофон" stacked>
          <div className="flex w-full flex-col gap-3">
            <Select
              label="Микрофон"
              className="w-full"
              value={voice.input_device}
              options={settings.choices.microphones}
              onChange={(input_device) => update('voice', { input_device })}
            />
            <LevelMeter />
          </div>
        </Field>
        <Field label="Обращение по имени" hint={`Скажите «${general.name}» и команду`}>
          <Toggle label="Обращение по имени" checked={voice.wake_word} onChange={(wake_word) => update('voice', { wake_word })} />
        </Field>
        <Field label="Имя ассистента" hint="На него он откликается">
          <NameInput value={general.name} onSave={(name) => update('general', { name })} />
        </Field>
        <Field label="Чувствительность к имени" hint="Выше — реже откликается случайно, но может не услышать">
          <div className="flex w-56 items-center gap-3">
            <Slider
              label="Чувствительность к имени"
              min={0.5}
              max={0.95}
              step={0.05}
              value={voice.wake_confidence}
              onChange={(wake_confidence) => update('voice', { wake_confidence }, 400)}
            />
            <span className="w-9 text-right text-[12.5px] text-muted tabular-nums">
              {Math.round(voice.wake_confidence * 100)}%
            </span>
          </div>
        </Field>
        <Field label="Звуковой сигнал" hint="Короткий звук, когда ассистент начинает слушать">
          <Toggle label="Звуковой сигнал" checked={voice.sounds} onChange={(sounds) => update('voice', { sounds })} />
        </Field>
      </Card>

      <Card>
        <CardTitle icon={<Keyboard size={16} />} title="Горячие клавиши" />
        <Field
          label="Слушать команду"
          hint={voice.push_to_talk_mode === 'hold' ? 'Говорите, пока клавиши зажаты' : 'Нажмите и говорите, конец — по паузе'}
        >
          <Toggle
            label="Горячая клавиша для команды"
            checked={voice.push_to_talk}
            onChange={(push_to_talk) => update('voice', { push_to_talk })}
          />
          <HotkeyInput
            label="Слушать команду"
            disabled={!voice.push_to_talk}
            value={voice.push_to_talk_key}
            onChange={(push_to_talk_key) => update('voice', { push_to_talk_key })}
          />
        </Field>
        <Field label="Режим клавиши">
          <Segmented
            label="Режим клавиши"
            disabled={!voice.push_to_talk}
            value={voice.push_to_talk_mode}
            options={[
              { value: 'press', title: 'Нажать' },
              { value: 'hold', title: 'Удерживать' },
            ]}
            onChange={(push_to_talk_mode) => update('voice', { push_to_talk_mode })}
          />
        </Field>
        <Field label="Остановить" hint="Прервать ответ и задачу нейросети">
          <HotkeyInput label="Остановить" value={voice.stop_key} onChange={(stop_key) => update('voice', { stop_key })} />
        </Field>
        <Field label="Быстрый чат" hint="Окно для команды текстом поверх всех окон">
          <HotkeyInput
            label="Быстрый чат"
            value={ui.quick_chat_key}
            onChange={(quick_chat_key) => update('ui', { quick_chat_key })}
          />
        </Field>
      </Card>

      <Card>
        <CardTitle icon={<AudioLines size={16} />} title="Распознавание речи" />
        <Field label="Модель" hint="Крупнее — точнее, но медленнее">
          <Select
            label="Модель распознавания"
            value={stt.model}
            options={settings.choices.stt_models}
            onChange={(model) => update('stt', { model })}
          />
        </Field>
        <Field label="Где считать">
          <Segmented
            label="Устройство распознавания"
            value={stt.device}
            options={[
              { value: 'auto', title: 'Авто' },
              { value: 'cuda', title: 'Видеокарта' },
              { value: 'cpu', title: 'Процессор' },
            ]}
            onChange={(device) => update('stt', { device })}
          />
        </Field>
        <Field label="Пауза в конце фразы" hint="Сколько тишины считать концом команды">
          <div className="flex w-56 items-center gap-3">
            <Slider
              label="Пауза в конце фразы"
              min={300}
              max={2000}
              step={100}
              value={voice.min_silence_ms}
              onChange={(min_silence_ms) => update('voice', { min_silence_ms }, 400)}
            />
            <span className="w-11 text-right text-[12.5px] text-muted tabular-nums">{seconds(voice.min_silence_ms / 1000)}</span>
          </div>
        </Field>
        <Field label="Слушать продолжение" hint="Сколько ждать следующую команду без имени">
          <div className="flex w-56 items-center gap-3">
            <Slider
              label="Слушать продолжение"
              min={0}
              max={15}
              step={1}
              value={voice.followup_seconds}
              onChange={(followup_seconds) => update('voice', { followup_seconds }, 400)}
            />
            <span className="w-11 text-right text-[12.5px] text-muted tabular-nums">{seconds(voice.followup_seconds)}</span>
          </div>
        </Field>
      </Card>

      <Card>
        <CardTitle icon={<Volume2 size={16} />} title="Голос ассистента" />
        <Field label="Синтез речи" hint={engineHint(tts.engine, settings.choices.silero_ready)} stacked>
          <Segmented<TtsEngine>
            label="Синтез речи"
            value={tts.engine}
            options={[
              { value: 'piper', title: 'Piper' },
              { value: 'silero', title: 'Silero' },
              { value: 'edge', title: 'Microsoft' },
              { value: 'sapi', title: 'Windows' },
              { value: 'openai', title: 'OpenAI' },
              { value: 'yandex', title: 'Яндекс' },
            ]}
            onChange={(engine) => update('tts', { engine })}
          />
        </Field>
        <Field label="Голос">
          {tts.engine === 'piper' && (
            <Select
              label="Голос Piper"
              value={tts.piper_voice}
              options={settings.choices.piper_voices}
              onChange={(piper_voice) => update('tts', { piper_voice })}
            />
          )}
          {tts.engine === 'silero' && (
            <Select
              label="Голос Silero"
              value={tts.silero_speaker}
              options={settings.choices.silero_speakers}
              onChange={(silero_speaker) => update('tts', { silero_speaker })}
            />
          )}
          {tts.engine === 'edge' && (
            <Select
              label="Голос Microsoft"
              value={tts.edge_voice}
              options={settings.choices.edge_voices}
              onChange={(edge_voice) => update('tts', { edge_voice })}
            />
          )}
          {tts.engine === 'sapi' && (
            <Select
              label="Голос Windows"
              value={tts.sapi_voice}
              options={settings.choices.sapi_voices}
              onChange={(sapi_voice) => update('tts', { sapi_voice })}
            />
          )}
          {tts.engine === 'openai' && (
            <Select
              label="Голос OpenAI"
              value={tts.openai_voice}
              options={settings.choices.openai_voices}
              onChange={(openai_voice) => update('tts', { openai_voice })}
            />
          )}
          {tts.engine === 'yandex' && (
            <Select
              label="Голос Яндекса"
              value={tts.yandex_voice}
              options={settings.choices.yandex_voices}
              onChange={(yandex_voice) => update('tts', { yandex_voice })}
            />
          )}
          <Button icon={<Play size={14} />} onClick={() => void call('preview_voice', tts)}>
            Прослушать
          </Button>
        </Field>
        {tts.engine === 'openai' && (
          <>
            <Field
              label="Манера речи"
              hint="Как говорить: характер, тон, темп. По умолчанию — вежливый ИИ-дворецкий в духе Джарвиса"
              stacked
            >
              <StyleInput
                value={tts.openai_style}
                fallback={settings.choices.openai_style_default}
                onSave={(openai_style) => update('tts', { openai_style })}
              />
            </Field>
            <Field label="Адрес API" hint="Можно OpenAI-совместимый прокси-сервис">
              <LineInput
                label="Адрес API"
                value={tts.openai_base_url}
                onSave={(openai_base_url) => update('tts', { openai_base_url })}
              />
            </Field>
          </>
        )}
        {(tts.engine === 'openai' || tts.engine === 'yandex') && (
          <Field label={tts.engine === 'openai' ? 'Ключ OpenAI' : 'Ключ Yandex Cloud'} hint={keyHint(tts.engine)} stacked>
            <VoiceKey
              key={tts.engine}
              engine={tts.engine}
              info={settings.tts_keys[tts.engine]}
              onResult={(result) => {
                apply(result)
                if (result?.ok && result.settings) setSettings(result.settings)
              }}
            />
          </Field>
        )}
        <Field label="Скорость речи">
          <div className="flex w-56 items-center gap-3">
            <Slider
              label="Скорость речи"
              min={0.5}
              max={2}
              step={0.1}
              value={tts.rate}
              onChange={(rate) => update('tts', { rate }, 400)}
            />
            <span className="w-11 text-right text-[12.5px] text-muted tabular-nums">
              ×{tts.rate.toLocaleString('ru-RU', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}
            </span>
          </div>
        </Field>
        <Field label="Громкость голоса">
          <div className="flex w-56 items-center gap-3">
            <Slider
              label="Громкость голоса"
              min={0.1}
              max={2}
              step={0.1}
              value={tts.volume}
              onChange={(volume) => update('tts', { volume }, 400)}
            />
            <span className="w-11 text-right text-[12.5px] text-muted tabular-nums">{Math.round(tts.volume * 100)}%</span>
          </div>
        </Field>
      </Card>

      <Card>
        <CardTitle icon={<Monitor size={16} />} title="Интерфейс" />
        <Field label="Тема">
          <Segmented
            label="Тема"
            value={ui.theme}
            options={[
              { value: 'dark', title: 'Тёмная' },
              { value: 'light', title: 'Светлая' },
            ]}
            onChange={(theme) => update('ui', { theme })}
          />
        </Field>
        <Field label="Акцентный цвет">
          <Swatches value={ui.accent} onChange={(accent) => update('ui', { accent })} />
        </Field>
        <Field label="Оверлей" hint="Внизу экрана: что услышал и что делает">
          <Toggle label="Оверлей" checked={ui.overlay} onChange={(overlay) => update('ui', { overlay })} />
        </Field>
        <Field label="Оформление оверлея" hint="Сфера — шар, который отзывается на голос, и карточка с ответом">
          <Segmented<OverlayStyle>
            label="Оформление оверлея"
            value={ui.overlay_style}
            disabled={!ui.overlay}
            options={[
              { value: 'classic', title: 'Плашка' },
              { value: 'orb', title: 'Сфера' },
            ]}
            onChange={(overlay_style) => update('ui', { overlay_style })}
          />
        </Field>
        <Field label="Итог голосовой команды" hint="Виден в оверлее до следующей фразы; клик по оверлею — спрятать">
          <Toggle
            label="Итог голосовой команды"
            checked={ui.overlay_keep}
            disabled={!ui.overlay}
            onChange={(overlay_keep) => update('ui', { overlay_keep })}
          />
        </Field>
        {general.autostart !== null && (
          <Field label="Запускать вместе с Windows" hint="Сразу в трей; ярлык — в папке «Автозагрузка»">
            <Toggle
              label="Запускать вместе с Windows"
              checked={general.autostart}
              onChange={(autostart) => update('general', { autostart })}
            />
          </Field>
        )}
      </Card>

      <Card>
        <CardTitle icon={<ShieldCheck size={16} />} title="Безопасность и файлы" />
        <Field label="Пробный режим (dry-run)" hint="Действия только записываются в журнал, но не выполняются">
          <Toggle label="Пробный режим" checked={general.dry_run} onChange={(dry_run) => update('general', { dry_run })} />
        </Field>
        <Field label="Файлы" hint="Откроются в программе по умолчанию" stacked>
          <div className="flex flex-wrap gap-2">
            <Button icon={<FileText size={14} />} onClick={() => void call('open_path', 'config')}>
              config.yaml
            </Button>
            <Button icon={<FolderOpen size={14} />} onClick={() => void call('open_path', 'commands')}>
              Папка команд
            </Button>
            <Button icon={<FileText size={14} />} onClick={() => void call('open_path', 'log')}>
              Лог
            </Button>
            <Button icon={<FileText size={14} />} onClick={() => void call('open_path', 'journal')}>
              Журнал действий
            </Button>
          </div>
        </Field>
      </Card>

      <UpdatesCard
        update={updates}
        checkUpdates={general.check_updates}
        onCheckUpdates={(check_updates) => update('general', { check_updates })}
      />

      <p className="flex items-center gap-2 pb-2 text-[12px] text-faint">
        <Sparkles size={13} />
        Настройки хранятся в config.yaml, ключи API — в .env рядом с ним.
      </p>
    </div>
  )
}

/** Поле имени: сохраняется по Enter или при уходе с поля, а не на каждую букву. */
function NameInput({ value, onSave }: { value: string; onSave: (value: string) => void }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = () => {
    const name = draft.trim().toLowerCase()
    if (name && name !== value) onSave(name)
    else setDraft(value)
  }
  return (
    <TextInput
      aria-label="Имя ассистента"
      className="w-48"
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === 'Enter') event.currentTarget.blur()
        if (event.key === 'Escape') setDraft(value)
      }}
    />
  )
}

/** Что нужно знать о движке озвучки: офлайн ли он и что скачивается. */
export function engineHint(engine: TtsEngine, sileroReady: boolean): string {
  switch (engine) {
    case 'piper':
      return 'Офлайн и быстро'
    case 'silero':
      return sileroReady
        ? 'Офлайн, нейросеть. Только для некоммерческого использования'
        : 'Офлайн, нейросеть. После перезапуска скачается около 270 МБ. Только для некоммерческого использования'
    case 'edge':
      return (
        'Голоса Microsoft через интернет, бесплатно. Многоязычные — новее и живее, но может слышаться ' +
        'лёгкий акцент. Без интернета говорит запасной голос'
      )
    case 'sapi':
      return 'Голоса, установленные в Windows'
    case 'openai':
      return (
        'Голоса OpenAI через интернет, по ключу: живая речь, манеру задаёте словами. ' +
        'Около 1,5 ₽ за минуту речи. Без интернета и ключа говорит запасной голос'
      )
    case 'yandex':
      return (
        'Yandex SpeechKit через интернет, по ключу Yandex Cloud: родной русский, работает в России. ' +
        'Около 0,2 ₽ за фразу. Без интернета и ключа говорит запасной голос'
      )
  }
}

function keyHint(engine: KeyedEngine) {
  return (
    <>
      {engine === 'openai'
        ? 'Тот же ключ, что у ChatGPT в разделе «Нейросеть». '
        : 'API-ключ сервисного аккаунта с ролью ai.speechkit-tts.user. '}
      <button
        type="button"
        aria-label="Где взять ключ голоса"
        onClick={() => void call('open_tts_key_page', engine)}
        className="inline-flex items-center gap-1 whitespace-nowrap text-accent hover:underline"
      >
        Где взять ключ <ExternalLink size={12} />
      </button>
    </>
  )
}

/** Подсказка в поле ключа голоса: какой ключ уже есть или какой нужен. */
export function voiceKeyPlaceholder(engine: KeyedEngine, info: TtsKey): string {
  if (info.set) {
    return info.from_system
      ? `Из переменной Windows ${info.hint} — вставьте ключ, чтобы сохранить свой`
      : `Сохранён ${info.hint} — вставьте новый, чтобы заменить`
  }
  return engine === 'openai' ? 'Вставьте ключ: sk-…' : 'Вставьте API-ключ Yandex Cloud'
}

/**
 * Ключ голоса по ключу: сохраняется кнопкой в .env; сохранённый можно удалить.
 * У каждого голоса поле своё (key={engine}): ключ Яндекса, вставленный до переключения
 * на OpenAI, не сохранится как ключ OpenAI.
 */
function VoiceKey({
  engine,
  info,
  onResult,
}: {
  engine: KeyedEngine
  info: TtsKey
  onResult: (result: SaveResult | undefined) => void
}) {
  const [draft, setDraft] = useState('')
  const [show, setShow] = useState(false)
  const [busy, setBusy] = useState(false)
  const run = async (action: () => Promise<SaveResult | undefined>) => {
    setBusy(true)
    try {
      const result = await action()
      if (result?.ok) setDraft('')
      onResult(result)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className="flex w-full items-center gap-2">
      <div className="relative flex-1">
        <TextInput
          aria-label="Ключ голоса"
          type={show ? 'text' : 'password'}
          autoComplete="off"
          spellCheck={false}
          className="w-full pr-10 font-mono"
          placeholder={voiceKeyPlaceholder(engine, info)}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && draft.trim()) void run(() => call('tts_save_key', engine, draft.trim()))
          }}
        />
        <IconButton label={show ? 'Скрыть ключ' : 'Показать ключ'} className="absolute top-0.5 right-0.5" onClick={() => setShow(!show)}>
          {show ? <EyeOff size={15} /> : <Eye size={15} />}
        </IconButton>
      </div>
      <Button
        aria-label="Сохранить ключ голоса"
        disabled={busy || !draft.trim()}
        onClick={() => void run(() => call('tts_save_key', engine, draft.trim()))}
      >
        Сохранить
      </Button>
      {info.set && !info.from_system && (
        <IconButton label="Удалить сохранённый ключ" disabled={busy} onClick={() => void run(() => call('tts_forget_key', engine))}>
          <Trash2 size={15} />
        </IconButton>
      )}
    </div>
  )
}

/** Манера речи: сохраняется при уходе с поля; «Как у Джарвиса» возвращает описание по умолчанию. */
function StyleInput({ value, fallback, onSave }: { value: string; fallback: string; onSave: (value: string) => void }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  return (
    <div className="flex w-full flex-col gap-2">
      <textarea
        aria-label="Манера речи"
        rows={6}
        spellCheck={false}
        className={cx(inputClass, 'h-auto w-full resize-y py-2 leading-relaxed')}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={() => draft.trim() !== value.trim() && onSave(draft.trim())}
      />
      {draft.trim() !== fallback && (
        <div>
          <Button
            icon={<RotateCw size={14} />}
            onClick={() => {
              setDraft(fallback)
              onSave(fallback)
            }}
          >
            Как у Джарвиса
          </Button>
        </div>
      )}
    </div>
  )
}

/** Строка, которая сохраняется по Enter или при уходе с поля. */
function LineInput({ label, value, onSave }: { label: string; value: string; onSave: (value: string) => void }) {
  const [draft, setDraft] = useState(value)
  useEffect(() => setDraft(value), [value])
  const commit = () => (draft.trim() && draft.trim() !== value ? onSave(draft.trim()) : setDraft(value))
  return (
    <TextInput
      aria-label={label}
      spellCheck={false}
      className="w-72"
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => event.key === 'Enter' && commit()}
    />
  )
}
