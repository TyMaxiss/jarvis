// Нейросеть: выбор провайдера, ключ API, модель, проверка связи и сохранение.
// Ключ уходит в Python только при проверке или сохранении и пишется в .env; обратно приходит лишь «••••3f2a».

import { CircleAlert, CircleCheck, ExternalLink, Eye, EyeOff, RefreshCw, Sparkles, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Button, Card, CardTitle, IconButton, TextInput, Toggle, cx, inputClass } from '../components/ui'
import { call } from '../lib/bridge'
import { formatMs } from '../lib/format'
import type { LlmProfileInfo, LlmRequest, Settings } from '../lib/types'

const OTHER_MODEL = '__other__'

interface Draft {
  model: string
  key: string
  base_url: string
  otherModel: boolean // модель вводится вручную
}

type Status =
  | { kind: 'idle' }
  | { kind: 'busy'; action: 'test' | 'save' | 'models' | 'forget' }
  | { kind: 'ok'; text: string }
  | { kind: 'error'; text: string }

function initialDraft(profile: LlmProfileInfo): Draft {
  return { model: profile.model, key: '', base_url: profile.base_url, otherModel: false }
}

/** Чего не хватает, чтобы проверить или сохранить профиль; пусто — всё заполнено. */
export function missingField(profile: LlmProfileInfo, draft: Draft): string {
  if (profile.base_url_editable && !draft.base_url.trim()) return 'Укажите адрес API'
  if (!draft.model.trim()) return 'Выберите модель'
  if (profile.key_env && !profile.key_optional && !profile.key_set && !draft.key.trim()) return 'Вставьте ключ API'
  return ''
}

export function LlmSection({
  settings,
  onSettings,
  onEnabled,
}: {
  settings: Settings
  onSettings: (settings: Settings) => void
  onEnabled: (enabled: boolean) => void
}) {
  const llm = settings.llm
  const [selected, setSelected] = useState(llm.profile)
  const [drafts, setDrafts] = useState<Record<string, Draft>>({})
  const [serverModels, setServerModels] = useState<Record<string, string[]>>({})
  const [status, setStatus] = useState<Status>({ kind: 'idle' })
  const [showKey, setShowKey] = useState(false)
  const busy = status.kind === 'busy'

  const profile = llm.profiles.find((item) => item.name === selected) ?? llm.profiles[0]
  const draft = drafts[profile.name] ?? initialDraft(profile)
  const setDraft = (values: Partial<Draft>) => {
    setDrafts((prev) => ({ ...prev, [profile.name]: { ...draft, ...values } }))
    if (status.kind !== 'busy') setStatus({ kind: 'idle' })
  }
  const choose = (name: string) => {
    if (busy || name === selected) return
    setSelected(name)
    setShowKey(false)
    setStatus({ kind: 'idle' })
  }

  const request = (): LlmRequest => ({
    profile: profile.name,
    model: draft.model.trim(),
    key: draft.key.trim(),
    base_url: profile.base_url_editable ? draft.base_url.trim() : '',
  })
  const missing = missingField(profile, draft)
  const active = profile.name === llm.profile
  const dirty =
    !active ||
    draft.model.trim() !== profile.model ||
    draft.key.trim() !== '' ||
    (profile.base_url_editable && draft.base_url.trim() !== profile.base_url)

  const test = async () => {
    setStatus({ kind: 'busy', action: 'test' })
    const result = await call('llm_test', request())
    if (!result) return setStatus({ kind: 'error', text: 'Не удалось связаться с ассистентом' })
    setStatus(
      result.ok
        ? { kind: 'ok', text: `Работает: ответ за ${formatMs(result.ms)}` }
        : { kind: 'error', text: result.error },
    )
  }

  const loadModels = async () => {
    setStatus({ kind: 'busy', action: 'models' })
    const result = await call('llm_models', request())
    if (!result) return setStatus({ kind: 'error', text: 'Не удалось связаться с ассистентом' })
    if (!result.ok) return setStatus({ kind: 'error', text: result.error })
    setServerModels((prev) => ({ ...prev, [profile.name]: result.models }))
    setStatus({ kind: 'ok', text: `Моделей по ключу: ${result.models.length}` })
  }

  const save = async () => {
    setStatus({ kind: 'busy', action: 'save' })
    const result = await call('llm_save', request())
    if (!result) return setStatus({ kind: 'error', text: 'Не удалось связаться с ассистентом' })
    if (!result.ok || !result.settings) return setStatus({ kind: 'error', text: result.errors.join('; ') })
    onSettings(result.settings)
    setDrafts((prev) => {
      const next = { ...prev }
      delete next[profile.name]
      return next
    })
    setShowKey(false)
    const saved = result.settings.llm.profiles.find((item) => item.name === profile.name)
    setStatus({ kind: 'ok', text: `Сохранено. Отвечает ${saved?.title ?? profile.title} · ${saved?.model ?? draft.model}` })
  }

  const forgetKey = async () => {
    setStatus({ kind: 'busy', action: 'forget' })
    const result = await call('llm_forget_key', profile.name)
    if (!result) return setStatus({ kind: 'error', text: 'Не удалось связаться с ассистентом' })
    if (!result.ok || !result.settings) return setStatus({ kind: 'error', text: result.errors.join('; ') })
    onSettings(result.settings)
    setStatus({ kind: 'ok', text: 'Ключ удалён с этого компьютера' })
  }

  const recommended = profile.models
  const fromServer = (serverModels[profile.name] ?? []).filter((model) => !recommended.includes(model))
  const known = [...recommended, ...fromServer]
  const manual = draft.otherModel || (!known.length && profile.base_url_editable)

  return (
    <Card>
      <CardTitle
        icon={<Sparkles size={16} />}
        title="Нейросеть"
        action={<Toggle label="Нейросеть" checked={llm.enabled} onChange={onEnabled} />}
      />
      <p className="-mt-1 mb-4 text-[12.5px] leading-relaxed text-muted">
        Отвечает на фразы, которых нет среди команд, и выполняет задачи из нескольких шагов. Ключ хранится только на этом
        компьютере (файл .env) и не попадает в логи.
      </p>

      <div role="radiogroup" aria-label="Провайдер" className="grid grid-cols-3 gap-2">
        {llm.profiles.map((item) => (
          <button
            key={item.name}
            type="button"
            role="radio"
            aria-checked={item.name === profile.name}
            onClick={() => choose(item.name)}
            className={cx(
              'relative flex flex-col items-start gap-1 rounded-xl border px-3 py-2.5 text-left transition duration-150',
              item.name === profile.name
                ? 'border-accent/60 bg-accent/10'
                : 'border-line bg-surface-2 hover:border-line-strong hover:bg-surface-3',
            )}
          >
            <span className="flex w-full items-center gap-2 text-[13.5px] font-medium">
              <span className="truncate">{item.title}</span>
              {item.name === llm.profile && (
                <span className="ml-auto rounded-md bg-accent/20 px-1.5 text-[10.5px] font-medium text-accent">сейчас</span>
              )}
            </span>
            <span className="flex items-center gap-1.5 text-[11.5px] text-muted">
              <span
                className={cx(
                  'h-1.5 w-1.5 rounded-full',
                  item.key_set ? 'bg-ok' : !item.key_env || item.key_optional ? 'bg-faint' : 'bg-warn',
                )}
              />
              {item.key_set
                ? item.key_from_system
                  ? 'ключ из Windows'
                  : 'ключ сохранён'
                : !item.key_env
                  ? 'без ключа'
                  : item.key_optional
                    ? 'ключ — если нужен'
                    : 'нужен ключ'}
            </span>
          </button>
        ))}
      </div>

      <div className="mt-4 flex flex-col gap-3.5 rounded-xl border border-line bg-bg/40 p-4">
        <p className="text-[12.5px] leading-relaxed text-pretty text-muted">
          <Note text={profile.note} />
          {profile.key_url && (
            <button
              type="button"
              onClick={() => void call('open_key_page', profile.name)}
              className="ml-1.5 inline-flex items-center gap-1 whitespace-nowrap text-accent hover:underline"
            >
              Где взять ключ <ExternalLink size={12} />
            </button>
          )}
        </p>

        {profile.base_url_editable && (
          <label className="flex flex-col gap-1.5">
            <span className="text-[12.5px] text-muted">Адрес API</span>
            <TextInput
              aria-label="Адрес API"
              placeholder="https://…/v1"
              spellCheck={false}
              value={draft.base_url}
              onChange={(event) => setDraft({ base_url: event.target.value })}
            />
          </label>
        )}

        {profile.key_env && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[12.5px] text-muted">Ключ API</span>
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <TextInput
                  aria-label="Ключ API"
                  type={showKey ? 'text' : 'password'}
                  autoComplete="off"
                  spellCheck={false}
                  className="w-full pr-10 font-mono"
                  placeholder={keyPlaceholder(profile)}
                  value={draft.key}
                  onChange={(event) => setDraft({ key: event.target.value })}
                />
                <IconButton
                  label={showKey ? 'Скрыть ключ' : 'Показать ключ'}
                  className="absolute top-0.5 right-0.5"
                  onClick={() => setShowKey(!showKey)}
                >
                  {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
                </IconButton>
              </div>
              {profile.key_set && (
                <IconButton label="Удалить сохранённый ключ" disabled={busy} onClick={() => void forgetKey()}>
                  <Trash2 size={15} />
                </IconButton>
              )}
            </div>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <span className="text-[12.5px] text-muted">Модель</span>
          <div className="flex items-center gap-2">
            {manual ? (
              <TextInput
                aria-label="Модель"
                className="flex-1"
                placeholder="Название модели"
                spellCheck={false}
                value={draft.model}
                onChange={(event) => setDraft({ model: event.target.value })}
              />
            ) : (
              <select
                aria-label="Модель"
                className={cx(inputClass, 'flex-1 cursor-pointer')}
                value={draft.model || OTHER_MODEL}
                onChange={(event) =>
                  event.target.value === OTHER_MODEL
                    ? setDraft({ otherModel: true, model: '' })
                    : setDraft({ model: event.target.value })
                }
              >
                {draft.model && !known.includes(draft.model) && <option value={draft.model}>{draft.model}</option>}
                {recommended.length > 0 && (
                  <optgroup label="Рекомендуемые">
                    {recommended.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </optgroup>
                )}
                {fromServer.length > 0 && (
                  <optgroup label="Доступные по ключу">
                    {fromServer.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))}
                  </optgroup>
                )}
                <option value={OTHER_MODEL}>Другая модель…</option>
              </select>
            )}
            {manual && known.length > 0 && (
              <Button variant="ghost" onClick={() => setDraft({ otherModel: false, model: profile.model })}>
                Из списка
              </Button>
            )}
            <IconButton
              label="Загрузить список моделей с сервера"
              disabled={busy || Boolean(missingField(profile, { ...draft, model: draft.model || '-' }))}
              onClick={() => void loadModels()}
            >
              <RefreshCw size={15} className={status.kind === 'busy' && status.action === 'models' ? 'animate-spin-slow' : ''} />
            </IconButton>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 pt-1">
          <Button
            disabled={Boolean(missing) || busy}
            busy={status.kind === 'busy' && status.action === 'test'}
            onClick={() => void test()}
          >
            Проверить
          </Button>
          <Button
            variant="primary"
            disabled={Boolean(missing) || busy || !dirty}
            busy={status.kind === 'busy' && status.action === 'save'}
            onClick={() => void save()}
          >
            {active ? 'Сохранить' : `Использовать ${profile.title}`}
          </Button>
          <StatusLine status={status} hint={missing} />
        </div>
      </div>
    </Card>
  )
}

/** Подсказка в поле ключа: какой ключ уже есть или как выглядит нужный. */
export function keyPlaceholder(profile: LlmProfileInfo): string {
  if (profile.key_set) {
    return profile.key_from_system
      ? `Из переменной Windows ${profile.key_hint} — вставьте ключ, чтобы сохранить свой`
      : `Сохранён ${profile.key_hint} — вставьте новый, чтобы заменить`
  }
  if (profile.key_optional) return 'Ключ, если сервису он нужен'
  return profile.key_example ? `Вставьте ключ: ${profile.key_example}` : 'Вставьте ключ'
}

/** Пояснение к провайдеру: названия моделей в `обратных кавычках` не переносятся по дефису. */
export function Note({ text }: { text: string }) {
  return (
    <>
      {text.split('`').map((part, index) =>
        index % 2 ? (
          <span key={index} className="whitespace-nowrap text-text">
            {part}
          </span>
        ) : (
          part
        ),
      )}
    </>
  )
}

function StatusLine({ status, hint }: { status: Status; hint: string }) {
  if (status.kind === 'ok') {
    return (
      <span role="status" className="animate-fade-up flex items-center gap-1.5 text-[12.5px] text-ok">
        <CircleCheck size={14} className="shrink-0" />
        {status.text}
      </span>
    )
  }
  if (status.kind === 'error') {
    return (
      <span role="alert" className="animate-fade-up flex min-w-0 flex-1 items-start gap-1.5 text-[12.5px] leading-snug text-danger">
        <CircleAlert size={14} className="mt-px shrink-0" />
        <span className="min-w-0 break-words">{status.text}</span>
      </span>
    )
  }
  if (status.kind === 'idle' && hint) return <span className="text-[12.5px] text-faint">{hint}</span>
  return null
}
