// Контракт между Python (assistant/ui) и интерфейсом. Имена полей совпадают с Python.

export type VoiceState =
  | 'off'
  | 'idle'
  | 'listening'
  | 'capturing'
  | 'holding'
  | 'recognizing'
  | 'processing'
  | 'paused'

export type Route = 'local' | 'llm'

export interface RecentItem {
  id: number
  time: string // ЧЧ:ММ
  phrase: string
  route: Route
  reply: string
  ok: boolean
  ms: number
  source: 'voice' | 'text'
}

export interface HistoryAction {
  action: string
  args: Record<string, unknown>
  status: string // done, failed, forbidden, cancelled, dry_run
  result: string
  duration_ms: number
}

/** Фраза из журнала: что сказано, куда ушло, какие действия и чем кончилось. */
export interface HistoryItem {
  id: string
  time: string // ISO с часовым поясом
  phrase: string
  source: string // voice, text, hotkey, local (консоль); '' — неизвестно (старый журнал)
  route: Route
  commands: string[]
  reply: string
  ok: boolean
  error: string
  duration_ms: number
  actions: HistoryAction[]
}

export type HistoryFilter = 'all' | 'errors' | 'llm' | 'local'

export interface HistoryQuery {
  limit?: number
  before?: string // id последней показанной фразы
  before_time?: string // её время — если её уже нет в журнале
  filter?: HistoryFilter
  search?: string
}

export interface HistoryPage {
  items: HistoryItem[]
  more: boolean
}

// --- Редактор команд -------------------------------------------------------------

export type SlotType = 'int' | 'float' | 'str' | 'text'

export interface CommandStep {
  action: string
  args: Record<string, unknown> // значения или слоты «{percent}»
}

/** Команда в форме редактора (assistant/commands/editor.py, form_data). */
export interface CommandForm {
  id: string
  phrases: string[]
  slots: Record<string, SlotType>
  when: { active_window_contains: string[]; active_process: string[] }
  confirm: boolean
  reply: string
  hotkey: string
  steps: CommandStep[]
}

export interface CommandSummary {
  id: string
  title: string // первая фраза
  phrases: number
  steps: number
  hotkey: string
  line: number
  active: boolean // работает именно эта команда
  replaced_by: string // её заменяет одноимённая команда из этого пака
  problem: string // почему не загрузилась
}

export interface CommandPack {
  file: string
  user: boolean // свой пак: правится на месте
  error: string // файл не читается
  commands: CommandSummary[]
}

export interface CommandsTree {
  packs: CommandPack[]
  user_file: string
}

export interface CommandDetails {
  ok: boolean
  errors: string[]
  file?: string
  user?: boolean
  command?: CommandForm
  problems?: string[]
  also_in?: string[] // паки, где есть команда с тем же id
}

export interface CommandSaveResult extends CommandsTree {
  ok: boolean
  errors: string[]
  id?: string
}

export interface PhraseTest {
  status: 'matched' | 'ambiguous' | 'not_found' | 'empty'
  reason: string
  normalized: string
  matches: Array<{ id: string; score: number; phrase: string; text: string; slots: Record<string, unknown>; draft: boolean }>
  candidates: Array<{ id: string; score: number; phrase: string }>
  errors: string[] // в черновике ошибки — проверено без него
}

/** JSON-схема параметра действия (то, что встречается в реестре). */
export interface JsonSchema {
  type?: string
  enum?: unknown[]
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  exclusiveMinimum?: number
  minLength?: number
  maxLength?: number
  items?: JsonSchema
  properties?: Record<string, JsonSchema>
  required?: string[]
  additionalProperties?: boolean
}

export interface ActionInfo {
  name: string
  description: string
  risk: 'safe' | 'confirm'
  group: string
  schema: JsonSchema
}

export interface Scenario {
  id: string
  title: string
  phrase: string
  steps: number
}

export interface LlmSummary {
  enabled: boolean
  profile: string
  title: string
  model: string
  ready: boolean
  problem: string // почему не готово: нет ключа и т. п.
}

/** Таймер или напоминание (assistant/timers.py, Alarm.as_dict). */
export interface AlarmItem {
  id: string
  kind: 'timer' | 'reminder'
  due: string // ISO с часовым поясом: когда сработает
  text: string // подпись таймера или о чём напомнить
  seconds: number // на сколько поставлен таймер
}

/** Новая версия — релиз на GitHub (assistant/updates.py, Release). */
export interface UpdateInfo {
  version: string
  title: string
  notes: string // что нового (описание релиза, Markdown)
  page: string // страница релиза
  installer: string
  size: number // байт
  sha256: string
}

/** Обновления (UiController.update_view). */
export interface UpdateState {
  supported: boolean // программа обновляется сама (собранный exe); из исходников — git pull и uv sync
  version: string // эта версия
  release: UpdateInfo | null // новая версия, если нашлась
  checked: boolean // в этом запуске GitHub уже ответил
  checking: boolean
  installing: boolean // скачивается и ставится обновление
  step: string // «Скачиваю обновление: 40% (20 МБ из 50 МБ)»
  progress: number | null // доля сделанного (0–1); null — неизвестна
  error: string
}

export interface AppState {
  name: string
  version: string
  voice: {
    enabled: boolean
    state: VoiceState
    paused: boolean
    wake_word: boolean
    push_to_talk_key: string
    stop_key: string
    loading: boolean // голос ещё загружается
    problem: string
    step: string // что делается, пока голос загружается: «Скачиваю модель Vosk: 40% (18 МБ из 45 МБ)»
    progress: number | null // доля сделанного (0–1); null — неизвестна
  }
  llm: LlmSummary
  dry_run: boolean
  volume: { percent: number | null; muted: boolean | null }
  recent: RecentItem[]
  scenarios: Scenario[]
  timers: AlarmItem[] // ближайшие первыми
  update: UpdateState
  ui: UiSettings
}

export interface UiSettings {
  theme: 'dark' | 'light'
  accent: string
  overlay: boolean
  quick_chat_key: string
}

export interface LlmProfileInfo {
  name: string
  title: string
  provider: 'openai' | 'anthropic'
  model: string
  models: string[] // рекомендуемые модели
  base_url: string
  base_url_editable: boolean // адрес API можно поменять (свой сервис, Ollama)
  configured: boolean // false — профиля ещё нет в конфиге (свой сервис)
  key_env: string // '' — ключ не нужен
  key_set: boolean
  key_hint: string // «••••3f2a»
  key_from_system: boolean // ключ из переменной окружения Windows, а не из .env
  key_optional: boolean // ключ можно не указывать (свой сервис)
  key_url: string // где взять ключ; '' — нигде
  key_example: string // как начинается ключ («sk-ant-api03-…»); '' — по-разному
  note: string
}

export interface Choice {
  value: string
  title: string
}

// piper и silero — офлайн, edge — голоса Microsoft через интернет, sapi — голоса Windows.
export type TtsEngine = 'piper' | 'silero' | 'edge' | 'sapi'

export interface Settings {
  general: {
    name: string
    dry_run: boolean
    check_updates: boolean // раз в сутки проверять, не вышла ли новая версия
    autostart: boolean | null // null — автозапуск недоступен
  }
  voice: {
    enabled: boolean
    wake_word: boolean
    push_to_talk: boolean
    push_to_talk_key: string
    push_to_talk_mode: 'press' | 'hold'
    stop_key: string
    input_device: string
    wake_confidence: number
    followup_seconds: number
    min_silence_ms: number
    sounds: boolean
  }
  stt: { model: string; device: 'auto' | 'cuda' | 'cpu' }
  tts: {
    engine: TtsEngine
    piper_voice: string
    silero_speaker: string
    edge_voice: string
    sapi_voice: string
    rate: number
    volume: number
  }
  llm: { enabled: boolean; profile: string; profiles: LlmProfileInfo[] }
  ui: UiSettings
  choices: {
    microphones: Choice[]
    piper_voices: Choice[]
    sapi_voices: Choice[]
    stt_models: Choice[]
    silero_speakers: Choice[]
    edge_voices: Choice[]
    silero_ready: boolean // PyTorch и модель Silero уже скачаны
  }
  restart_pending: boolean // есть изменения, которые применятся после перезапуска
}

export type SettingsPatch = {
  [K in keyof Omit<Settings, 'choices' | 'llm'>]?: Partial<Settings[K]>
} & { llm?: { enabled?: boolean } }

export interface SaveResult {
  ok: boolean
  errors: string[]
  restart: boolean // часть изменений применится после перезапуска
  settings: Settings | null
}

export interface LlmRequest {
  profile: string
  model: string
  key: string // пусто — оставить сохранённый
  base_url: string // пусто — оставить как есть
}

export interface LlmTestResult {
  ok: boolean
  text: string
  ms: number
  error: string
}

export interface LlmModelsResult {
  ok: boolean
  models: string[]
  error: string
}

export interface Step {
  number: number
  total: number | null
  action: string
  text: string
  ok: boolean
}

export type AssistantEvent =
  | { type: 'state'; payload: AppState }
  | { type: 'voice'; payload: { state: VoiceState; detail: string } }
  | { type: 'voice_progress'; payload: { text: string; fraction: number | null } }
  | { type: 'level'; payload: { value: number } }
  | { type: 'recognized'; payload: { text: string; source?: 'voice' | 'text' | 'hotkey' } }
  | { type: 'step'; payload: Step }
  | { type: 'reply'; payload: { text: string } }
  | { type: 'result'; payload: RecentItem }
  | { type: 'confirm'; payload: { id: string; question: string } }
  | { type: 'confirm_done'; payload: { id: string } }
  | { type: 'commands'; payload: { count: number; issues: string[] } }
  | { type: 'toast'; payload: { kind: 'info' | 'error' | 'alarm'; text: string } } // alarm висит, пока не закроют
  | { type: 'chat_shown'; payload: Record<string, never> }
  | { type: 'update'; payload: UpdateState }

export type WindowName = 'main' | 'overlay' | 'chat'

// Методы Python (класс Api в assistant/ui/api.py), вызываются через window.pywebview.api.
export interface AssistantApi {
  get_state(): Promise<AppState>
  get_settings(): Promise<Settings>
  save_settings(patch: SettingsPatch): Promise<SaveResult>
  llm_test(request: LlmRequest): Promise<LlmTestResult>
  llm_models(request: LlmRequest): Promise<LlmModelsResult>
  llm_save(request: LlmRequest): Promise<SaveResult>
  llm_forget_key(profile: string): Promise<SaveResult>
  open_key_page(profile: string): Promise<void>
  voice_pause(paused: boolean): Promise<AppState>
  voice_listen(): Promise<void>
  send_text(text: string): Promise<void>
  confirm_answer(id: string, yes: boolean): Promise<void>
  set_volume(percent: number): Promise<AppState>
  toggle_mute(): Promise<AppState>
  run_scenario(id: string): Promise<void>
  preview_voice(tts: Settings['tts']): Promise<void>
  level_meter(on: boolean): Promise<void>
  pause_hotkeys(paused: boolean): Promise<void> // пока записывается новое сочетание
  window_action(window: WindowName, action: 'minimize' | 'maximize' | 'close' | 'hide'): Promise<void>
  window_resize(width: number, height: number, seq: number): Promise<void> // главное окно без рамки
  open_path(kind: 'config' | 'log' | 'journal' | 'commands'): Promise<void>
  history(query: HistoryQuery): Promise<HistoryPage>
  commands_tree(): Promise<CommandsTree>
  actions(): Promise<ActionInfo[]>
  command_get(file: string, id: string): Promise<CommandDetails>
  command_save(file: string, originalId: string, form: CommandForm): Promise<CommandSaveResult>
  command_delete(file: string, id: string): Promise<CommandSaveResult>
  command_test(text: string, form: CommandForm | null, originalId: string): Promise<PhraseTest>
  command_run(form: CommandForm, phrase: string): Promise<{ ok: boolean; errors: string[] }>
  reload_commands(): Promise<{ count: number; issues: string[] }>
  timer_cancel(id: string): Promise<AppState>
  update_check(): Promise<UpdateState>
  update_install(): Promise<UpdateState> // программа закроется, установщик запустит новую версию
  update_open_page(): Promise<void>
  restart(): Promise<void>
}
