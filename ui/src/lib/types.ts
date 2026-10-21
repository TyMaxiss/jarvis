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
  }
  llm: LlmSummary
  dry_run: boolean
  volume: { percent: number | null; muted: boolean | null }
  recent: RecentItem[]
  scenarios: Scenario[]
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

export interface Settings {
  general: { name: string; dry_run: boolean }
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
  tts: { engine: 'piper' | 'sapi'; piper_voice: string; sapi_voice: string; rate: number; volume: number }
  llm: { enabled: boolean; profile: string; profiles: LlmProfileInfo[] }
  ui: UiSettings
  choices: { microphones: Choice[]; piper_voices: Choice[]; sapi_voices: Choice[]; stt_models: Choice[] }
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
  | { type: 'level'; payload: { value: number } }
  | { type: 'recognized'; payload: { text: string; source?: 'voice' | 'text' | 'hotkey' } }
  | { type: 'step'; payload: Step }
  | { type: 'reply'; payload: { text: string } }
  | { type: 'result'; payload: RecentItem }
  | { type: 'confirm'; payload: { id: string; question: string } }
  | { type: 'confirm_done'; payload: { id: string } }
  | { type: 'commands'; payload: { count: number; issues: string[] } }
  | { type: 'toast'; payload: { kind: 'info' | 'error'; text: string } }
  | { type: 'chat_shown'; payload: Record<string, never> }

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
  reload_commands(): Promise<{ count: number; issues: string[] }>
  restart(): Promise<void>
}
