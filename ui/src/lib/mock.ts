// Подделка Python-API для разработки интерфейса в обычном браузере (адрес с ?mock).
// Повторяет поведение настоящего API настолько, чтобы можно было пощёлкать все экраны.

import type {
  AppState,
  AssistantApi,
  AssistantEvent,
  LlmProfileInfo,
  LlmRequest,
  RecentItem,
  Settings,
  SettingsPatch,
} from './types'

const profiles: LlmProfileInfo[] = [
  {
    name: 'deepseek',
    title: 'DeepSeek',
    provider: 'openai',
    model: 'deepseek-flash',
    models: ['deepseek-flash'],
    base_url: 'https://api.deepseek.com',
    base_url_editable: false,
    configured: true,
    key_env: 'DEEPSEEK_API_KEY',
    key_set: true,
    key_hint: '••••3f2a',
    key_optional: false,
    key_url: 'https://platform.deepseek.com/api_keys',
    note: 'Быстрая и недорогая. Ключ работает, только если на счёте есть деньги.',
  },
  {
    name: 'openai',
    title: 'ChatGPT',
    provider: 'openai',
    model: 'gpt-6-luna',
    models: ['gpt-6-luna', 'gpt-5.4-mini', 'gpt-6-sol', 'gpt-5.5'],
    base_url: 'https://api.openai.com/v1',
    base_url_editable: false,
    configured: true,
    key_env: 'OPENAI_API_KEY',
    key_set: false,
    key_hint: '',
    key_optional: false,
    key_url: 'https://platform.openai.com/settings/organization/api-keys',
    note: 'Модели OpenAI. В списке те, что умеют вызывать действия; быстрее всех — `gpt-6-luna`.',
  },
  {
    name: 'anthropic',
    title: 'Claude',
    provider: 'anthropic',
    model: 'claude-opus-5-5',
    models: ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5'],
    base_url: '',
    base_url_editable: false,
    configured: true,
    key_env: 'ANTHROPIC_API_KEY',
    key_set: false,
    key_hint: '',
    key_optional: false,
    key_url: 'https://platform.claude.com/settings/keys',
    note: 'Модели Anthropic. Сложные задачи лучше всех решает `claude-opus-5-5`, быстрее всех отвечает `claude-haiku-5-5`.',
  },
  {
    name: 'ollama',
    title: 'Ollama',
    provider: 'openai',
    model: 'qwen3:8b',
    models: ['qwen3:8b'],
    base_url: 'http://localhost:11434/v1',
    base_url_editable: true,
    configured: true,
    key_env: '',
    key_set: false,
    key_hint: '',
    key_optional: false,
    key_url: '',
    note: 'Работает без интернета на этом компьютере. Нужна установленная Ollama и скачанная модель.',
  },
  {
    name: 'custom',
    title: 'Другой сервис',
    provider: 'openai',
    model: '',
    models: [],
    base_url: '',
    base_url_editable: true,
    configured: false,
    key_env: 'CUSTOM_LLM_API_KEY',
    key_set: false,
    key_hint: '',
    key_optional: true,
    key_url: '',
    note: 'Любой сервис с OpenAI-совместимым API: укажите адрес, модель и ключ.',
  },
]

const recent: RecentItem[] = [
  { id: 3, time: '18:42', phrase: 'Сделай громче', route: 'local', reply: 'Громкость 60', ok: true, ms: 412, source: 'voice' },
  { id: 2, time: '18:40', phrase: 'Создай на рабочем столе папку Проекты', route: 'llm', reply: 'Создал папку «Проекты» на рабочем столе.', ok: true, ms: 2310, source: 'voice' },
  { id: 1, time: '18:31', phrase: 'Открой ютуб', route: 'local', reply: '', ok: true, ms: 380, source: 'text' },
]

function makeState(): AppState {
  return {
    name: 'Джарвис',
    version: '0.4.0',
    voice: {
      enabled: true,
      state: 'idle',
      paused: false,
      wake_word: true,
      push_to_talk_key: 'ctrl+alt+space',
      stop_key: 'ctrl+alt+q',
      problem: '',
    },
    llm: { enabled: true, profile: 'deepseek', title: 'DeepSeek', model: 'deepseek-flash', ready: true, problem: '' },
    dry_run: false,
    volume: { percent: 55, muted: false },
    recent,
    scenarios: [
      { id: 'work_mode', title: 'Рабочий режим', phrase: 'рабочий режим', steps: 3 },
      { id: 'minimize_all', title: 'Свернуть все окна', phrase: 'сверни все окна', steps: 1 },
    ],
    ui: { theme: 'dark', accent: '#7c6cff', overlay: true, quick_chat_key: 'ctrl+alt+enter' },
  }
}

function makeSettings(state: AppState): Settings {
  return {
    general: { name: 'джарвис', dry_run: state.dry_run },
    voice: {
      enabled: true,
      wake_word: true,
      push_to_talk: true,
      push_to_talk_key: 'ctrl+alt+space',
      push_to_talk_mode: 'press',
      stop_key: 'ctrl+alt+q',
      input_device: '',
      wake_confidence: 0.8,
      followup_seconds: 5,
      min_silence_ms: 700,
      sounds: true,
    },
    stt: { model: 'auto', device: 'auto' },
    tts: { engine: 'piper', piper_voice: 'models/piper/ru_RU-dmitri-medium.onnx', sapi_voice: '', rate: 1, volume: 1 },
    llm: { enabled: true, profile: state.llm.profile, profiles },
    ui: state.ui,
    choices: {
      microphones: [
        { value: '', title: 'Микрофон по умолчанию' },
        { value: '1', title: 'Микрофон (Realtek High Definition Audio)' },
        { value: '3', title: 'Гарнитура (USB Audio Device)' },
      ],
      piper_voices: [
        { value: 'models/piper/ru_RU-dmitri-medium.onnx', title: 'Дмитрий' },
        { value: 'models/piper/ru_RU-irina-medium.onnx', title: 'Ирина' },
      ],
      sapi_voices: [
        { value: '', title: 'Первый русский голос' },
        { value: 'Irina', title: 'Microsoft Irina Desktop' },
      ],
      stt_models: [
        { value: 'auto', title: 'Автоматически' },
        { value: 'small', title: 'small — для процессора' },
        { value: 'large-v3-turbo', title: 'large-v3-turbo — для видеокарты' },
      ],
    },
    restart_pending: false,
  }
}

export function createMockApi(emit: (event: AssistantEvent) => void): AssistantApi {
  const state = makeState()
  let settings = makeSettings(state)
  let meter: number | undefined
  const pause = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

  const pushState = () => emit({ type: 'state', payload: structuredClone(state) })

  const simulate = async (text: string) => {
    emit({ type: 'voice', payload: { state: 'processing', detail: '' } })
    emit({ type: 'recognized', payload: { text } })
    await pause(500)
    emit({ type: 'step', payload: { number: 1, total: null, action: 'open_app', text: 'Открываю Блокнот', ok: true } })
    await pause(600)
    const reply = 'Готово, открыл Блокнот.'
    emit({ type: 'reply', payload: { text: reply } })
    const item: RecentItem = {
      id: Date.now(),
      time: new Date().toTimeString().slice(0, 5),
      phrase: text,
      route: 'llm',
      reply,
      ok: true,
      ms: 1100,
      source: 'text',
    }
    state.recent = [item, ...state.recent].slice(0, 20)
    emit({ type: 'result', payload: item })
    emit({ type: 'voice', payload: { state: 'idle', detail: '' } })
  }

  return {
    async get_state() {
      return structuredClone(state)
    },
    async get_settings() {
      return structuredClone(settings)
    },
    async save_settings(patch: SettingsPatch) {
      settings = {
        ...settings,
        general: { ...settings.general, ...patch.general },
        voice: { ...settings.voice, ...patch.voice },
        stt: { ...settings.stt, ...patch.stt },
        tts: { ...settings.tts, ...patch.tts },
        ui: { ...settings.ui, ...patch.ui },
      }
      state.dry_run = settings.general.dry_run
      state.ui = settings.ui
      pushState()
      return { ok: true, errors: [], restart: Boolean(patch.stt), settings: structuredClone(settings) }
    },
    async llm_test(request: LlmRequest) {
      await pause(700)
      const profile = profiles.find((p) => p.name === request.profile)
      if (profile?.key_env && !profile.key_set && !request.key) {
        return { ok: false, text: '', ms: 0, error: `нет ключа API для профиля ${request.profile}` }
      }
      return { ok: true, text: 'Да.', ms: 840, error: '' }
    },
    async llm_models(request: LlmRequest) {
      await pause(400)
      const profile = profiles.find((p) => p.name === request.profile)
      return { ok: true, models: profile?.models ?? [], error: '' }
    },
    async llm_save(request: LlmRequest) {
      const profile = profiles.find((p) => p.name === request.profile)
      if (profile) {
        profile.model = request.model
        profile.configured = true
        if (request.base_url) profile.base_url = request.base_url
        if (request.key) {
          profile.key_set = true
          profile.key_hint = `••••${request.key.slice(-4)}`
        }
        state.llm = { ...state.llm, profile: profile.name, title: profile.title, model: profile.model, ready: true, problem: '' }
        settings = { ...settings, llm: { ...settings.llm, profile: profile.name } }
      }
      pushState()
      return { ok: true, errors: [], restart: false, settings: structuredClone(settings) }
    },
    async llm_forget_key(name: string) {
      const profile = profiles.find((p) => p.name === name)
      if (profile) {
        profile.key_set = false
        profile.key_hint = ''
      }
      return { ok: true, errors: [], restart: false, settings: structuredClone(settings) }
    },
    async open_key_page() {},
    async voice_pause(paused: boolean) {
      state.voice.paused = paused
      state.voice.state = paused ? 'paused' : 'idle'
      pushState()
      return structuredClone(state)
    },
    async voice_listen() {
      emit({ type: 'voice', payload: { state: 'listening', detail: '' } })
    },
    async send_text(text: string) {
      void simulate(text)
    },
    async confirm_answer() {},
    async set_volume(percent: number) {
      state.volume.percent = percent
      return structuredClone(state)
    },
    async toggle_mute() {
      state.volume.muted = !state.volume.muted
      return structuredClone(state)
    },
    async run_scenario(id: string) {
      void simulate(state.scenarios.find((s) => s.id === id)?.phrase ?? id)
    },
    async preview_voice() {},
    async level_meter(on: boolean) {
      window.clearInterval(meter)
      if (on) {
        meter = window.setInterval(
          () => emit({ type: 'level', payload: { value: Math.abs(Math.sin(Date.now() / 300)) * 0.7 } }),
          80,
        )
      }
    },
    async window_action() {},
    async window_resize() {},
    async pause_hotkeys() {},
    async open_path() {},
    async reload_commands() {
      return { count: 42, issues: [] }
    },
    async restart() {},
  }
}
