// Подделка Python-API для разработки интерфейса в обычном браузере (адрес с ?mock).
// Повторяет поведение настоящего API настолько, чтобы можно было пощёлкать все экраны.

import type {
  AppState,
  AssistantApi,
  AssistantEvent,
  HistoryItem,
  HistoryQuery,
  LlmProfileInfo,
  LlmRequest,
  RecentItem,
  Settings,
  SettingsPatch,
} from './types'
import { createMockCommands } from './mockCommands'

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
    key_from_system: false,
    key_optional: false,
    key_url: 'https://platform.deepseek.com/api_keys',
    key_example: 'sk-…',
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
    key_from_system: false,
    key_optional: false,
    key_url: 'https://platform.openai.com/settings/organization/api-keys',
    key_example: 'sk-proj-…',
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
    key_from_system: false,
    key_optional: false,
    key_url: 'https://platform.claude.com/settings/keys',
    key_example: 'sk-ant-api03-…',
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
    key_from_system: false,
    key_optional: false,
    key_url: '',
    key_example: '',
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
    key_from_system: false,
    key_optional: true,
    key_url: '',
    key_example: '',
    note: 'Любой сервис с OpenAI-совместимым API: укажите адрес, модель и ключ.',
  },
]

const recent: RecentItem[] = [
  { id: 3, time: '18:42', phrase: 'Сделай громче', route: 'local', reply: 'Громкость 60', ok: true, ms: 412, source: 'voice' },
  { id: 2, time: '18:40', phrase: 'Создай на рабочем столе папку Проекты', route: 'llm', reply: 'Создал папку «Проекты» на рабочем столе.', ok: true, ms: 2310, source: 'voice' },
  { id: 1, time: '18:31', phrase: 'Открой ютуб', route: 'local', reply: '', ok: true, ms: 380, source: 'text' },
]

/** Время «столько-то минут назад» в формате журнала (ISO с часовым поясом). */
function ago(minutes: number): string {
  const date = new Date(Date.now() - minutes * 60_000)
  const offset = -date.getTimezoneOffset()
  const sign = offset >= 0 ? '+' : '-'
  const pad = (n: number) => String(Math.trunc(Math.abs(n))).padStart(2, '0')
  const local = new Date(date.getTime() + offset * 60_000).toISOString().slice(0, 23)
  return `${local}${sign}${pad(offset / 60)}:${pad(offset % 60)}`
}

function makeHistory(): HistoryItem[] {
  const done = (action: string, args: Record<string, unknown>, result: string, ms = 120) => ({
    action,
    args,
    status: 'done',
    result,
    duration_ms: ms,
  })
  const item = (fields: Partial<HistoryItem> & Pick<HistoryItem, 'id' | 'time' | 'phrase'>): HistoryItem => ({
    source: 'voice',
    route: 'local',
    commands: [],
    reply: '',
    ok: true,
    error: '',
    duration_ms: 400,
    actions: [],
    ...fields,
  })
  return [
    item({
      id: 'h1',
      time: ago(3),
      phrase: 'сделай громче',
      commands: ['volume_up'],
      reply: 'Громкость 60',
      actions: [done('volume_change', { delta: 10 }, 'Громкость 60', 35)],
    }),
    item({
      id: 'h2',
      time: ago(5),
      phrase: 'создай на рабочем столе папку Проекты и открой её',
      route: 'llm',
      reply: 'Создал папку «Проекты» на рабочем столе и открыл её.',
      duration_ms: 2310,
      actions: [
        done('create_folder', { path: 'Рабочий стол\\Проекты' }, 'Папка создана: C:\\Users\\user\\Desktop\\Проекты'),
        done('open_folder', { path: 'C:\\Users\\user\\Desktop\\Проекты' }, 'Открываю Проекты'),
      ],
    }),
    item({
      id: 'h3',
      time: ago(14),
      phrase: 'открой телеграм',
      source: 'text',
      commands: ['open_app'],
      ok: false,
      error: 'Не нашёл приложение «телеграм»',
      actions: [
        { action: 'open_app', args: { name: 'телеграм' }, status: 'failed', result: 'Не нашёл приложение «телеграм»', duration_ms: 210 },
      ],
    }),
    item({
      id: 'h4',
      time: ago(31),
      phrase: 'расскажи анекдот',
      route: 'llm',
      duration_ms: 1840,
      reply: 'Программист заходит в бар… и заказывает 1 пиво, 0 пива, 999999 пив и −1 пиво.',
    }),
    item({
      id: 'h5',
      time: ago(40),
      phrase: 'выключи компьютер',
      commands: ['shutdown'],
      ok: false,
      error: 'Отменено',
      reply: 'Отменено',
      actions: [{ action: 'shutdown', args: {}, status: 'cancelled', result: 'не подтверждено', duration_ms: 0 }],
    }),
    item({
      id: 'h6',
      time: ago(60 * 26),
      phrase: 'рабочий режим',
      source: 'hotkey',
      commands: ['work_mode'],
      reply: 'Все системы готовы',
      duration_ms: 1520,
      actions: [
        done('open_app', { name: 'Telegram' }, 'Открываю Telegram', 640),
        done('open_app', { name: 'Visual Studio Code' }, 'Открываю Visual Studio Code', 710),
        done('volume_set', { percent: 30 }, 'Громкость 30', 40),
      ],
    }),
    item({
      id: 'h7',
      time: ago(60 * 27),
      phrase: 'сколько места свободно на диске C',
      route: 'llm',
      source: 'text',
      reply: 'На диске C свободно 182 ГБ из 476.',
      duration_ms: 3900,
      actions: [done('run_shell', { command: 'Get-PSDrive C' }, 'Free: 195 421 380 608', 880)],
    }),
  ]
}

/** Время через столько-то секунд — ISO, как его отдаёт Python. */
function later(seconds: number): string {
  return new Date(Date.now() + seconds * 1000).toISOString()
}

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
      loading: false,
      problem: '',
      step: '',
      progress: null,
    },
    llm: { enabled: true, profile: 'deepseek', title: 'DeepSeek', model: 'deepseek-flash', ready: true, problem: '' },
    dry_run: false,
    volume: { percent: 55, muted: false },
    recent,
    scenarios: [
      { id: 'work_mode', title: 'Рабочий режим', phrase: 'рабочий режим', steps: 3 },
      { id: 'minimize_all', title: 'Свернуть все окна', phrase: 'сверни все окна', steps: 1 },
    ],
    timers: [
      { id: 't1', kind: 'timer', due: later(4 * 60 + 12), text: 'чай', seconds: 300 },
      { id: 'r1', kind: 'reminder', due: later(2 * 3600), text: 'позвонить маме', seconds: 0 },
    ],
    ui: { theme: 'dark', accent: '#7c6cff', overlay: true, quick_chat_key: 'ctrl+alt+enter' },
  }
}

function makeSettings(state: AppState): Settings {
  return {
    general: { name: 'джарвис', dry_run: state.dry_run, autostart: false },
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
    tts: {
      engine: 'piper',
      piper_voice: 'models/piper/ru_RU-dmitri-medium.onnx',
      silero_speaker: 'aidar',
      edge_voice: 'ru-RU-DmitryNeural',
      sapi_voice: '',
      rate: 1,
      volume: 1,
    },
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
      silero_speakers: [
        { value: 'aidar', title: 'Айдар' },
        { value: 'eugene', title: 'Евгений' },
        { value: 'baya', title: 'Бая' },
        { value: 'kseniya', title: 'Ксения' },
        { value: 'xenia', title: 'Ксения (xenia)' },
      ],
      edge_voices: [
        { value: 'ru-RU-DmitryNeural', title: 'Дмитрий' },
        { value: 'ru-RU-SvetlanaNeural', title: 'Светлана' },
      ],
      silero_ready: false,
    },
    restart_pending: false,
  }
}

export function createMockApi(emit: (event: AssistantEvent) => void): AssistantApi {
  const state = makeState()
  const history = makeHistory()
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
    history.unshift({
      id: `m${item.id}`,
      time: ago(0),
      phrase: text,
      source: 'text',
      route: 'llm',
      commands: [],
      reply,
      ok: true,
      error: '',
      duration_ms: 1100,
      actions: [{ action: 'open_app', args: { name: 'Блокнот' }, status: 'done', result: 'Открываю Блокнот', duration_ms: 300 }],
    })
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
    ...createMockCommands((phrase) => void simulate(phrase)),
    async history(query: HistoryQuery) {
      let items = history
      const index = query.before ? items.findIndex((item) => item.id === query.before) : -1
      if (index >= 0) items = items.slice(index + 1)
      const needle = (query.search ?? '').trim().toLowerCase()
      items = items.filter((item) => {
        if (query.filter === 'errors' && item.ok) return false
        if ((query.filter === 'llm' || query.filter === 'local') && item.route !== query.filter) return false
        return !needle || [item.phrase, item.reply, item.error].some((text) => text.toLowerCase().includes(needle))
      })
      const limit = query.limit ?? 50
      return structuredClone({ items: items.slice(0, limit), more: items.length > limit })
    },
    async reload_commands() {
      return { count: 42, issues: [] }
    },
    async timer_cancel(id: string) {
      state.timers = state.timers.filter((alarm) => alarm.id !== id)
      pushState()
      return structuredClone(state)
    },
    async restart() {},
  }
}
