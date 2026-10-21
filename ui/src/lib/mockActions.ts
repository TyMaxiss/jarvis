// Действия для «+ Действие» в ?mock — снимок настоящего каталога (assistant/commands/editor.py).

import type { ActionInfo } from './types'

export const mockActions: ActionInfo[] = [
  {
    name: 'open_app',
    description: 'Открыть (запустить) программу по названию',
    risk: 'safe',
    group: 'Программы',
    schema: {
      additionalProperties: false,
      properties: {
        name: {
          description: 'Название программы, например Telegram',
          minLength: 1,
          type: 'string',
        },
      },
      required: ['name'],
      type: 'object',
    },
  },
  {
    name: 'close_app',
    description: 'Закрыть программу по названию. force=true принудительно завершает процесс (несохранённые данные пропадут)',
    risk: 'safe',
    group: 'Программы',
    schema: {
      additionalProperties: false,
      properties: {
        name: {
          description: 'Название программы',
          minLength: 1,
          type: 'string',
        },
        force: {
          default: false,
          description: 'Принудительно завершить процесс',
          type: 'boolean',
        },
      },
      required: ['name'],
      type: 'object',
    },
  },
  {
    name: 'window_action',
    description: 'Действие с активным окном: свернуть, развернуть, восстановить, закрыть; minimize_all сворачивает все окна',
    risk: 'safe',
    group: 'Окна',
    schema: {
      additionalProperties: false,
      properties: {
        action: {
          description: 'minimize, maximize, restore, close или minimize_all',
          enum: ['minimize', 'maximize', 'restore', 'close', 'minimize_all'],
          type: 'string',
        },
      },
      required: ['action'],
      type: 'object',
    },
  },
  {
    name: 'hotkey',
    description: 'Нажать сочетание клавиш в активном окне, например win+d, ctrl+shift+esc, alt+f4, k',
    risk: 'safe',
    group: 'Клавиатура и мышь',
    schema: {
      additionalProperties: false,
      properties: {
        keys: {
          description: 'Клавиши через +, например ctrl+c или win+d',
          type: 'string',
        },
      },
      required: ['keys'],
      type: 'object',
    },
  },
  {
    name: 'type_text',
    description: 'Напечатать текст в активном окне (поддерживается кириллица)',
    risk: 'safe',
    group: 'Клавиатура и мышь',
    schema: {
      additionalProperties: false,
      properties: {
        text: {
          description: 'Текст для ввода',
          maxLength: 5000,
          minLength: 1,
          type: 'string',
        },
      },
      required: ['text'],
      type: 'object',
    },
  },
  {
    name: 'volume_set',
    description: 'Установить громкость системы в процентах (0–100)',
    risk: 'safe',
    group: 'Звук',
    schema: {
      additionalProperties: false,
      properties: {
        percent: {
          description: 'Громкость в процентах',
          maximum: 100,
          minimum: 0,
          type: 'integer',
        },
      },
      required: ['percent'],
      type: 'object',
    },
  },
  {
    name: 'volume_change',
    description: 'Изменить громкость на указанное число процентов (минус — тише)',
    risk: 'safe',
    group: 'Звук',
    schema: {
      additionalProperties: false,
      properties: {
        delta: {
          description: 'На сколько процентов: 10 или -10',
          maximum: 100,
          minimum: -100,
          type: 'integer',
        },
      },
      required: ['delta'],
      type: 'object',
    },
  },
  {
    name: 'mute',
    description: 'Выключить или включить звук; без параметра — переключить',
    risk: 'safe',
    group: 'Звук',
    schema: {
      additionalProperties: false,
      properties: {
        state: {
          default: null,
          description: 'true — выключить звук, false — включить, не указано — переключить',
          type: 'boolean',
        },
      },
      type: 'object',
    },
  },
  {
    name: 'media',
    description: 'Управление воспроизведением музыки и видео: пауза/продолжить, следующий, предыдущий трек, стоп',
    risk: 'safe',
    group: 'Музыка и видео',
    schema: {
      additionalProperties: false,
      properties: {
        action: {
          description: 'play_pause — пауза или продолжить, next, prev, stop',
          enum: ['play_pause', 'next', 'prev', 'stop'],
          type: 'string',
        },
      },
      required: ['action'],
      type: 'object',
    },
  },
  {
    name: 'open_url',
    description: 'Открыть сайт в браузере по умолчанию',
    risk: 'safe',
    group: 'Браузер и поиск',
    schema: {
      additionalProperties: false,
      properties: {
        url: {
          description: 'Адрес, например https://youtube.com или vk.com',
          type: 'string',
        },
      },
      required: ['url'],
      type: 'object',
    },
  },
  {
    name: 'web_search',
    description: 'Найти в интернете (открывает страницу поиска в браузере)',
    risk: 'safe',
    group: 'Браузер и поиск',
    schema: {
      additionalProperties: false,
      properties: {
        query: {
          description: 'Поисковый запрос',
          minLength: 1,
          type: 'string',
        },
      },
      required: ['query'],
      type: 'object',
    },
  },
  {
    name: 'get_time',
    description: 'Узнать текущее время и дату',
    risk: 'safe',
    group: 'Разное',
    schema: {
      additionalProperties: false,
      properties: {},
      type: 'object',
    },
  },
  {
    name: 'wait',
    description: 'Подождать указанное число секунд (например, пока откроется программа)',
    risk: 'safe',
    group: 'Разное',
    schema: {
      additionalProperties: false,
      properties: {
        seconds: {
          description: 'Сколько секунд ждать, до 60',
          exclusiveMinimum: 0,
          maximum: 60,
          type: 'number',
        },
      },
      required: ['seconds'],
      type: 'object',
    },
  },
  {
    name: 'say',
    description: 'Произнести фразу вслух',
    risk: 'safe',
    group: 'Разное',
    schema: {
      additionalProperties: false,
      properties: {
        text: {
          description: 'Что сказать',
          minLength: 1,
          type: 'string',
        },
      },
      required: ['text'],
      type: 'object',
    },
  },
  {
    name: 'shutdown',
    description: 'Выключить компьютер',
    risk: 'confirm',
    group: 'Питание',
    schema: {
      additionalProperties: false,
      properties: {},
      type: 'object',
    },
  },
]
