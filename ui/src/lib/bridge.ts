// Связь с Python: вызовы — через window.pywebview.api (js_api в pywebview),
// события — через window.__assistantEvent({type, payload}), который вызывает Python.

import type { AssistantApi, AssistantEvent, WindowName } from './types'

declare global {
  interface Window {
    pywebview?: { api: AssistantApi }
    __assistantEvent?: (event: AssistantEvent) => void
  }
}

type Listener = (event: AssistantEvent) => void

const listeners = new Set<Listener>()

export function emit(event: AssistantEvent): void {
  for (const listener of [...listeners]) listener(event)
}

window.__assistantEvent = emit

export function onEvent(listener: Listener): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

/** Подделка API для разработки в браузере: `npm run dev` и адрес с ?mock. */
export function isMock(): boolean {
  return new URLSearchParams(window.location.search).has('mock')
}

let apiPromise: Promise<AssistantApi> | null = null

export function getApi(): Promise<AssistantApi> {
  if (apiPromise) return apiPromise
  apiPromise = isMock()
    ? import('./mock').then((mock) => mock.createMockApi(emit))
    : new Promise<AssistantApi>((resolve) => {
        if (window.pywebview?.api) {
          resolve(window.pywebview.api)
          return
        }
        window.addEventListener('pywebviewready', () => resolve(window.pywebview!.api), { once: true })
      })
  return apiPromise
}

/** Вызов метода Python; ошибки не роняют интерфейс, а пишутся в консоль разработчика. */
export async function call<K extends keyof AssistantApi>(
  method: K,
  ...args: Parameters<AssistantApi[K]>
): Promise<Awaited<ReturnType<AssistantApi[K]>> | undefined> {
  const api = await getApi()
  try {
    const fn = api[method] as (...a: unknown[]) => Promise<unknown>
    return (await fn(...args)) as Awaited<ReturnType<AssistantApi[K]>>
  } catch (error) {
    console.error(`Ошибка вызова ${String(method)}`, error)
    return undefined
  }
}

/** Какое окно открыто: главное, оверлей или быстрый чат (index.html?view=overlay).
 *
 * Не через #: pywebview в Windows превращает «#/overlay» в «#\overlay» при разборе пути. */
export function currentWindow(): WindowName {
  const view = new URLSearchParams(window.location.search).get('view')
  return view === 'overlay' || view === 'chat' ? view : 'main'
}
