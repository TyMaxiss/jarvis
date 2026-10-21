// Вместо pywebview в браузере: window.pywebview.api отправляет вызовы по HTTP настоящему
// Python (tests/test_ui_e2e.py), а события Python приходят опросом /events.
// Подключается до скриптов страницы (Playwright addInitScript).
;(() => {
  const view = new URLSearchParams(location.search).get('view') || 'main'
  const api = new Proxy(
    {},
    {
      get(_, method) {
        // Promise проверяет then у результата — у моста его нет.
        if (method === 'then') return undefined
        return async (...args) => {
          const response = await fetch(`/api/${view}/${String(method)}`, { method: 'POST', body: JSON.stringify(args) })
          const data = await response.json()
          if (!response.ok) throw new Error(data.error)
          return data.result
        }
      },
    },
  )
  window.pywebview = { api }
  async function poll() {
    for (;;) {
      try {
        const response = await fetch(`/events/${view}`)
        for (const script of await response.json()) (0, eval)(script)
      } catch {
        await new Promise((resolve) => setTimeout(resolve, 200))
      }
    }
  }
  void poll()
})()
