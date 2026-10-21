import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { currentWindow, isMock } from './lib/bridge'
import { applyTheme, useAppState } from './lib/store'
import { MainWindow } from './views/MainWindow'
import { Overlay } from './views/Overlay'
import { QuickChat } from './views/QuickChat'

// Одна сборка на три окна: какое показать, решает адрес (?view=overlay, ?view=chat).
const view = currentWindow()
const mock = isMock()

// Тема из адреса — до первой отрисовки, чтобы светлое окно не мигало тёмным.
const params = new URLSearchParams(window.location.search)
const theme = params.get('theme')
if (theme === 'light' || theme === 'dark') applyTheme({ theme, accent: params.get('accent') ?? '' })

function Root() {
  useAppState() // загрузка состояния и темы для любого окна
  if (view === 'overlay') return <Overlay framed={mock} />
  if (view === 'chat') return <QuickChat framed={mock} />
  return <MainWindow />
}

if (mock && view !== 'main') {
  // В браузере окна оверлея и чата рисуются по центру страницы.
  document.body.classList.add('mock-stage')
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Root />
  </StrictMode>,
)
