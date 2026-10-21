import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { currentWindow, isMock } from './lib/bridge'
import { useAppState } from './lib/store'
import { MainWindow } from './views/MainWindow'
import { Overlay } from './views/Overlay'
import { QuickChat } from './views/QuickChat'

// Одна сборка на три окна: какое показать, решает адрес (?view=overlay, ?view=chat).
const view = currentWindow()
const mock = isMock()

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
