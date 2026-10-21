import { History, House, ListChecks, Settings2 } from 'lucide-react'
import type { ReactNode } from 'react'
import type { AppState } from '../lib/types'
import { voiceLabel } from '../lib/format'
import { cx } from './ui'

export type Page = 'home' | 'commands' | 'history' | 'settings'

const items: Array<{ page: Page; title: string; icon: ReactNode }> = [
  { page: 'home', title: 'Главная', icon: <House size={17} /> },
  { page: 'commands', title: 'Команды', icon: <ListChecks size={17} /> },
  { page: 'history', title: 'История', icon: <History size={17} /> },
  { page: 'settings', title: 'Настройки', icon: <Settings2 size={17} /> },
]

export function Sidebar({ page, onPage, state }: { page: Page; onPage: (page: Page) => void; state: AppState | null }) {
  const voice = state?.voice
  const live = voice && !voice.paused && voice.state !== 'off'
  return (
    <nav className="flex w-52 shrink-0 flex-col border-r border-line px-3 py-4" aria-label="Разделы">
      <ul className="flex flex-col gap-0.5">
        {items.map((item) => (
          <li key={item.page}>
            <button
              type="button"
              onClick={() => onPage(item.page)}
              aria-current={page === item.page ? 'page' : undefined}
              className={cx(
                'relative flex h-9 w-full items-center gap-3 rounded-xl px-3 text-[13.5px] transition-colors duration-150',
                page === item.page ? 'bg-surface-2 text-text' : 'text-muted hover:bg-surface hover:text-text',
              )}
            >
              {page === item.page && <span className="absolute top-2 bottom-2 left-0 w-[3px] rounded-full bg-accent" />}
              {item.icon}
              <span className="flex-1 text-left">{item.title}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className="flex-1" />
      {voice && (
        <div className="flex items-center gap-2.5 rounded-xl px-3 py-2 text-[12.5px] text-muted">
          <span className={cx('h-2 w-2 rounded-full', live ? 'bg-ok' : 'bg-faint')} />
          <span className="truncate">{voiceLabel(state)}</span>
        </div>
      )}
    </nav>
  )
}
