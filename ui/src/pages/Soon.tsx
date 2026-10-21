import { FileText, FolderOpen, History, ListChecks } from 'lucide-react'
import { Button } from '../components/ui'
import { call } from '../lib/bridge'

/** Заглушка разделов следующего этапа: что уже можно сделать руками. */
export function Soon({ page }: { page: 'commands' | 'history' }) {
  const commands = page === 'commands'
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-7 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface-2 text-muted">
        {commands ? <ListChecks size={26} /> : <History size={26} />}
      </span>
      <div>
        <h1 className="font-display text-[19px] font-semibold">{commands ? 'Редактор команд' : 'История'}</h1>
        <p className="mt-1.5 max-w-md text-[13px] leading-relaxed text-muted">
          {commands
            ? 'Появится на следующем этапе. Пока свои команды можно добавить в commands/my.yaml — ассистент подхватит их сразу после сохранения файла.'
            : 'Появится на следующем этапе. Последние команды видны на главной, а полный журнал действий — в logs/actions.jsonl.'}
        </p>
      </div>
      {commands ? (
        <Button icon={<FolderOpen size={14} />} onClick={() => void call('open_path', 'commands')}>
          Открыть папку команд
        </Button>
      ) : (
        <Button icon={<FileText size={14} />} onClick={() => void call('open_path', 'journal')}>
          Открыть журнал
        </Button>
      )}
    </div>
  )
}
