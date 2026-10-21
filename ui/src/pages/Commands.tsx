// Раздел «Команды»: слева паки и их команды, справа редактор выбранной команды.

import { CircleAlert, FileCode2, FolderOpen, ListChecks, Plus, Search } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { SlotText } from '../components/Chips'
import { Button, IconButton, Keys, TextInput, cx } from '../components/ui'
import { call } from '../lib/bridge'
import { emptyForm } from '../lib/commands'
import { useAssistantEvent } from '../lib/store'
import type { ActionInfo, CommandForm, CommandPack, CommandSaveResult, CommandsTree } from '../lib/types'
import { CommandEditor, type EditorTarget } from './CommandEditor'

export function CommandsPage({ draft, onDraftUsed }: { draft?: CommandForm | null; onDraftUsed?: () => void }) {
  const [tree, setTree] = useState<CommandsTree | null>(null)
  const [actions, setActions] = useState<ActionInfo[]>([])
  const [target, setTarget] = useState<EditorTarget | null>(null)
  const [version, setVersion] = useState(0) // новая открытая команда — новый редактор
  const [search, setSearch] = useState('')
  const [error, setError] = useState('')
  const dirty = useRef(false)
  const onDirty = useCallback((value: boolean) => {
    dirty.current = value
  }, [])

  const loadTree = useCallback(async () => {
    const next = await call('commands_tree')
    if (next) setTree(next)
  }, [])

  useEffect(() => {
    void loadTree()
    void call('actions').then((list) => list && setActions(list))
  }, [loadTree])
  // Файлы команд поменялись снаружи (редактор текста, git pull) — перечитать список.
  useAssistantEvent((event) => {
    if (event.type === 'commands') void loadTree()
  })

  const leave = () => !dirty.current || window.confirm('Изменения не сохранены. Уйти от команды?')

  const open = async (file: string, id: string) => {
    if (!leave()) return
    const details = await call('command_get', file, id)
    if (!details) return
    if (!details.ok || !details.command) {
      setError(details.errors.join('; '))
      return
    }
    setError('')
    dirty.current = false
    setVersion((current) => current + 1)
    setTarget({
      file,
      originalId: id,
      user: Boolean(details.user),
      form: details.command,
      problems: details.problems ?? [],
      alsoIn: details.also_in ?? [],
    })
  }

  const create = useCallback((form: CommandForm = emptyForm()) => {
    dirty.current = false
    setError('')
    setVersion((current) => current + 1)
    setTarget({ file: '', originalId: '', user: true, form, problems: [], alsoIn: [] })
  }, [])

  // Черновик из Истории («Сохранить как команду») открывается новой командой.
  useEffect(() => {
    if (draft) {
      create(draft)
      onDraftUsed?.()
    }
  }, [draft, create, onDraftUsed])

  const saved = (result: CommandSaveResult) => {
    setTree(result)
    dirty.current = false
    if (result.id) void open(result.user_file, result.id)
  }

  const deleted = (result: CommandSaveResult) => {
    setTree(result)
    dirty.current = false
    setTarget(null)
  }

  const takenIds = new Set(tree?.packs.find((pack) => pack.user)?.commands.map((c) => c.id) ?? [])
  const selected = target ? `${target.file}/${target.originalId}` : ''
  return (
    <div className="flex h-full min-h-0">
      <aside className="flex w-72 shrink-0 flex-col border-r border-line">
        <div className="flex flex-col gap-3 p-4 pb-3">
          <div className="flex items-center gap-2">
            <h2 className="flex-1 font-display text-[17px] font-semibold">Команды</h2>
            <IconButton label="Открыть папку команд" onClick={() => void call('open_path', 'commands')}>
              <FolderOpen size={16} />
            </IconButton>
          </div>
          <div className="relative">
            <Search size={14} className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-faint" />
            <TextInput
              aria-label="Поиск команды"
              placeholder="Поиск по фразе или id"
              className="w-full pl-8"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <Button variant="primary" icon={<Plus size={14} />} onClick={() => leave() && create()}>
            Новая команда
          </Button>
        </div>
        <nav aria-label="Паки команд" className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
          {tree?.packs.map((pack) => (
            <PackSection key={pack.file} pack={pack} search={search} selected={selected} onOpen={(id) => void open(pack.file, id)} />
          ))}
        </nav>
      </aside>
      <section className="min-w-0 flex-1 overflow-y-auto" aria-label="Редактор команды">
        {error && <p className="m-7 text-[13px] text-danger">{error}</p>}
        {target && tree ? (
          <CommandEditor
            key={version}
            target={target}
            actions={actions}
            userFile={tree.user_file}
            takenIds={takenIds}
            onSaved={saved}
            onDeleted={deleted}
            onDirty={onDirty}
          />
        ) : (
          <Empty onCreate={() => create()} />
        )}
      </section>
    </div>
  )
}

function PackSection({
  pack,
  search,
  selected,
  onOpen,
}: {
  pack: CommandPack
  search: string
  selected: string
  onOpen: (id: string) => void
}) {
  const needle = search.trim().toLowerCase()
  const commands = pack.commands.filter(
    (c) => !needle || c.title.toLowerCase().includes(needle) || c.id.toLowerCase().includes(needle),
  )
  if (needle && commands.length === 0) return null
  return (
    <div className="mt-2">
      <div className="flex items-center gap-2 px-2 py-1.5 text-[12px] font-medium text-muted">
        <FileCode2 size={13} className="shrink-0" />
        <span className="flex-1 truncate">{pack.user ? `Мои команды · ${pack.file}` : pack.file}</span>
        <span className="text-faint tabular-nums">{pack.commands.length}</span>
      </div>
      {pack.error && <p className="px-2 py-1 text-[12px] leading-snug text-danger">{pack.error}</p>}
      {pack.user && pack.commands.length === 0 && !pack.error && (
        <p className="px-2 py-1 text-[12px] leading-snug text-faint">Пока пусто — создайте свою команду или измените встроенную.</p>
      )}
      <ul>
        {commands.map((command) => {
          const key = `${pack.file}/${command.id}`
          return (
            <li key={key}>
              <button
                type="button"
                onClick={() => onOpen(command.id)}
                aria-current={selected === key ? 'true' : undefined}
                className={cx(
                  'flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-[13px] transition-colors',
                  selected === key ? 'bg-surface-2 text-text' : 'text-text/90 hover:bg-surface',
                  !command.active && 'opacity-60',
                )}
              >
                <span className="min-w-0 flex-1 truncate">
                  <SlotText text={command.title} />
                </span>
                {command.problem && (
                  <span title={command.problem} aria-label="Не загружается" className="shrink-0 text-danger">
                    <CircleAlert size={13} />
                  </span>
                )}
                {command.replaced_by && !command.problem && (
                  <span title={`Работает версия из ${command.replaced_by}`} className="shrink-0 text-[11px] text-faint">
                    заменена
                  </span>
                )}
                {command.hotkey && (
                  <span className="shrink-0 scale-90">
                    <Keys value={command.hotkey} />
                  </span>
                )}
              </button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function Empty({ onCreate }: { onCreate: () => void }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 p-7 text-center">
      <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-surface-2 text-muted">
        <ListChecks size={26} />
      </span>
      <div>
        <h1 className="font-display text-[19px] font-semibold">Команды без нейросети</h1>
        <p className="mt-1.5 max-w-md text-[13px] leading-relaxed text-muted">
          Выберите команду слева, чтобы изменить фразы и действия, или создайте свою. Ассистент выполняет их сразу,
          без запроса к нейросети.
        </p>
      </div>
      <Button icon={<Plus size={14} />} onClick={onCreate}>
        Новая команда
      </Button>
    </div>
  )
}
