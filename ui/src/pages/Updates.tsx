// Обновления: баннер «Доступна версия» на главной и раздел в настройках (assistant/updates.py).

import { Download, ExternalLink, RefreshCw } from 'lucide-react'
import { Button, Card, CardTitle, Field, ProgressBar, Toggle, cx } from '../components/ui'
import { call } from '../lib/bridge'
import type { UpdateState } from '../lib/types'

const SOURCE_HINT = 'Джарвис запущен из исходников — обновите их: git pull и uv sync.'

/** Что сейчас с обновлениями — одной строкой. */
export function updateStatus(update: UpdateState): string {
  if (update.installing) return update.step || 'Готовлю обновление'
  if (update.checking) return 'Проверяю…'
  if (update.error) return update.error
  if (update.release) return `Доступна версия ${update.release.version}`
  if (update.checked) return 'У вас последняя версия'
  return ''
}

/** Ход скачивания и установки. */
function InstallProgress({ update }: { update: UpdateState }) {
  const percent = update.progress === null ? null : Math.round(update.progress * 100)
  return (
    <div className="mt-2 w-full">
      {percent !== null && <ProgressBar label="Ход обновления" value={percent} />}
      <p className="mt-1.5 text-[12px] text-faint">Настройки, ключи, свои команды и модели останутся. Джарвис перезапустится сам.</p>
    </div>
  )
}

/** Кнопки «Обновить» и «Что нового». */
function UpdateActions({ update }: { update: UpdateState }) {
  return (
    <div className="flex shrink-0 items-center gap-2">
      <Button variant="ghost" icon={<ExternalLink size={14} />} onClick={() => void call('update_open_page')}>
        Что нового
      </Button>
      {update.supported && (
        <Button
          variant="primary"
          icon={<Download size={14} />}
          busy={update.installing}
          onClick={() => void call('update_install')}
        >
          Обновить
        </Button>
      )}
    </div>
  )
}

/** Главная: вышла новая версия. */
export function UpdateBanner({ update }: { update: UpdateState }) {
  const release = update.release
  if (!release) return null
  return (
    <div
      role="status"
      aria-label="Обновление"
      className="animate-fade-up flex items-center gap-4 rounded-card border border-accent/30 bg-accent/10 px-5 py-4"
    >
      <Download size={18} className="shrink-0 text-accent" />
      <div className="min-w-0 flex-1">
        <div className="text-[14px] font-medium">Доступна новая версия {release.version}</div>
        <p className={cx('mt-0.5 truncate text-[12.5px]', update.error && !update.installing ? 'text-danger' : 'text-muted')}>
          {update.installing
            ? updateStatus(update)
            : update.error || (update.supported ? `У вас ${update.version}. Обновление займёт пару минут.` : SOURCE_HINT)}
        </p>
        {update.installing && <InstallProgress update={update} />}
      </div>
      <UpdateActions update={update} />
    </div>
  )
}

/** Настройки: версия, автоматическая проверка и «Проверить сейчас». */
export function UpdatesCard({
  update,
  checkUpdates,
  onCheckUpdates,
}: {
  update: UpdateState
  checkUpdates: boolean
  onCheckUpdates: (value: boolean) => void
}) {
  const status = updateStatus(update)
  return (
    <Card>
      <CardTitle icon={<RefreshCw size={16} />} title="Обновления" />
      <Field label="Версия" hint={update.supported ? 'Новые версии — на странице релизов на GitHub' : SOURCE_HINT}>
        <span className="font-display text-[15px] font-semibold tabular-nums">{update.version}</span>
      </Field>
      <Field label="Проверять обновления" hint="Раз в сутки; о новой версии скажет уведомление">
        <Toggle label="Проверять обновления" checked={checkUpdates} onChange={onCheckUpdates} />
      </Field>
      <Field label="Новая версия" hint={status || 'Ещё не проверяли'} stacked={update.installing}>
        {update.release && !update.installing ? (
          <UpdateActions update={update} />
        ) : update.installing ? (
          <InstallProgress update={update} />
        ) : (
          <Button
            icon={<RefreshCw size={14} />}
            busy={update.checking}
            onClick={() => void call('update_check')}
          >
            Проверить сейчас
          </Button>
        )}
      </Field>
    </Card>
  )
}
