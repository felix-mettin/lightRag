import { useState, useCallback, useEffect, useMemo } from 'react'
import Button from '@/components/ui/Button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter
} from '@/components/ui/Dialog'
import Input from '@/components/ui/Input'
import { toast } from 'sonner'
import { errorMessage } from '@/lib/utils'
import { deleteDocuments } from '@/api/lightrag'
import type { DocStatusResponse } from '@/api/lightrag'

import { TrashIcon, AlertTriangleIcon } from 'lucide-react'
import { useTranslation } from 'react-i18next'

// Simple Label component
const Label = ({
  htmlFor,
  className,
  children,
  ...props
}: React.LabelHTMLAttributes<HTMLLabelElement>) => (
  <label
    htmlFor={htmlFor}
    className={className}
    {...props}
  >
    {children}
  </label>
)

interface DeleteDocumentsDialogProps {
  selectedDocIds: string[]
  selectedDocs?: DocStatusResponse[]
  onDocumentsDeleted?: () => Promise<void>
}

export default function DeleteDocumentsDialog({ selectedDocIds, selectedDocs, onDocumentsDeleted }: DeleteDocumentsDialogProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [confirmText, setConfirmText] = useState('')
  const [deleteFile, setDeleteFile] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [deleteLLMCache, setDeleteLLMCache] = useState(false)
  const isConfirmEnabled = confirmText.toLowerCase() === 'yes' && !isDeleting

  const parseHttpErrorDetail = (err: any): string | null => {
    const msg = errorMessage(err)
    if (!msg) return null

    // Try to extract JSON body like {"detail":"..."}
    const jsonMatch = msg.match(/\{[\s\S]*\}/)
    if (jsonMatch) {
      try {
        const obj = JSON.parse(jsonMatch[0])
        if (obj && obj.detail) return String(obj.detail)
      } catch (e) {
        // ignore
      }
    }

    // Fallback: return whole message if it mentions 'not found' or 'invalid'
    if (/not found/i.test(msg) || /invalid/i.test(msg) || /cannot delete/i.test(msg)) {
      return msg
    }
    return null
  }

  // Reset state when dialog closes
  useEffect(() => {
    if (!open) {
      setConfirmText('')
      setDeleteFile(false)
      setDeleteLLMCache(false)
      setIsDeleting(false)
    }
  }, [open])

  const handleDelete = useCallback(async () => {
    if (!isConfirmEnabled || selectedDocIds.length === 0) return

    setIsDeleting(true)
    try {
      if (selectedDocs && selectedDocs.length > 0) {
        const groups: Record<string, string[]> = {}
        for (const doc of selectedDocs) {
          const ws = (doc as any).workspace || 'others'
          groups[ws] = groups[ws] || []
          groups[ws].push(doc.id)
        }

        let overallSuccess = true
        let anyBusy = false
        let anyNotAllowed = false
        const failedMsgs: string[] = []
        const warningMsgs: string[] = []

        for (const [ws, ids] of Object.entries(groups)) {
          try {
            const result = await deleteDocuments(ids, deleteFile, deleteLLMCache, ws)
            if (result.status === 'deletion_started') {
              // If backend mentions some docs not found, record warning
              if (result.message && /not found/i.test(result.message)) {
                warningMsgs.push(`工作区 ${ws}: ${result.message}`)
                overallSuccess = false
              }
            } else if (result.status === 'busy') {
              anyBusy = true
              overallSuccess = false
            } else if (result.status === 'not_allowed') {
              anyNotAllowed = true
              overallSuccess = false
            } else {
              overallSuccess = false
              failedMsgs.push(`${ws}: ${result.message || '未知错误'}`)
            }
          } catch (err) {
            overallSuccess = false
            const detail = parseHttpErrorDetail(err)
            if (detail) {
              if (/not found/i.test(detail)) {
                failedMsgs.push(`工作区 ${ws}: 部分文档不存在或已被删除 (${detail})`)
              } else if (/Invalid standard_type/i.test(detail) || /invalid standard_type/i.test(detail)) {
                failedMsgs.push(`工作区 ${ws}: 无效的工作区 (${detail})`)
              } else {
                failedMsgs.push(`工作区 ${ws}: ${detail}`)
              }
            } else {
              failedMsgs.push(`工作区 ${ws}: ${errorMessage(err)}`)
            }
          }
        }

        if (overallSuccess && failedMsgs.length === 0 && warningMsgs.length === 0) {
          toast.success(t('documentPanel.deleteDocuments.success', { count: selectedDocIds.length }))
        } else if (anyBusy) {
          toast.error('删除请求未能发送：目标工作区正在处理其他任务，请稍后重试。')
          setConfirmText('')
          setIsDeleting(false)
          return
        } else if (anyNotAllowed) {
          toast.error('删除请求被拒绝：当前操作不允许在该状态下执行。')
          setConfirmText('')
          setIsDeleting(false)
          return
        } else {
          // Show combined messages: failures and warnings
          if (failedMsgs.length) {
            toast.error('删除失败：' + failedMsgs.join('; '))
          }
          if (warningMsgs.length) {
            toast.error('部分文档未找到或已被删除：' + warningMsgs.join('; '))
          }
          setConfirmText('')
          setIsDeleting(false)
          return
        }

      } else {
        // Fallback: legacy single call with ids only
        try {
          const result = await deleteDocuments(selectedDocIds, deleteFile, deleteLLMCache)
          if (result.status === 'deletion_started') {
            if (result.message && /not found/i.test(result.message)) {
              toast.error('删除部分发起，但有文档未找到：' + result.message)
            } else {
              toast.success(t('documentPanel.deleteDocuments.success', { count: selectedDocIds.length }))
            }
          } else if (result.status === 'busy') {
            toast.error('删除请求未能发送：目标工作区正在处理其他任务，请稍后重试。')
            setConfirmText('')
            setIsDeleting(false)
            return
          } else if (result.status === 'not_allowed') {
            toast.error('删除请求被拒绝：当前操作不允许在该状态下执行。')
            setConfirmText('')
            setIsDeleting(false)
            return
          } else {
            toast.error('删除失败：' + (result.message || '未知错误'))
            setConfirmText('')
            setIsDeleting(false)
            return
          }
        } catch (err) {
          const detail = parseHttpErrorDetail(err)
          if (detail && /not found/i.test(detail)) {
            toast.error('删除失败：文档不存在或已被删除。请刷新列表后重试。')
          } else if (detail && /Invalid standard_type/i.test(detail)) {
            toast.error('删除失败：无效的工作区，请确认选择。')
          } else {
            toast.error(t('documentPanel.deleteDocuments.error', { error: errorMessage(err) }))
          }
          setConfirmText('')
          setIsDeleting(false)
          return
        }
      }

      // Refresh document list if provided
      if (onDocumentsDeleted) {
        onDocumentsDeleted().catch(console.error)
      }

      // Close dialog after successful operation
      setOpen(false)
    } catch (err) {
      toast.error(t('documentPanel.deleteDocuments.error', { error: errorMessage(err) }))
      setConfirmText('')
    } finally {
      setIsDeleting(false)
    }
  }, [isConfirmEnabled, selectedDocIds, selectedDocs, deleteFile, deleteLLMCache, setOpen, t, onDocumentsDeleted])

  const workspaceGroups = useMemo(() => {
    const groups: Record<string, string[]> = {}
    if (!selectedDocs) return groups
    for (const doc of selectedDocs) {
      const ws = (doc as any).workspace || 'others'
      groups[ws] = groups[ws] || []
      groups[ws].push(doc.id)
    }
    return groups
  }, [selectedDocs])

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button
          variant="destructive"
          side="bottom"
          tooltip={t('documentPanel.deleteDocuments.tooltip', { count: selectedDocIds.length })}
          size="sm"
        >
          <TrashIcon/> {t('documentPanel.deleteDocuments.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-red-500 dark:text-red-400 font-bold">
            <AlertTriangleIcon className="h-5 w-5" />
            {t('documentPanel.deleteDocuments.title')}
          </DialogTitle>
          <DialogDescription className="pt-2">
            {t('documentPanel.deleteDocuments.description', { count: selectedDocIds.length })}
          </DialogDescription>
        </DialogHeader>

        <div className="text-red-500 dark:text-red-400 font-semibold mb-4">
          {t('documentPanel.deleteDocuments.warning')}
        </div>

        <div className="mb-4">
          {t('documentPanel.deleteDocuments.confirm', { count: selectedDocIds.length })}
        </div>

        {selectedDocs && Object.keys(workspaceGroups).length > 0 && (
          <div className="mb-4">
            <div className="text-sm text-gray-700 dark:text-gray-300 font-medium mb-1">{t('documentPanel.deleteDocuments.groupSummary')}</div>
            <ul className="list-disc ml-5 text-sm text-gray-600 dark:text-gray-400">
              {Object.entries(workspaceGroups).map(([ws, ids]) => (
                <li key={ws}>{ws}: {ids.length} {t('documentPanel.deleteDocuments.documentsLabel')}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="confirm-text" className="text-sm font-medium">
              {t('documentPanel.deleteDocuments.confirmPrompt')}
            </Label>
            <Input
              id="confirm-text"
              value={confirmText}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setConfirmText(e.target.value)}
              placeholder={t('documentPanel.deleteDocuments.confirmPlaceholder')}
              className="w-full"
              disabled={isDeleting}
            />
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="delete-file"
              checked={deleteFile}
              onChange={(e) => setDeleteFile(e.target.checked)}
              disabled={isDeleting}
              className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
            />
            <Label htmlFor="delete-file" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.deleteDocuments.deleteFileOption')}
            </Label>
          </div>

          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              id="delete-llm-cache"
              checked={deleteLLMCache}
              onChange={(e) => setDeleteLLMCache(e.target.checked)}
              disabled={isDeleting}
              className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 rounded"
            />
            <Label htmlFor="delete-llm-cache" className="text-sm font-medium cursor-pointer">
              {t('documentPanel.deleteDocuments.deleteLLMCacheOption')}
            </Label>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isDeleting}>
            {t('common.cancel')}
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={!isConfirmEnabled}
          >
            {isDeleting ? t('documentPanel.deleteDocuments.deleting') : t('documentPanel.deleteDocuments.confirmButton')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
