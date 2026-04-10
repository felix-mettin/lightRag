import { useEffect, useState } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useTranslation } from 'react-i18next'
import { DatabaseIcon } from 'lucide-react'
import { getWorkspaces } from '@/api/lightrag'
import { toast } from 'sonner'

interface WorkspaceSelectorProps {
  currentWorkspace: string
  onWorkspaceChange: (workspace: string) => void
  onWorkspacesLoaded?: (workspaces: Array<{
    name: string
    has_data: boolean
    label_count: number
  }>) => void
  className?: string
}

/**
 * Generic workspace selector component that can be used across different pages.
 */
const WorkspaceSelector = ({
  currentWorkspace,
  onWorkspaceChange,
  onWorkspacesLoaded,
  className = ''
}: WorkspaceSelectorProps) => {
  const { t } = useTranslation()
  const [availableWorkspaces, setAvailableWorkspaces] = useState<Array<{
    name: string
    has_data: boolean
    label_count: number
  }>>([])
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const loadWorkspaces = async () => {
      try {
        setIsLoading(true)
        const response = await getWorkspaces()
        const workspaces = Object.values(response.workspaces)
        setAvailableWorkspaces(workspaces)
        onWorkspacesLoaded?.(workspaces)
      } catch (error) {
        console.error('Error loading workspaces:', error)
        toast.error(t('graphPanel.workspaceSelector.loadError'))
      } finally {
        setIsLoading(false)
      }
    }

    loadWorkspaces()
  }, [onWorkspacesLoaded, t])

  const handleWorkspaceChange = (workspace: string) => {
    onWorkspaceChange(workspace)
    toast.success(t('graphPanel.workspaceSelector.switched', { workspace }))
  }

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <DatabaseIcon className="h-4 w-4 text-muted-foreground" />
      <Select
        value={currentWorkspace}
        onValueChange={handleWorkspaceChange}
        disabled={isLoading}
      >
        <SelectTrigger className="w-40">
          <SelectValue placeholder={t('graphPanel.workspaceSelector.selectWorkspace')} />
        </SelectTrigger>
        <SelectContent>
          {availableWorkspaces.map((workspace) => (
            <SelectItem key={workspace.name} value={workspace.name}>
              <div className="flex items-center justify-between w-full">
                <span>{workspace.name}</span>
                {workspace.has_data && (
                  <span className="text-xs text-muted-foreground ml-2">
                    ({workspace.label_count})
                  </span>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export default WorkspaceSelector