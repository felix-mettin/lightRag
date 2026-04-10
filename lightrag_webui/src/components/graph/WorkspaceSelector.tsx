import { useEffect, useState } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/Select'
import { useTranslation } from 'react-i18next'
import { DatabaseIcon } from 'lucide-react'
import { getWorkspaces } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'
import { toast } from 'sonner'

/**
 * Component that provides workspace selection controls for graph data isolation.
 */
const WorkspaceSelector = () => {
  const { t } = useTranslation()
  const currentWorkspace = useGraphStore.use.currentWorkspace()
  const availableWorkspaces = useGraphStore.use.availableWorkspaces()
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const loadWorkspaces = async () => {
      try {
        setIsLoading(true)
        const response = await getWorkspaces()
        const workspaces = Object.values(response.workspaces)
        const state = useGraphStore.getState()
        state.setAvailableWorkspaces(workspaces)

        const workspaceNames = new Set(workspaces.map((workspace) => workspace.name))
        const shouldDefaultToGb =
          workspaceNames.has('GB') &&
          (state.currentWorkspace === 'others' || !workspaceNames.has(state.currentWorkspace))

        if (shouldDefaultToGb) {
          state.setCurrentWorkspace('GB')
          state.incrementGraphDataVersion()
        }
      } catch (error) {
        console.error('Error loading workspaces:', error)
        toast.error(t('graphPanel.workspaceSelector.loadError'))
      } finally {
        setIsLoading(false)
      }
    }

    if (availableWorkspaces.length === 0) {
      loadWorkspaces()
    }
  }, [t, availableWorkspaces.length])

  const handleWorkspaceChange = (workspace: string) => {
    useGraphStore.getState().setCurrentWorkspace(workspace)
    // Trigger graph data refresh when workspace changes
    useGraphStore.getState().incrementGraphDataVersion()
    toast.success(t('graphPanel.workspaceSelector.switched', { workspace }))
  }

  return (
    <div className="flex items-center gap-2">
      <DatabaseIcon className="h-4 w-4 text-muted-foreground" />
      <Select
        value={currentWorkspace}
        onValueChange={handleWorkspaceChange}
        disabled={isLoading}
      >
        <SelectTrigger className="w-32 h-8 text-xs">
          <SelectValue placeholder={t('graphPanel.workspaceSelector.selectWorkspace')} />
        </SelectTrigger>
        <SelectContent>
          {availableWorkspaces.map((workspace) => (
            <SelectItem key={workspace.name} value={workspace.name}>
              <div className="flex items-center justify-between w-full">
                <span>{workspace.name}</span>
                <span className="text-xs text-muted-foreground ml-2">
                  ({workspace.label_count})
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export default WorkspaceSelector