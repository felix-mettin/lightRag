import { create } from 'zustand'
import { createSelectors } from '@/lib/utils'

interface Workspace {
  name: string
  has_data: boolean
  label_count: number
}

interface WorkspaceState {
  documentWorkspace: string
  setDocumentWorkspace: (workspace: string) => void
  availableWorkspaces: Workspace[]
  setAvailableWorkspaces: (workspaces: Workspace[]) => void
}

const useWorkspaceStoreBase = create<WorkspaceState>((set) => ({
  documentWorkspace: 'others',
  setDocumentWorkspace: (workspace: string) => set({ documentWorkspace: workspace }),
  availableWorkspaces: [],
  setAvailableWorkspaces: (workspaces: Workspace[]) => set({ availableWorkspaces: workspaces })
}))

export const useWorkspaceStore = createSelectors(useWorkspaceStoreBase)
