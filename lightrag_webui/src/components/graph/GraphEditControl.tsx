import { useState } from 'react'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/Popover'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import { controlButtonVariant } from '@/lib/constants'
import { EditIcon, PlusCircle, Link2, Edit3, Trash2, Users, ArrowRight } from 'lucide-react'
import { toast } from 'sonner'
import { createEntity, createRelation, updateEntity, updateRelation, deleteEntity, deleteRelation, checkEntityNameExists } from '@/api/lightrag'
import { useGraphStore } from '@/stores/graph'

type TabType = 'create-entity' | 'create-relation' | 'update-entity' | 'update-relation' | 'delete-entity' | 'delete-relation'

const GraphEditControl = () => {
  const currentWorkspace = useGraphStore.use.currentWorkspace()
  const createEdgeMode = useGraphStore.use.createEdgeMode()
  const [opened, setOpened] = useState<boolean>(false)
  const [activeTab, setActiveTab] = useState<TabType>('create-entity')

  const [entityName, setEntityName] = useState('')
  const [entityDescription, setEntityDescription] = useState('')
  const [entityType, setEntityType] = useState('')

  const [updateEntityName, setUpdateEntityName] = useState('')
  const [updateEntityDescription, setUpdateEntityDescription] = useState('')
  const [updateEntityType, setUpdateEntityType] = useState('')

  const [deleteEntityName, setDeleteEntityName] = useState('')

  const [sourceEntity, setSourceEntity] = useState('')
  const [targetEntity, setTargetEntity] = useState('')
  const [relationDescription, setRelationDescription] = useState('')
  const [relationKeywords, setRelationKeywords] = useState('')

  const [updateSourceEntity, setUpdateSourceEntity] = useState('')
  const [updateTargetEntity, setUpdateTargetEntity] = useState('')
  const [updateRelationDescription, setUpdateRelationDescription] = useState('')
  const [updateRelationKeywords, setUpdateRelationKeywords] = useState('')

  const [deleteSourceEntity, setDeleteSourceEntity] = useState('')
  const [deleteTargetEntity, setDeleteTargetEntity] = useState('')

  const [isSubmitting, setIsSubmitting] = useState(false)

  const resetForm = () => {
    setEntityName('')
    setEntityDescription('')
    setEntityType('')
    setUpdateEntityName('')
    setUpdateEntityDescription('')
    setUpdateEntityType('')
    setDeleteEntityName('')
    setSourceEntity('')
    setTargetEntity('')
    setRelationDescription('')
    setRelationKeywords('')
    setUpdateSourceEntity('')
    setUpdateTargetEntity('')
    setUpdateRelationDescription('')
    setUpdateRelationKeywords('')
    setDeleteSourceEntity('')
    setDeleteTargetEntity('')
  }

  const handleCreateEntity = async () => {
    if (!entityName.trim()) {
      toast.error('请输入实体名称')
      return
    }

    setIsSubmitting(true)
    try {
      const exists = await checkEntityNameExists(entityName.trim(), currentWorkspace)
      if (exists) {
        toast.error('实体已存在')
        return
      }

      const entityData: Record<string, any> = {}
      if (entityDescription.trim()) entityData.description = entityDescription.trim()
      if (entityType.trim()) entityData.entity_type = entityType.trim()

      await createEntity(entityName.trim(), entityData, currentWorkspace)

      toast.success('实体创建成功')
      resetForm()
      setOpened(false)
      useGraphStore.getState().incrementGraphDataVersion()

    } catch (error) {
      console.error('Error creating entity:', error)
      toast.error('创建实体失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleCreateRelation = async () => {
    if (!sourceEntity.trim() || !targetEntity.trim()) {
      toast.error('请输入起始实体和目标实体')
      return
    }

    if (sourceEntity.trim() === targetEntity.trim()) {
      toast.error('起始实体和目标实体不能相同')
      return
    }

    setIsSubmitting(true)
    try {
      const sourceExists = await checkEntityNameExists(sourceEntity.trim(), currentWorkspace)
      const targetExists = await checkEntityNameExists(targetEntity.trim(), currentWorkspace)

      if (!sourceExists) {
        toast.error('起始实体不存在')
        return
      }
      if (!targetExists) {
        toast.error('目标实体不存在')
        return
      }

      const relationData: Record<string, any> = {}
      if (relationDescription.trim()) relationData.description = relationDescription.trim()
      if (relationKeywords.trim()) relationData.keywords = relationKeywords.trim()

      await createRelation(sourceEntity.trim(), targetEntity.trim(), relationData, currentWorkspace)

      toast.success('关系创建成功')
      resetForm()
      setOpened(false)
      useGraphStore.getState().incrementGraphDataVersion()

    } catch (error) {
      console.error('Error creating relation:', error)
      toast.error('创建关系失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleUpdateEntity = async () => {
    if (!updateEntityName.trim()) {
      toast.error('请输入实体名称')
      return
    }

    setIsSubmitting(true)
    try {
      const entityData: Record<string, any> = {}
      if (updateEntityDescription.trim()) entityData.description = updateEntityDescription.trim()
      if (updateEntityType.trim()) entityData.entity_type = updateEntityType.trim()

      await updateEntity(updateEntityName.trim(), entityData, currentWorkspace)

      toast.success('实体更新成功')
      resetForm()
      setOpened(false)
      useGraphStore.getState().incrementGraphDataVersion()

    } catch (error) {
      console.error('Error updating entity:', error)
      toast.error('更新实体失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDeleteEntity = async () => {
    if (!deleteEntityName.trim()) {
      toast.error('请输入实体名称')
      return
    }

    setIsSubmitting(true)
    try {
      await deleteEntity(deleteEntityName.trim(), currentWorkspace)

      toast.success('实体删除成功')
      resetForm()
      setOpened(false)
      useGraphStore.getState().incrementGraphDataVersion()

    } catch (error) {
      console.error('Error deleting entity:', error)
      toast.error('删除实体失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleUpdateRelation = async () => {
    if (!updateSourceEntity.trim() || !updateTargetEntity.trim()) {
      toast.error('请输入起始实体和目标实体')
      return
    }

    setIsSubmitting(true)
    try {
      const relationData: Record<string, any> = {}
      if (updateRelationDescription.trim()) relationData.description = updateRelationDescription.trim()
      if (updateRelationKeywords.trim()) relationData.keywords = updateRelationKeywords.trim()

      await updateRelation(updateSourceEntity.trim(), updateTargetEntity.trim(), relationData, currentWorkspace)

      toast.success('关系更新成功')
      resetForm()
      setOpened(false)
      useGraphStore.getState().incrementGraphDataVersion()

    } catch (error) {
      console.error('Error updating relation:', error)
      toast.error('更新关系失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDeleteRelation = async () => {
    if (!deleteSourceEntity.trim() || !deleteTargetEntity.trim()) {
      toast.error('请输入起始实体和目标实体')
      return
    }

    setIsSubmitting(true)
    try {
      await deleteRelation(deleteSourceEntity.trim(), deleteTargetEntity.trim(), currentWorkspace)

      toast.success('关系删除成功')
      resetForm()
      setOpened(false)
      useGraphStore.getState().incrementGraphDataVersion()

    } catch (error) {
      console.error('Error deleting relation:', error)
      toast.error('删除关系失败')
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleSubmit = () => {
    switch (activeTab) {
      case 'create-entity':
        handleCreateEntity()
        break
      case 'create-relation':
        handleCreateRelation()
        break
      case 'update-entity':
        handleUpdateEntity()
        break
      case 'update-relation':
        handleUpdateRelation()
        break
      case 'delete-entity':
        handleDeleteEntity()
        break
      case 'delete-relation':
        handleDeleteRelation()
        break
    }
  }

  const tabs = [
    { id: 'create-entity' as TabType, label: '新建实体', icon: PlusCircle, group: 'entity' },
    { id: 'update-entity' as TabType, label: '编辑实体', icon: Edit3, group: 'entity' },
    { id: 'delete-entity' as TabType, label: '删除实体', icon: Trash2, group: 'entity' },
    { id: 'create-relation' as TabType, label: '新建关系', icon: Link2, icon2: PlusCircle, group: 'relation' },
    { id: 'update-relation' as TabType, label: '编辑关系', icon: Link2, icon2: Edit3, group: 'relation' },
    { id: 'delete-relation' as TabType, label: '删除关系', icon: Link2, icon2: Trash2, group: 'relation' },
  ]

  const entityTabs = tabs.filter(t => t.group === 'entity')
  const relationTabs = tabs.filter(t => t.group === 'relation')

  return (
    <Popover open={opened} onOpenChange={setOpened}>
      <PopoverTrigger asChild>
        <Button
          variant={controlButtonVariant}
          tooltip="图数据编辑"
          size="icon"
          disabled={isSubmitting}
        >
          <EditIcon />
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="right"
        align="end"
        sideOffset={8}
        collisionPadding={5}
        className="p-0 w-96"
        onCloseAutoFocus={(e) => e.preventDefault()}
      >
        <div className="flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b bg-muted/30">
            <h3 className="text-base font-semibold">图数据编辑</h3>
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">拖拽连线</span>
              <Button
                size="sm"
                variant={createEdgeMode ? 'default' : 'outline'}
                onClick={() => useGraphStore.getState().setCreateEdgeMode(!createEdgeMode)}
                className="h-7 px-3 text-xs"
              >
                {createEdgeMode ? '开启' : '关闭'}
              </Button>
            </div>
          </div>

          <div className="p-4 space-y-4">
            <div className="space-y-2">
              <div className="flex items-center gap-1 text-xs text-muted-foreground mb-2">
                <Users className="h-3.5 w-3.5" />
                <span>实体操作</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                {entityTabs.map((tab) => (
                  <button
                    key={tab.id}
                    className={`flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded-md transition-colors ${
                      activeTab === tab.id
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    <tab.icon className="h-3.5 w-3.5" />
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1 text-xs text-muted-foreground mb-2">
                <ArrowRight className="h-3.5 w-3.5" />
                <span>关系操作</span>
              </div>
              <div className="grid grid-cols-3 gap-1">
                {relationTabs.map((tab) => (
                  <button
                    key={tab.id}
                    className={`flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded-md transition-colors ${
                      activeTab === tab.id
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80'
                    }`}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    <tab.icon className="h-3 w-3.5" />
                    <span>{tab.label}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="border-t pt-4">
              {activeTab === 'create-entity' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-primary">
                    <PlusCircle className="h-4 w-4" />
                    <span>新建实体</span>
                  </div>
                  <div className="space-y-2">
                    <div>
                      <label className="text-xs text-muted-foreground">实体名称 *</label>
                      <Input
                        value={entityName}
                        onChange={(e) => setEntityName(e.target.value)}
                        placeholder="输入实体名称"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">实体类型</label>
                      <Input
                        value={entityType}
                        onChange={(e) => setEntityType(e.target.value)}
                        placeholder="输入实体类型（可选）"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">实体描述</label>
                      <Textarea
                        value={entityDescription}
                        onChange={(e) => setEntityDescription(e.target.value)}
                        placeholder="输入实体描述（可选）"
                        className="mt-1 min-h-[60px] text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'update-entity' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-primary">
                    <Edit3 className="h-4 w-4" />
                    <span>编辑实体</span>
                  </div>
                  <div className="space-y-2">
                    <div>
                      <label className="text-xs text-muted-foreground">实体名称 *</label>
                      <Input
                        value={updateEntityName}
                        onChange={(e) => setUpdateEntityName(e.target.value)}
                        placeholder="输入要编辑的实体名称"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">实体类型</label>
                      <Input
                        value={updateEntityType}
                        onChange={(e) => setUpdateEntityType(e.target.value)}
                        placeholder="输入新的实体类型"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">实体描述</label>
                      <Textarea
                        value={updateEntityDescription}
                        onChange={(e) => setUpdateEntityDescription(e.target.value)}
                        placeholder="输入新的实体描述"
                        className="mt-1 min-h-[60px] text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'delete-entity' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-destructive">
                    <Trash2 className="h-4 w-4" />
                    <span>删除实体</span>
                  </div>
                  <div className="space-y-2">
                    <div>
                      <label className="text-xs text-muted-foreground">实体名称 *</label>
                      <Input
                        value={deleteEntityName}
                        onChange={(e) => setDeleteEntityName(e.target.value)}
                        placeholder="输入要删除的实体名称"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                    <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                      ⚠️ 删除实体会同时删除与该实体相关的所有关系
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'create-relation' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-primary">
                    <Link2 className="h-4 w-4" />
                    <PlusCircle className="h-3 w-3" />
                    <span>新建关系</span>
                  </div>
                  <div className="space-y-2">
                    <div className="grid grid-cols-5 gap-2 items-center">
                      <div className="col-span-2">
                        <label className="text-xs text-muted-foreground">起始实体 *</label>
                        <Input
                          value={sourceEntity}
                          onChange={(e) => setSourceEntity(e.target.value)}
                          placeholder="起始实体"
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                      <div className="flex items-center justify-center pt-4">
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                      <div className="col-span-2">
                        <label className="text-xs text-muted-foreground">目标实体 *</label>
                        <Input
                          value={targetEntity}
                          onChange={(e) => setTargetEntity(e.target.value)}
                          placeholder="目标实体"
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">关系描述</label>
                      <Textarea
                        value={relationDescription}
                        onChange={(e) => setRelationDescription(e.target.value)}
                        placeholder="输入关系描述（可选）"
                        className="mt-1 min-h-[50px] text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">关键词</label>
                      <Input
                        value={relationKeywords}
                        onChange={(e) => setRelationKeywords(e.target.value)}
                        placeholder="输入关键词，用逗号分隔"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                    <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                      💡 提示：开启"拖拽连线"后，可在图中直接拖拽节点创建关系
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'update-relation' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-primary">
                    <Link2 className="h-4 w-4" />
                    <Edit3 className="h-3 w-3" />
                    <span>编辑关系</span>
                  </div>
                  <div className="space-y-2">
                    <div className="grid grid-cols-5 gap-2 items-center">
                      <div className="col-span-2">
                        <label className="text-xs text-muted-foreground">起始实体 *</label>
                        <Input
                          value={updateSourceEntity}
                          onChange={(e) => setUpdateSourceEntity(e.target.value)}
                          placeholder="起始实体"
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                      <div className="flex items-center justify-center pt-4">
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                      <div className="col-span-2">
                        <label className="text-xs text-muted-foreground">目标实体 *</label>
                        <Input
                          value={updateTargetEntity}
                          onChange={(e) => setUpdateTargetEntity(e.target.value)}
                          placeholder="目标实体"
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">关系描述</label>
                      <Textarea
                        value={updateRelationDescription}
                        onChange={(e) => setUpdateRelationDescription(e.target.value)}
                        placeholder="输入新的关系描述"
                        className="mt-1 min-h-[50px] text-sm"
                      />
                    </div>
                    <div>
                      <label className="text-xs text-muted-foreground">关键词</label>
                      <Input
                        value={updateRelationKeywords}
                        onChange={(e) => setUpdateRelationKeywords(e.target.value)}
                        placeholder="输入新的关键词"
                        className="mt-1 h-8 text-sm"
                      />
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'delete-relation' && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-medium text-destructive">
                    <Link2 className="h-4 w-4" />
                    <Trash2 className="h-3 w-3" />
                    <span>删除关系</span>
                  </div>
                  <div className="space-y-2">
                    <div className="grid grid-cols-5 gap-2 items-center">
                      <div className="col-span-2">
                        <label className="text-xs text-muted-foreground">起始实体 *</label>
                        <Input
                          value={deleteSourceEntity}
                          onChange={(e) => setDeleteSourceEntity(e.target.value)}
                          placeholder="起始实体"
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                      <div className="flex items-center justify-center pt-4">
                        <ArrowRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                      <div className="col-span-2">
                        <label className="text-xs text-muted-foreground">目标实体 *</label>
                        <Input
                          value={deleteTargetEntity}
                          onChange={(e) => setDeleteTargetEntity(e.target.value)}
                          placeholder="目标实体"
                          className="mt-1 h-8 text-sm"
                        />
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                      ⚠️ 删除关系后无法恢复
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="flex gap-2 pt-2 border-t">
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() => {
                  resetForm()
                  setOpened(false)
                }}
              >
                取消
              </Button>
              <Button
                size="sm"
                className="flex-1"
                onClick={handleSubmit}
                disabled={isSubmitting}
              >
                {isSubmitting ? '处理中...' : '确认'}
              </Button>
            </div>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  )
}

export default GraphEditControl
