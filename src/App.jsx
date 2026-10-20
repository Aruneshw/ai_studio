import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Sidebar, { NAV_ITEMS } from './components/common/Sidebar';
import TopHeader from './components/common/TopHeader';
import CommandApprovalModal from './components/common/CommandApprovalModal';
import { api, useApi } from './api';

import DashboardView from './views/DashboardView';
import ModelImportView from './views/ModelImportView';
import ModelOverviewView from './views/ModelOverviewView';
import ArchitectureOverallView from './views/ArchitectureOverallView';
import LayerDetailedView from './views/LayerDetailedView';
import TensorInspectorView from './views/TensorInspectorView';
import TokenizerView from './views/TokenizerView';
import DatasetManagerView from './views/DatasetManagerView';
import FineTuningConfigView from './views/FineTuningConfigView';
import TrainingProgressView from './views/TrainingProgressView';
import BeforeAfterComparisonView from './views/BeforeAfterComparisonView';
import TerminalView from './views/TerminalView';
import SystemMonitorView from './views/SystemMonitorView';
import ExperimentManagerView from './views/ExperimentManagerView';
import InferenceChatView from './views/InferenceChatView';
import ModelExportView from './views/ModelExportView';

function stored(key, fallback) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}

export default function App() {
  const [activeScreen, setActiveScreen] = useState(() => stored('studio.screen', 'dashboard'));
  const [activeModelId, setActiveModelId] = useState(() => stored('studio.model', null));
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedTensor, setSelectedTensor] = useState(null);
  const [activeRunId, setActiveRunId] = useState(null);
  const [terminalTab, setTerminalTab] = useState('terminal');
  const [pending, setPending] = useState(null); // {command, params, explanation}
  const [launchError, setLaunchError] = useState(null);

  const modelsRes = useApi('/api/models');
  const system = useApi('/api/system', 3000);
  const env = useApi('/api/env', 5000);
  const runsRes = useApi('/api/runs', 5000);

  const models = useMemo(() => modelsRes.data?.models || [], [modelsRes.data]);
  const activeModel = models.find((m) => m.id === activeModelId) || models[0] || null;
  const runs = useMemo(() => runsRes.data?.runs || [], [runsRes.data]);

  useEffect(() => { try { localStorage.setItem('studio.screen', activeScreen); } catch { /* ignore */ } }, [activeScreen]);
  useEffect(() => {
    if (activeModel) { try { localStorage.setItem('studio.model', activeModel.id); } catch { /* ignore */ } }
  }, [activeModel]);
  useEffect(() => {
    if (!activeRunId && runs.length) {
      const live = runs.find((r) => r.state === 'running');
      setActiveRunId((live || runs[0]).id);
    }
  }, [runs, activeRunId]);

  const navigate = useCallback((s, opts = {}) => {
    if (opts.terminalTab) setTerminalTab(opts.terminalTab);
    setActiveScreen(s);
  }, []);
  const selectModel = useCallback((m) => { setActiveModelId(m.id); setSelectedTensor(null); setSelectedLayer(0); }, []);

  const handleRequestLaunch = ({ command, params, config }) => {
    setLaunchError(null);
    setPending({
      command, params,
      explanation: `Fine-tune ${config.model_name} with ${config.method.toUpperCase()} on ${config.dataset} ` +
        `(${config.dataset_records} records, ${config.dataset_format} format). Runs Unsloth in a separate ` +
        `process group; logs and metrics stream to data/runs/${config.run_id}/.`,
    });
  };

  const handleApprove = async (command) => {
    try {
      const r = await api('/api/train/launch', { method: 'POST', body: { params: pending.params, command } });
      setActiveRunId(r.runId);
      setPending(null);
      runsRes.reload();
      setActiveScreen('training_live');
    } catch (e) {
      setLaunchError(e.message);
    }
  };

  const currentNav = NAV_ITEMS.find((n) => n.id === activeScreen) || NAV_ITEMS[0];
  const common = { activeModel, models, onNavigate: navigate, onSelectModel: selectModel };

  return (
    <div className="app-container">
      <Sidebar activeScreen={activeScreen} onSelectScreen={(s) => setActiveScreen(s)}
        modelCount={models.length} runCount={runs.length} system={system.data}
        trainingLive={runs.some((r) => r.state === 'running')} />

      <div className="main-wrapper">
        <TopHeader
          currentScreenTitle={currentNav.label}
          currentScreenSubtitle={activeModel ? `Active: ${activeModel.name} (${activeModel.format})` : 'No model selected'}
          activeModel={activeModel} models={models} onSelectModel={selectModel}
          system={system.data} env={env.data} backendError={modelsRes.error || system.error}
          onOpenTerminal={() => navigate('terminal', { terminalTab: 'terminal' })}
          onOpenEnv={() => navigate('terminal', { terminalTab: 'doctor' })}
        />

        <main className="page-viewport">
          {activeScreen === 'dashboard' && <DashboardView {...common} system={system.data} env={env.data} runs={runs} />}
          {activeScreen === 'model_import' && <ModelImportView {...common} library={modelsRes.data?.library} loading={modelsRes.loading}
            error={modelsRes.error} onModelsChanged={modelsRes.reload} env={env.data} />}
          {activeScreen === 'model_overview' && <ModelOverviewView {...common}
            onSelectTensor={(t) => { setSelectedTensor(t); setActiveScreen('tensor_inspect'); }} />}
          {activeScreen === 'arch_overall' && <ArchitectureOverallView {...common}
            onSelectLayer={(l) => { setSelectedLayer(l); setActiveScreen('layer_detail'); }} />}
          {activeScreen === 'layer_detail' && <LayerDetailedView {...common} activeLayer={selectedLayer} onSelectLayer={setSelectedLayer}
            onSelectTensor={(t) => { setSelectedTensor(t); setActiveScreen('tensor_inspect'); }} />}
          {activeScreen === 'tensor_inspect' && <TensorInspectorView {...common} selectedTensor={selectedTensor} onSelectTensor={setSelectedTensor} />}
          {activeScreen === 'tokenizer' && <TokenizerView {...common} />}
          {activeScreen === 'datasets' && <DatasetManagerView {...common} />}
          {activeScreen === 'fine_tuning' && <FineTuningConfigView {...common} env={env.data} system={system.data}
            onRequestLaunchTraining={handleRequestLaunch} />}
          {activeScreen === 'training_live' && <TrainingProgressView {...common} runs={runs} runId={activeRunId}
            onSelectRun={setActiveRunId} onRunsChanged={runsRes.reload} system={system.data} />}
          {activeScreen === 'comparison' && <BeforeAfterComparisonView {...common} />}
          {activeScreen === 'monitoring' && <SystemMonitorView system={system.data} env={env.data} />}
          {activeScreen === 'experiments' && <ExperimentManagerView {...common} runs={runs} onRunsChanged={runsRes.reload}
            onOpenRun={(id) => { setActiveRunId(id); setActiveScreen('training_live'); }} onModelsChanged={modelsRes.reload} />}
          {activeScreen === 'inference_chat' && <InferenceChatView {...common} />}
          {activeScreen === 'export_deploy' && <ModelExportView {...common} env={env.data} onModelsChanged={modelsRes.reload} />}
          {/* Terminal stays mounted so the shell session survives navigation */}
          <div style={{ display: activeScreen === 'terminal' ? 'block' : 'none' }}>
            <TerminalView visible={activeScreen === 'terminal'} tab={terminalTab} onTabChange={setTerminalTab}
              env={env.data} onEnvChanged={env.reload} />
          </div>
        </main>
      </div>

      <CommandApprovalModal
        key={pending?.command || 'none'}
        isOpen={!!pending}
        onClose={() => setPending(null)}
        onApprove={handleApprove}
        command={pending?.command || ''}
        explanation={pending?.explanation}
        error={launchError}
        riskLevel="Resource Intensive"
      />
    </div>
  );
}
