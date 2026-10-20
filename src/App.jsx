import React, { useState } from 'react';
import Sidebar, { NAV_ITEMS } from './components/common/Sidebar';
import TopHeader from './components/common/TopHeader';
import CommandApprovalModal from './components/common/CommandApprovalModal';
import { MODEL_CATALOG } from './data/mockModels';
import { studioApi } from './lib/studioApi';

// 16 Complete Screen Views
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

export default function App() {
  const [activeScreen, setActiveScreen] = useState('dashboard');
  const [activeModel, setActiveModel] = useState(MODEL_CATALOG[0]);
  const [selectedLayer, setSelectedLayer] = useState(12);
  const [selectedTensorKey, setSelectedTensorKey] = useState('model.layers.12.self_attn.q_proj.weight');

  // Command Approval Modal (Human-in-the-Loop) State
  const [approvalModalOpen, setApprovalModalOpen] = useState(false);
  const [pendingCommand, setPendingCommand] = useState('');
  const [pendingExplanation, setPendingExplanation] = useState('');

  // Backend job whose live output is shown in the Fine-Tuning and Terminal screens
  const [activeJobId, setActiveJobId] = useState(null);

  // Technical Docs / Specification Modal State
  const [docsModalOpen, setDocsModalOpen] = useState(false);

  // Training state
  const [trainingState, setTrainingState] = useState({
    isRunning: true,
    step: 520,
    maxSteps: 6000,
    epoch: 1,
    maxEpochs: 3,
    loss: 0.853,
    evalLoss: 0.912,
    vramUsed: 5.12,
    vramTotal: 6.00,
    gpuUtil: 88,
    elapsed: '1h 23m 12s',
    eta: '2h 10m'
  });

  const currentNav = NAV_ITEMS.find(n => n.id === activeScreen) || NAV_ITEMS[0];

  const handleRequestLaunchTraining = ({ command, config }) => {
    setPendingCommand(command);
    setPendingExplanation(`Fine-tuning ${config.hfRepo} with ${config.method} (LoRA r=${config.loraRank}, ${config.use4bit ? '4-bit' : '16-bit'}). This runs the Unsloth trainer from ~/.ai_studio/venv in its own process group and streams its output live to the terminal below. The first run downloads the model weights from Hugging Face.`);
    setApprovalModalOpen(true);
  };

  const handleApproveCommand = (executedCmd) => {
    studioApi.startJob({ command: executedCmd, label: 'Unsloth fine-tuning run', kind: 'train' })
      .then((job) => setActiveJobId(job.id))
      .catch((e) => window.alert(`Could not start training: ${e.message}`));
  };

  return (
    <div className="app-container">
      {/* 1. Cyber Sidebar Navigation (16 items) */}
      <Sidebar 
        activeScreen={activeScreen} 
        onSelectScreen={(screenId) => setActiveScreen(screenId)} 
      />

      {/* 2. Main Workstation Area */}
      <div className="main-wrapper">
        <TopHeader 
          currentScreenTitle={currentNav.label}
          currentScreenSubtitle={`Active: ${activeModel.name} (${activeModel.format})`}
          activeModel={activeModel}
          models={MODEL_CATALOG}
          onSelectModel={setActiveModel}
          onOpenTerminal={() => setActiveScreen('terminal')}
          onOpenDocs={() => setDocsModalOpen(true)}
        />

        {/* 3. Screen Viewport */}
        <main className="page-viewport">
          {activeScreen === 'dashboard' && (
            <DashboardView 
              onNavigate={(s) => setActiveScreen(s)} 
              activeModel={activeModel} 
            />
          )}

          {activeScreen === 'model_import' && (
            <ModelImportView 
              models={MODEL_CATALOG}
              activeModel={activeModel}
              onSelectModel={setActiveModel}
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'model_overview' && (
            <ModelOverviewView 
              activeModel={activeModel}
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'arch_overall' && (
            <ArchitectureOverallView 
              activeModel={activeModel}
              onSelectLayer={(l) => {
                setSelectedLayer(l);
                setActiveScreen('layer_detail');
              }}
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'layer_detail' && (
            <LayerDetailedView 
              activeLayer={selectedLayer}
              onSelectLayer={setSelectedLayer}
              onNavigate={(s) => setActiveScreen(s)}
              onSelectTensor={(t) => {
                setSelectedTensorKey(t);
                setActiveScreen('tensor_inspect');
              }}
            />
          )}

          {activeScreen === 'tensor_inspect' && (
            <TensorInspectorView 
              selectedTensorKey={selectedTensorKey}
            />
          )}

          {activeScreen === 'tokenizer' && (
            <TokenizerView 
              activeModel={activeModel}
            />
          )}

          {activeScreen === 'datasets' && (
            <DatasetManagerView 
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'fine_tuning' && (
            <FineTuningConfigView 
              activeModel={activeModel}
              models={MODEL_CATALOG}
              onRequestLaunchTraining={handleRequestLaunchTraining}
              activeJobId={activeJobId}
              onJobStarted={setActiveJobId}
            />
          )}

          {activeScreen === 'training_live' && (
            <TrainingProgressView 
              trainingState={trainingState}
              onStopTraining={() => setActiveScreen('fine_tuning')}
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'comparison' && (
            <BeforeAfterComparisonView 
              activeModel={activeModel}
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'terminal' && (
            <TerminalView activeJobId={activeJobId} onSelectJob={setActiveJobId} />
          )}

          {activeScreen === 'monitoring' && (
            <SystemMonitorView />
          )}

          {activeScreen === 'experiments' && (
            <ExperimentManagerView 
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}

          {activeScreen === 'inference_chat' && (
            <InferenceChatView 
              activeModel={activeModel}
            />
          )}

          {activeScreen === 'export_deploy' && (
            <ModelExportView 
              activeModel={activeModel}
              onNavigate={(s) => setActiveScreen(s)}
            />
          )}
        </main>
      </div>

      {/* Human-in-the-Loop Command Approval Security Modal */}
      <CommandApprovalModal 
        isOpen={approvalModalOpen}
        onClose={() => setApprovalModalOpen(false)}
        onApprove={handleApproveCommand}
        command={pendingCommand}
        explanation={pendingExplanation}
        riskLevel="Resource Intensive"
      />

      {/* Master Technical Architecture Modal */}
      {docsModalOpen && (
        <div className="modal-overlay" onClick={() => setDocsModalOpen(false)}>
          <div className="modal-dialog" style={{ maxWidth: '800px', maxHeight: '85vh', display: 'flex', flexDirection: 'column' }} onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <h3 style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc' }}>
                Master Technical Architecture & Specification (20 Deliverables)
              </h3>
              <button className="btn-icon" onClick={() => setDocsModalOpen(false)}>✕</button>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '14px', background: '#090e1a', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)', fontSize: '12.5px', color: '#cbd5e1', lineHeight: '1.6' }}>
              <p style={{ color: '#38bdf8', marginBottom: '8px' }}>
                The complete Master Specification document has been generated in your workspace artifacts:
              </p>
              <div style={{ fontFamily: 'JetBrains Mono', background: '#040711', padding: '8px 12px', borderRadius: '6px', color: '#10b981', marginBottom: '14px' }}>
                master_technical_specification.md
              </div>
              <h4 style={{ color: '#fff', fontSize: '13px', marginTop: '10px' }}>Key Architecture Highlights:</h4>
              <ul style={{ paddingLeft: '18px', marginTop: '6px' }}>
                <li><strong>Zero-Copy Header Introspection:</strong> Uses mmap to read Safetensors JSON descriptors without loading weights into RAM.</li>
                <li><strong>WebGL/Three.js 3D Acceleration:</strong> Dynamic parametric surface elevation meshes with Jet/Spectral colormaps, 3D loss terrain, and token trajectories.</li>
                <li><strong>Unsloth & QLoRA Pipeline:</strong> Triton/CUDA kernels reducing VRAM by up to 80% with 4-bit NormalFloat base quantization.</li>
                <li><strong>PTY Process Isolation:</strong> Linux process groups with os.setsid() and Human-in-the-Loop execution gates.</li>
                <li><strong>Differential Parameter Analysis:</strong> Delta W = B * A matrix elevation comparisons between frozen base and LoRA checkpoints.</li>
              </ul>
            </div>
            <div style={{ marginTop: '14px', textAlign: 'right' }}>
              <button className="btn btn-primary" onClick={() => setDocsModalOpen(false)}>
                Close Viewer
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
