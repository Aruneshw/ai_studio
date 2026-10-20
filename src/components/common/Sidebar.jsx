import React from 'react';
import { 
  LayoutDashboard, 
  Library, 
  Layers, 
  Network, 
  Box, 
  Sliders, 
  FileCode, 
  Database, 
  Play, 
  Activity, 
  GitCompare, 
  Terminal, 
  Cpu, 
  History, 
  MessageSquare, 
  Share2,
  Sparkles
} from 'lucide-react';

export const NAV_ITEMS = [
  { id: 'dashboard', label: '1. Home / Dashboard', icon: LayoutDashboard, category: 'EXPLORE' },
  { id: 'model_import', label: '2. Model Import / Library', icon: Library, category: 'EXPLORE', badge: '5' },
  { id: 'model_overview', label: '3. Model Overview', icon: Layers, category: 'EXPLORE' },
  { id: 'arch_overall', label: '4. Neural Network (3D)', icon: Network, category: 'EXPLORE', badge: '3D' },
  { id: 'layer_detail', label: '5. Layer-level View', icon: Box, category: 'EXPLORE' },
  { id: 'tensor_inspect', label: '6. Tensor / 3D Weights', icon: Sliders, category: 'EXPLORE', badge: '3D' },
  { id: 'tokenizer', label: '7. Tokenizer Explorer', icon: FileCode, category: 'EXPLORE', badge: '3D' },
  { id: 'datasets', label: '8. Dataset Management', icon: Database, category: 'TRAINING' },
  { id: 'fine_tuning', label: '9. Fine-Tuning Setup', icon: Play, category: 'TRAINING' },
  { id: 'training_live', label: '10. Training Live (3D)', icon: Activity, category: 'TRAINING', badge: 'Live', hot: true },
  { id: 'comparison', label: '11. Before vs After (3D)', icon: GitCompare, category: 'TRAINING', badge: '3D' },
  { id: 'terminal', label: '12. Linux Terminal', icon: Terminal, category: 'SYSTEM', badge: 'CLI' },
  { id: 'monitoring', label: '13. GPU & System (3D)', icon: Cpu, category: 'SYSTEM', badge: '3D' },
  { id: 'experiments', label: '14. Experiments', icon: History, category: 'SYSTEM', badge: '4' },
  { id: 'inference_chat', label: '15. Chat / Testing', icon: MessageSquare, category: 'DEPLOY' },
  { id: 'export_deploy', label: '16. Export / Deploy', icon: Share2, category: 'DEPLOY' }
];

export default function Sidebar({ activeScreen, onSelectScreen }) {
  const categories = ['EXPLORE', 'TRAINING', 'SYSTEM', 'DEPLOY'];

  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-header">
        <div className="brand-icon">
          <Sparkles size={18} color="#fff" />
        </div>
        <div>
          <div className="brand-text">AI Model Studio</div>
          <div style={{ fontSize: '10px', color: '#64748b' }}>v2.4 &bull; 3D Workstation</div>
        </div>
        <span className="brand-badge">Local</span>
      </div>

      {/* Nav List */}
      <div className="sidebar-nav">
        {categories.map((cat) => {
          const items = NAV_ITEMS.filter((item) => item.category === cat);
          return (
            <React.Fragment key={cat}>
              <div className="nav-section-title">{cat}</div>
              {items.map((item) => {
                const Icon = item.icon;
                const isActive = activeScreen === item.id;
                return (
                  <div
                    key={item.id}
                    className={`nav-item ${isActive ? 'active' : ''}`}
                    onClick={() => onSelectScreen(item.id)}
                  >
                    <Icon className="nav-icon" />
                    <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {item.label}
                    </span>
                    {item.badge && (
                      <span className={`nav-badge-pill ${item.hot ? 'hot' : item.badge === '3D' ? 'accent' : ''}`}>
                        {item.badge}
                      </span>
                    )}
                  </div>
                );
              })}
            </React.Fragment>
          );
        })}
      </div>

      {/* Hardware Status Footer */}
      <div className="sidebar-footer">
        <div className="hw-pill-compact">
          <span style={{ color: '#06b6d4' }}>● RTX 3050</span>
          <span style={{ color: '#10b981' }}>5.1 / 6.0 GB</span>
        </div>
      </div>
    </aside>
  );
}
