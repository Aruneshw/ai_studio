import React, { useState } from 'react';
import { Terminal, X, Play, Copy, Check } from 'lucide-react';

export default function CommandApprovalModal({ 
  isOpen, 
  onClose, 
  onApprove, 
  command = '', 
  title = 'Command Execution Approval Required',
  explanation = 'The studio has generated the following CLI command. Please inspect and approve before execution.',
  riskLevel = 'Safe', // 'Safe', 'Resource Intensive', 'Destructive'
  error = null
}) {
  const [editedCommand, setEditedCommand] = useState(command);
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(editedCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getRiskBadge = () => {
    switch (riskLevel) {
      case 'Resource Intensive':
        return <span className="badge badge-amber">⚠️ High GPU / VRAM Allocation</span>;
      case 'Destructive':
        return <span className="badge badge-rose">🛑 File System Modification</span>;
      default:
        return <span className="badge badge-emerald">🛡️ Isolated Execution</span>;
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '6px', background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Terminal size={18} color="#06b6d4" />
            </div>
            <div>
              <h3 style={{ fontSize: '15px', fontWeight: 600, color: '#f1f5f9' }}>{title}</h3>
              <p style={{ fontSize: '12px', color: '#94a3b8' }}>Human-in-the-Loop Security Gate</p>
            </div>
          </div>
          <button className="btn-icon" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        {/* Risk Assessment */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
          <span style={{ fontSize: '12px', color: '#94a3b8' }}>Safety Classification:</span>
          {getRiskBadge()}
        </div>

        {/* Explanation */}
        <div style={{ background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255, 255, 255, 0.07)', borderRadius: '8px', padding: '12px', marginBottom: '14px', fontSize: '12.5px', color: '#cbd5e1', lineHeight: '1.5' }}>
          {explanation}
        </div>

        {/* Command Editor / Viewer */}
        <div style={{ marginBottom: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
            <span style={{ fontSize: '12px', fontWeight: 500, color: '#94a3b8' }}>Generated Command Preview:</span>
            <button 
              className="btn btn-secondary" 
              style={{ padding: '2px 8px', fontSize: '11px' }} 
              onClick={handleCopy}
            >
              {copied ? <Check size={12} color="#10b981" /> : <Copy size={12} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
          </div>
          <textarea
            className="textarea"
            style={{ 
              fontFamily: 'JetBrains Mono, monospace', 
              fontSize: '12px', 
              color: '#38bdf8', 
              background: '#040711',
              border: '1px solid rgba(6, 182, 212, 0.3)',
              borderRadius: '8px',
              minHeight: '90px',
              padding: '10px'
            }}
            value={editedCommand}
            onChange={(e) => setEditedCommand(e.target.value)}
          />
        </div>

        {error && (
          <div style={{ marginBottom: '12px', padding: '8px 10px', borderRadius: '6px', background: 'rgba(244,63,94,0.1)', border: '1px solid rgba(244,63,94,0.35)', color: '#fda4af', fontSize: '12px' }}>
            {error}
          </div>
        )}

        {/* Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
          <button className="btn btn-secondary" onClick={onClose}>
            Cancel Execution
          </button>
          <button 
            className="btn btn-primary" 
            onClick={() => onApprove(editedCommand)}
          >
            <Play size={14} />
            Approve & Execute Command
          </button>
        </div>
      </div>
    </div>
  );
}
