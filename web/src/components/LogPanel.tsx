/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState } from 'react';
import { Terminal, Trash2, Copy, ShieldAlert, Check } from 'lucide-react';

interface LogPanelProps {
  logs: string[];
  onClearLogs: () => void;
}

export default function LogPanel({ logs, onClearLogs }: LogPanelProps) {
  const terminalEndRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterText, setFilterText] = useState('');
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (autoScroll) {
      terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, autoScroll]);

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(logs.join('\n'));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy logs', err);
    }
  };

  const getLineColorClass = (line: string) => {
    const text = line.toLowerCase();
    if (text.includes('✅') || text.includes('success') || text.includes('完成') || text.includes('completed')) {
      return 'text-green-400';
    }
    if (text.includes('❌') || text.includes('error') || text.includes('失败') || text.includes('failed') || text.includes('exception')) {
      return 'text-red-400 font-medium';
    }
    if (text.includes('⚠️') || text.includes('warning') || text.includes('警告') || text.includes('warn')) {
      return 'text-yellow-400';
    }
    if (text.includes('📋') || text.includes('info') || text.includes('处理') || text.includes('processing') || text.includes('start')) {
      return 'text-sky-400';
    }
    return 'text-gray-400';
  };

  const filteredLogs = logs.filter(log =>
    log.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="bg-panel/40 border border-border-subtle rounded-xl p-5 flex flex-col h-[350px]" id="log-panel-container">
      {/* Panel Header */}
      <div className="flex items-center justify-between border-b border-border-subtle pb-3 mb-3">
        <div className="flex items-center space-x-2">
          <Terminal className="h-5 w-5 text-brand" />
          <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-widest">实时控制台日志 (Console Logs)</h2>
        </div>
        <div className="flex items-center space-x-2">
          {/* Copy logs */}
          <button
            type="button"
            id="btn-copy-logs"
            onClick={copyToClipboard}
            disabled={logs.length === 0}
            className="p-1.5 rounded bg-panel border border-border-subtle hover:border-gray-700 text-gray-400 hover:text-gray-200 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            title="复制所有日志"
          >
            {copied ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
          </button>
          {/* Clear logs */}
          <button
            type="button"
            id="btn-clear-logs"
            onClick={onClearLogs}
            disabled={logs.length === 0}
            className="p-1.5 rounded bg-panel border border-border-subtle hover:border-gray-700 text-gray-400 hover:text-brand transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            title="清空日志"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Filters & Options */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center space-y-2 sm:space-y-0 sm:space-x-3 mb-3">
        <div className="relative flex-1">
          <input
            type="text"
            id="input-filter-logs"
            value={filterText}
            onChange={(e) => setFilterText(e.target.value)}
            className="w-full h-8 bg-terminal border border-border-subtle rounded px-3 text-xs text-gray-300 placeholder-gray-600 outline-none focus:border-brand"
            placeholder="过滤日志关键字..."
          />
        </div>
        <label className="flex items-center space-x-1.5 text-xs text-gray-400 select-none cursor-pointer">
          <input
            type="checkbox"
            id="checkbox-autoscroll"
            checked={autoScroll}
            onChange={(e) => setAutoScroll(e.target.checked)}
            className="rounded border-border-subtle bg-panel text-brand h-3.5 w-3.5 accent-brand"
          />
          <span>自动滚动到最新</span>
        </label>
      </div>

      {/* Terminal View */}
      <div 
        id="terminal-box"
        className="flex-1 bg-terminal rounded-lg border border-border-subtle p-4 font-mono text-xs overflow-y-auto space-y-1.5 scrollbar-thin"
      >
        {filteredLogs.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-gray-600 space-y-1 py-10">
            <Terminal className="h-5 w-5 opacity-40 mb-1" />
            <p>控制台就绪。等待音频处理任务...</p>
          </div>
        ) : (
          filteredLogs.map((line, index) => (
            <div key={index} id={`log-line-${index}`} className={`whitespace-pre-wrap leading-relaxed ${getLineColorClass(line)}`}>
              {line}
            </div>
          ))
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
}
