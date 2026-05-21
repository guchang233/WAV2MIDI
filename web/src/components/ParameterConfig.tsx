/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { SlidersHorizontal, Sliders, Music, Info, HelpCircle } from 'lucide-react';
import { TranscriptionParams, Engine } from '../types';

interface ParameterConfigProps {
  engines: Engine[];
  params: TranscriptionParams;
  setParams: React.Dispatch<React.SetStateAction<TranscriptionParams>>;
  isLoadingEngines: boolean;
}

export default function ParameterConfig({ engines, params, setParams, isLoadingEngines }: ParameterConfigProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleParamChange = <K extends keyof TranscriptionParams>(key: K, value: TranscriptionParams[K]) => {
    setParams(prev => ({
      ...prev,
      [key]: value
    }));
  };

  return (
    <div className="bg-panel/40 border border-border-subtle rounded-xl p-5 space-y-5" id="parameter-config-panel">
      <div className="flex items-center justify-between border-b border-border-subtle pb-3">
        <div className="flex items-center space-x-2">
          <Sliders className="h-5 w-5 text-brand" />
          <h2 className="text-sm font-semibold text-gray-200 uppercase tracking-widest">转谱参数配置</h2>
        </div>
        <button
          type="button"
          id="toggle-advanced-btn"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-xs flex items-center space-x-1.5 text-gray-400 hover:text-brand transition-colors bg-panel border border-border-subtle px-3 py-1.5 rounded-lg font-medium"
        >
          <SlidersHorizontal className="h-3.5 w-3.5" />
          <span>{showAdvanced ? '隐藏高级设置' : '高级参数'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4" id="core-params-grid">
        {/* Engine Selection */}
        <div className="space-y-1.5" id="engine-field-group">
          <label className="text-xs font-semibold text-gray-400 flex items-center space-x-1">
            <span>转谱算法引擎 (Engine)</span>
            <span className="group relative cursor-help">
              <Info className="h-3.5 w-3.5 text-gray-500" />
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-terminal border border-border-subtle text-[11px] text-gray-300 rounded p-2 w-48 z-20 shadow-lg leading-relaxed">
                Ensemble (PT + BP) 是推荐算法，融合了神经网络钢琴检测和 Basic Pitch。
              </span>
            </span>
          </label>
          <div className="relative">
            {isLoadingEngines ? (
              <div className="w-full h-10 bg-panel border border-border-subtle rounded-lg px-3 animate-pulse flex items-center text-xs text-gray-500">
                正在加载引擎列表...
              </div>
            ) : (
              <select
                id="select-engine"
                value={params.engine}
                onChange={(e) => handleParamChange('engine', e.target.value)}
                className="w-full h-10 bg-panel border border-border-subtle hover:border-gray-700 rounded-lg px-3 text-sm text-gray-200 outline-none focus:border-brand transition-all font-medium appearance-none cursor-pointer"
              >
                {engines.length === 0 ? (
                  <option value="Ensemble (PT + BP)">Ensemble (PT + BP) [默认]</option>
                ) : (
                  engines.map(engine => (
                    <option key={engine.name} value={engine.name}>
                      {engine.name}
                    </option>
                  ))
                )}
              </select>
            )}
            <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-gray-400 text-xs">▼</div>
          </div>
        </div>

        {/* BPM & Auto BPM */}
        <div className="space-y-1.5" id="bpm-field-group">
          <div className="flex justify-between items-center">
            <label className="text-xs font-semibold text-gray-400">速度 (BPM)</label>
            <label className="flex items-center space-x-1 text-xs cursor-pointer select-none">
              <input
                type="checkbox"
                id="checkbox-auto-bpm"
                checked={params.auto_bpm}
                onChange={(e) => handleParamChange('auto_bpm', e.target.checked)}
                className="rounded border-border-subtle bg-panel text-brand focus:ring-0 focus:ring-offset-0 h-3.5 w-3.5 cursor-pointer accent-brand"
              />
              <span className="text-brand font-medium">自动检测 BPM</span>
            </label>
          </div>
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-xs font-semibold">BPM</span>
            <input
              type="number"
              id="input-bpm"
              min="30"
              max="400"
              disabled={params.auto_bpm}
              value={params.bpm}
              onChange={(e) => handleParamChange('bpm', parseFloat(e.target.value) || 120)}
              className="w-full h-10 bg-panel border border-border-subtle hover:border-gray-700 disabled:opacity-50 disabled:hover:border-border-subtle disabled:cursor-not-allowed rounded-lg pl-12 pr-3 text-sm text-gray-200 font-mono outline-none focus:border-brand transition-all"
              placeholder="120"
            />
          </div>
        </div>
      </div>

      {/* Advanced Settings Section */}
      {showAdvanced && (
        <div
          id="advanced-params-section"
          className="pt-4 border-t border-border-subtle/50 space-y-5 animate-in fade-in slide-in-from-top-2 duration-300"
        >
          {/* Signal pre-processing */}
          <div className="space-y-3">
            <h3 className="text-xs font-bold text-gray-400/80 uppercase tracking-widest">预处理设置 (Pre-processing)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Normalize Toggle */}
              <label className="flex items-start p-3 bg-panel/30 border border-border-subtle/50 hover:border-border-subtle rounded-lg cursor-pointer transition-colors">
                <input
                  type="checkbox"
                  id="checkbox-normalize"
                  checked={params.normalize}
                  onChange={(e) => handleParamChange('normalize', e.target.checked)}
                  className="mt-0.5 rounded border-border-subtle bg-panel text-brand h-4 w-4 accent-brand cursor-pointer"
                />
                <div className="ml-2.5">
                  <span className="text-xs font-bold text-gray-300 block">自动音量归一化</span>
                  <span className="text-[11px] text-gray-500 block mt-0.5">将音频输入最大化以提升静音处音符识别率（推荐开启）</span>
                </div>
              </label>

              {/* Preemphasis Toggle */}
              <label className="flex items-start p-3 bg-panel/30 border border-border-subtle/50 hover:border-border-subtle rounded-lg cursor-pointer transition-colors">
                <input
                  type="checkbox"
                  id="checkbox-preemphasis"
                  checked={params.preemphasis}
                  onChange={(e) => handleParamChange('preemphasis', e.target.checked)}
                  className="mt-0.5 rounded border-border-subtle bg-panel text-brand h-4 w-4 accent-brand cursor-pointer"
                />
                <div className="ml-2.5">
                  <span className="text-xs font-bold text-gray-300 block">高频预加重 (Pre-emphasis)</span>
                  <span className="text-[11px] text-gray-500 block mt-0.5">增强声音高频成分。注: 神经网络(PT)引擎将自动忽略此项设置</span>
                </div>
              </label>
            </div>
          </div>

          {/* Model Post-processing */}
          <div className="space-y-4">
            <h3 className="text-xs font-bold text-gray-400/80 uppercase tracking-widest">后处理阈值 (Post-processing Options)</h3>
            
            <div className="space-y-4">
              {/* Velocity Stretch Toggle */}
              <div className="flex items-start p-3 bg-panel/30 border border-border-subtle/50 hover:border-border-subtle rounded-lg">
                <label className="flex items-start cursor-pointer select-none">
                  <input
                    type="checkbox"
                    id="checkbox-velocity-stretch"
                    checked={params.velocity_stretch}
                    onChange={(e) => handleParamChange('velocity_stretch', e.target.checked)}
                    className="mt-0.5 rounded border-border-subtle bg-panel text-brand h-4 w-4 accent-brand cursor-pointer"
                  />
                  <div className="ml-2.5">
                    <span className="text-xs font-bold text-gray-300 block">力度拉伸 (Velocity Stretch)</span>
                    <span className="text-[11px] text-gray-500 block mt-0.5">根据幅度信息拉伸 MIDI 音符力度，增强转谱表现力</span>
                  </div>
                </label>
              </div>

              {/* Confidence Threshold Slider */}
              <div className="space-y-1.5 p-3 bg-panel/20 border border-border-subtle/30 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-gray-300 flex items-center space-x-1">
                    <span>置信度阈值 (Confidence Threshold)</span>
                    <span className="group relative cursor-help">
                      <HelpCircle className="h-3 w-3 text-gray-500" />
                      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block bg-terminal border border-border-subtle text-[10px] text-gray-300 rounded p-2 w-48 z-20 shadow-lg leading-relaxed normal-case">
                        用于过滤 Basic Pitch 算法补充音符的阈值。较低的阈值会生成更多音符，较高的阈值较为严谨。
                      </span>
                    </span>
                  </span>
                  <span className="text-xs font-mono text-brand font-semibold">{params.confidence_threshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  id="slider-confidence-threshold"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={params.confidence_threshold}
                  onChange={(e) => handleParamChange('confidence_threshold', parseFloat(e.target.value))}
                  className="w-full h-1 bg-border-subtle rounded-lg appearance-none cursor-pointer accent-brand"
                />
                <div className="flex justify-between text-[10px] text-gray-600 font-mono">
                  <span>0.0 (宽松)</span>
                  <span>0.2 (默认)</span>
                  <span>1.0 (极严)</span>
                </div>
              </div>

              {/* Basic Pitch Specifics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Onset Threshold */}
                <div className="space-y-1.5 p-3 bg-panel/20 border border-border-subtle/30 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-bold text-gray-300">Basic Pitch 起音判定</span>
                    <span className="text-xs font-mono text-brand font-semibold">{params.bp_onset_threshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    id="slider-bp-onset-threshold"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={params.bp_onset_threshold}
                    onChange={(e) => handleParamChange('bp_onset_threshold', parseFloat(e.target.value))}
                    className="w-full h-1 bg-border-subtle rounded-lg appearance-none cursor-pointer accent-brand"
                  />
                  <div className="flex justify-between text-[10px] text-gray-600 font-mono">
                    <span>0.1 (易触发)</span>
                    <span>0.35 (默认)</span>
                    <span>0.9 (难触发)</span>
                  </div>
                </div>

                {/* Frame Threshold */}
                <div className="space-y-1.5 p-3 bg-panel/20 border border-border-subtle/30 rounded-lg">
                  <div className="flex justify-between items-center">
                    <span className="text-xs font-bold text-gray-300">Basic Pitch 持续判定</span>
                    <span className="text-xs font-mono text-brand font-semibold">{params.bp_frame_threshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    id="slider-bp-frame-threshold"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={params.bp_frame_threshold}
                    onChange={(e) => handleParamChange('bp_frame_threshold', parseFloat(e.target.value))}
                    className="w-full h-1 bg-border-subtle rounded-lg appearance-none cursor-pointer accent-brand"
                  />
                  <div className="flex justify-between text-[10px] text-gray-600 font-mono">
                    <span>0.1 (易冗长)</span>
                    <span>0.20 (默认)</span>
                    <span>0.9 (易截断)</span>
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>
      )}
    </div>
  );
}
