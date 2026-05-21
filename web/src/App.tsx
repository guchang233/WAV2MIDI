/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  Music, 
  Server, 
  Settings, 
  Play, 
  HelpCircle, 
  Info, 
  Check, 
  RefreshCw,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { Engine, TranscriptionParams, JobStatus, JobState } from './types';
import UploadZone from './components/UploadZone';
import ParameterConfig from './components/ParameterConfig';
import LogPanel from './components/LogPanel';
import JobProgress from './components/JobProgress';

interface MappedResultFile {
  name: string;
  url: string;
}

export default function App() {
  // Backend config
  const [apiBaseUrl, setApiBaseUrl] = useState<string>(() => {
    return localStorage.getItem('audio_midi_api_url') || 'http://localhost:8000';
  });
  const [isUrlSaved, setIsUrlSaved] = useState(false);

  // Engines state
  const [engines, setEngines] = useState<Engine[]>([]);
  const [isLoadingEngines, setIsLoadingEngines] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);

  // Files state
  const [files, setFiles] = useState<File[]>([]);

  // Processing settings
  const [transcribeMode, setTranscribeMode] = useState<'sync' | 'async'>('sync');
  const [params, setParams] = useState<TranscriptionParams>({
    engine: 'Ensemble (PT + BP)',
    bpm: 120.0,
    auto_bpm: false,
    normalize: true,
    preemphasis: false,
    velocity_stretch: true,
    confidence_threshold: 0.2,
    bp_onset_threshold: 0.35,
    bp_frame_threshold: 0.20
  });

  // Transcription active progress states
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [message, setMessage] = useState<string>('');
  const [resultFiles, setResultFiles] = useState<MappedResultFile[]>([]);
  const [logs, setLogs] = useState<string[]>([]);

  // Refs for tracking active WebSocket connections
  const wsRef = useRef<WebSocket | null>(null);

  // Persist API URL automatically
  const saveApiUrl = (url: string) => {
    let cleanUrl = url.trim();
    if (cleanUrl.endsWith('/')) {
      cleanUrl = cleanUrl.slice(0, -1);
    }
    setApiBaseUrl(cleanUrl);
    localStorage.setItem('audio_midi_api_url', cleanUrl);
    setIsUrlSaved(true);
    setTimeout(() => setIsUrlSaved(false), 2000);
    addLog(`📋 [INFO] 更新后端服务器主入口地址: ${cleanUrl}`);
  };

  const addLog = (logLine: string) => {
    const timestamp = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    setLogs(prev => [...prev, `[${timestamp}] ${logLine}`]);
  };

  // Fetch available engines
  const fetchEngines = async (baseUrl: string) => {
    setIsLoadingEngines(true);
    setEngineError(null);
    try {
      addLog(`[INFO] 正在自 ${baseUrl}/api/engines 获取算法列表...`);
      const response = await fetch(`${baseUrl}/api/engines`);
      if (!response.ok) {
        throw new Error(`服务请求失败: ${response.status} ${response.statusText}`);
      }
      const data = await response.json();
      if (Array.isArray(data)) {
        setEngines(data);
        if (data.length > 0) {
          // If available list contains the default, choose it, otherwise pick the first
          const fallbackEngine = data.some(e => e.name === 'Ensemble (PT + BP)')
            ? 'Ensemble (PT + BP)'
            : data[0].name;
          setParams(prev => ({ ...prev, engine: fallbackEngine }));
        }
        addLog(`✅ [SUCCESS] 加载算法列表成功, 获得 ${data.length} 个可用转谱模型！`);
      } else {
        throw new Error('未返回数组类型的算法对象列表');
      }
    } catch (err: any) {
      const msg = err.message || '连接失败';
      setEngineError(msg);
      addLog(`⚠️ [WARNING] 无法自动加载可用的模型算法: ${msg}。已开启默认备用选项。`);
      // Fallbacks
      const fallbackList: Engine[] = [
        { name: 'Ensemble (PT + BP)' },
        { name: 'Piano Transcription (Neural)' },
        { name: 'Basic Pitch' },
        { name: 'Harmonic Salience [DEBUG ONLY]' },
        { name: 'Spectral Peaks [DEBUG ONLY]' }
      ];
      setEngines(fallbackList);
      setParams(prev => ({ ...prev, engine: 'Ensemble (PT + BP)' }));
    } finally {
      setIsLoadingEngines(false);
    }
  };

  // Fetch engines on mount + after url change
  useEffect(() => {
    fetchEngines(apiBaseUrl);
  }, []);

  // Force single-file mode adjustments
  useEffect(() => {
    if (files.length > 1) {
      setTranscribeMode('async');
    }
  }, [files]);

  // Clean WebSocket connection on destruct
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleAddFiles = (newFiles: File[]) => {
    setFiles(prev => [...prev, ...newFiles]);
    addLog(`📋 [INFO] 导入了 ${newFiles.length} 个本地音频切片。当前工作流共有 ${files.length + newFiles.length} 个文件。`);
  };

  const handleRemoveFile = (index: number) => {
    const removedName = files[index]?.name;
    setFiles(prev => prev.filter((_, i) => i !== index));
    if (removedName) {
      addLog(`📋 [INFO] 从待处理中剔除了音频: ${removedName}`);
    }
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  // Launch synchronous transcription
  const startSyncTranscribe = async () => {
    if (files.length === 0) return;
    const file = files[0];

    const fd = new FormData();
    fd.append('file', file);
    fd.append('engine', params.engine);
    fd.append('bpm', params.bpm.toString());
    fd.append('auto_bpm', params.auto_bpm.toString());
    fd.append('normalize', params.normalize.toString());
    fd.append('preemphasis', params.preemphasis.toString());
    fd.append('velocity_stretch', params.velocity_stretch.toString());
    fd.append('confidence_threshold', params.confidence_threshold.toString());
    fd.append('bp_onset_threshold', params.bp_onset_threshold.toString());
    fd.append('bp_frame_threshold', params.bp_frame_threshold.toString());

    setLogs([]);
    addLog(`🚀 [TRANSCRIPTION_START] 启动快速单文件同步极速转录: ${file.name}`);
    addLog(`[PARAM] 引擎: ${params.engine} | 归一化: ${params.normalize} | 自动BPM: ${params.auto_bpm}`);
    
    setJobStatus('running');
    setProgress(15);
    setMessage('上传至后端服务器，正在排队演算中 (神经网络单声道处理时长可能需要十几秒至数分钟)...');
    setResultFiles([]);

    try {
      const response = await fetch(`${apiBaseUrl}/api/transcribe`, {
        method: 'POST',
        body: fd
      });

      if (!response.ok) {
        let errorMsg = '服务内部演算失败';
        try {
          const errData = await response.json();
          errorMsg = errData.error || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      addLog('✅ [SUCCESS] 谱面提取核心计算完成！正在解析下行 MIDI 数据流...');
      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const outName = file.name.replace(/\.[^/.]+$/, "") + ".mid";
      
      setResultFiles([{ name: outName, url: downloadUrl }]);
      setProgress(100);
      setJobStatus('done');
      setMessage('✅ 快速转录转换百分之百完美完成！');
      addLog(`✅ [SUCCESS] 临时 MIDI 文件下载卡片已构建完毕 (${outName})`);
    } catch (err: any) {
      const msg = err.message || '网络连接超时或网关不可达';
      addLog(`❌ [ERROR] 转换终止失败: ${msg}`);
      setJobStatus('failed');
      setMessage(`转换失败: ${msg}`);
    }
  };

  // Launch asynchronous WebSocket transcription
  const startAsyncTranscribe = async () => {
    if (files.length === 0) return;

    const fd = new FormData();
    files.forEach(f => fd.append('files', f)); // note: files (plural) as per API Spec
    fd.append('engine', params.engine);
    fd.append('bpm', params.bpm.toString());
    fd.append('auto_bpm', params.auto_bpm.toString());
    fd.append('normalize', params.normalize.toString());
    fd.append('preemphasis', params.preemphasis.toString());
    fd.append('velocity_stretch', params.velocity_stretch.toString());
    fd.append('confidence_threshold', params.confidence_threshold.toString());
    fd.append('bp_onset_threshold', params.bp_onset_threshold.toString());
    fd.append('bp_frame_threshold', params.bp_frame_threshold.toString());

    setLogs([]);
    addLog(`🚀 [TRANSCRIPTION_START] 启动批量异步队列转录...`);
    addLog(`[PARAM] 文件总数: ${files.length} | 所用引擎: ${params.engine}`);
    
    setJobStatus('pending');
    setProgress(5);
    setMessage('建立排队任务容器并推入批量就绪区...');
    setResultFiles([]);

    try {
      const response = await fetch(`${apiBaseUrl}/api/jobs`, {
        method: 'POST',
        body: fd
      });

      if (!response.ok) {
        let errorMsg = '异步工作任务声明提交失败';
        try {
          const errData = await response.json();
          errorMsg = errData.error || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      const jobId = data.job_id;
      if (!jobId) {
        throw new Error('未返回任务流流水作业 ID (job_id)');
      }

      addLog(`📋 [INFO] 创建排队批流成功! 流水 ID: ${jobId}`);
      setProgress(10);
      setJobStatus('running');
      setMessage('等待建立实时状态信息推送通道 (WebSocket)...');

      connectWebSocket(jobId);

    } catch (err: any) {
      const msg = err.message || '网络无法写入网域';
      addLog(`❌ [ERROR] 无法启动转换: ${msg}`);
      setJobStatus('failed');
      setMessage(`排队提取失败: ${msg}`);
    }
  };

  // WebSocket progress receiver logic
  const connectWebSocket = (jobId: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    // Determine WS protocol
    const wsProto = apiBaseUrl.startsWith('https') ? 'wss:' : 'ws:';
    const wsHost = apiBaseUrl.replace(/^https?:\/\//, '');
    const wsUrl = `${wsProto}//${wsHost}/ws/${jobId}`;

    addLog(`📋 [INFO] 正在连接 WebSocket 推送流: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      addLog(`📋 [INFO] 状态握手完成，实时消息监听已激活。`);
    };

    ws.onmessage = (e) => {
      try {
        const d: JobState = JSON.parse(e.data);
        
        // Match response to view states
        if (d.progress !== undefined) setProgress(d.progress);
        if (d.message) setMessage(d.message);
        if (d.status) setJobStatus(d.status);

        // Appending latest logs safely
        if (Array.isArray(d.logs)) {
          d.logs.forEach(singleLine => {
            // Strip datetime strings or double lines if they exist, but generally let's log them beautifully
            addLog(singleLine);
          });
        }

        // Set result files if they are in the message
        if (d.result_files && d.result_files.length > 0) {
          const mapped = d.result_files.map(filename => ({
            name: filename,
            url: `${apiBaseUrl}/api/jobs/${jobId}/download/${encodeURIComponent(filename)}`
          }));
          setResultFiles(mapped);
        }

        // Complete state action
        if (d.status === 'done') {
          addLog(`✅ [SUCCESS] 恭喜！当前流水账中任务全部完成。共转换出来 ${d.result_files?.length || 0} 个 MIDI 谱面文件。`);
          ws.close();
        } else if (d.status === 'failed') {
          addLog(`❌ [ERROR] 服务端流程异常终止: ${d.message}`);
          ws.close();
        } else if (d.status === 'cancelled') {
          addLog(`⚠️ [WARNING] 任务响应客户端控制，已主动断开该流水提取。`);
          ws.close();
        }

      } catch (err) {
        console.error('WebSocket payload parse error', err);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error', err);
      addLog('⚠️ [WARNING] WebSocket 遇到中断或连通错误。尝试通过常规拉取接口备用。');
    };

    ws.onclose = (ev) => {
      addLog(`📋 [INFO] WebSocket 监听流正常关闭 (代码: ${ev.code})`);
      wsRef.current = null;
    };
  };

  // Cancel running tasks
  const handleCancelTask = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      addLog('📋 [INFO] 发出取消信号 "cancel" 给服务器...');
      wsRef.current.send('cancel');
      setJobStatus('cancelling');
    } else {
      // For synchronous, reset manually
      addLog('📋 [INFO] 任务转换已主动停止重置。');
      setJobStatus(null);
      setProgress(0);
      setMessage('');
    }
  };

  const handleStartTranscribe = () => {
    if (files.length === 0) return;
    if (transcribeMode === 'sync') {
      startSyncTranscribe();
    } else {
      startAsyncTranscribe();
    }
  };

  return (
    <div className="min-h-screen flex flex-col selection:bg-brand/30 selection:text-white" id="root-layout-wrapper">
      {/* Banner / Header Bar */}
      <header className="border-b border-border-subtle bg-panel/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4" id="applet-header">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4">
          <div className="flex items-center space-x-3.5">
            <div className="p-2.5 rounded-xl bg-brand/10 border border-brand/20 text-brand scale-100 hover:scale-105 transition-all">
              <Music className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight text-white flex items-center space-x-2">
                <span>AI 音频转 MIDI 提取器</span>
                <span className="text-[10px] bg-brand/10 text-brand px-2 py-0.5 rounded-full border border-brand/20 uppercase font-mono tracking-widest font-bold">PRO v2.5</span>
              </h1>
              <p className="text-xs text-gray-400 mt-0.5">用户上传音频文件 → AI 自动识别并转录为高质量标准 MIDI 格式</p>
            </div>
          </div>

          {/* Connection URL Address config */}
          <div className="flex items-center space-x-2 bg-terminal border border-border-subtle px-3 py-1.5 rounded-xl self-start md:self-auto" id="endpoint-config">
            <Server className="h-4 w-4 text-gray-500" />
            <input
              type="text"
              id="input-api-url"
              className="bg-transparent text-xs text-gray-300 w-44 md:w-56 outline-none font-mono border-none placeholder-gray-700"
              placeholder="API 网址, 例: http://127.0.0.1:8000"
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
            />
            <button
              type="button"
              id="save-api-url"
              onClick={() => {
                saveApiUrl(apiBaseUrl);
                fetchEngines(apiBaseUrl);
              }}
              className="text-[11px] bg-panel hover:bg-brand text-gray-400 hover:text-white border border-border-subtle hover:border-brand px-2.5 py-1 rounded-md cursor-pointer transition-all font-semibold flex items-center space-x-1"
            >
              {isUrlSaved ? <Check className="h-3 w-3 text-green-400" /> : <RefreshCw className="h-3 w-3" />}
              <span>保存地址</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Responsive Grid Layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6" id="dashboard-main">
        {/* Left Bento: Upload & Configuration Parameters */}
        <section className="lg:col-span-5 space-y-6 flex flex-col" id="left-column">
          {/* File Upload Zone */}
          <div className="bg-panel/40 border border-border-subtle rounded-xl p-5" id="deck-uploader">
            <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-4 flex items-center space-x-1.5">
              <span>第一步: 导入待转换音轨</span>
            </h2>
            <UploadZone
              files={files}
              onAddFiles={handleAddFiles}
              onRemoveFile={handleRemoveFile}
            />
          </div>

          {/* Parameter Settings */}
          <ParameterConfig
            engines={engines}
            params={params}
            setParams={setParams}
            isLoadingEngines={isLoadingEngines}
          />
        </section>

        {/* Right Bento: Interactive Control, Progress Tracker, Output log */}
        <section className="lg:col-span-7 space-y-6 flex flex-col" id="right-column">
          {/* Transcription Launcher Deck */}
          <div className="bg-panel/40 border border-border-subtle rounded-xl p-5 space-y-5" id="engine-launcher">
            <div className="flex items-center justify-between border-b border-border-subtle pb-3">
              <h2 className="text-xs font-bold text-gray-400 uppercase tracking-widest flex items-center space-x-1.5">
                <span>第二步: 模式选择与提取</span>
              </h2>

              <div className="flex items-center space-x-2" id="toggle-modes">
                <span className="text-[11px] text-gray-500 font-medium">处理模式:</span>
                <div className="bg-terminal border border-border-subtle p-0.5 rounded-lg flex">
                  <button
                    type="button"
                    id="btn-mode-sync"
                    disabled={files.length > 1}
                    onClick={() => setTranscribeMode('sync')}
                    className={`text-[10px] uppercase font-bold tracking-wider px-2.5 py-1.5 rounded-md transition-all ${
                      transcribeMode === 'sync'
                        ? 'bg-panel border border-border-subtle text-brand font-semibold'
                        : 'text-gray-500 hover:text-gray-300 disabled:opacity-30'
                    }`}
                    title={files.length > 1 ? "多文件不支持同步模式" : "单文件快速直出"}
                  >
                    同步直出
                  </button>
                  <button
                    type="button"
                    id="btn-mode-async"
                    onClick={() => setTranscribeMode('async')}
                    className={`text-[10px] uppercase font-bold tracking-wider px-2.5 py-1.5 rounded-md transition-all ${
                      transcribeMode === 'async'
                        ? 'bg-panel border border-border-subtle text-brand font-semibold'
                        : 'text-gray-500 hover:text-gray-300'
                    }`}
                    title="非阻塞多任务批量队列"
                  >
                    异步队列
                  </button>
                </div>
              </div>
            </div>

            {/* Launch CTA */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4 bg-terminal/40 p-4 rounded-xl border border-border-subtle/50" id="deck-cta">
              <div className="text-left">
                <p className="text-xs text-gray-400">
                  {files.length === 0 ? (
                    <span className="text-gray-600 font-medium">请先导入本地音频文件进行算法识别。</span>
                  ) : (
                    <span>
                      已就绪 <strong className="text-brand font-mono">{files.length}</strong> 个音轨，转录模式为{' '}
                      <strong className="text-brand">
                        {transcribeMode === 'sync' ? '【同步直出】' : '【异步批量队列】'}
                      </strong>
                    </span>
                  )}
                </p>
                <p className="text-[10px] text-gray-500 mt-0.5">
                  通过 AI 识别分析音高、节奏，生成可自由编辑、分轨的优质 MIDI 文件。
                </p>
              </div>

              <button
                type="button"
                id="btn-launch-transcription"
                onClick={handleStartTranscribe}
                disabled={files.length === 0 || jobStatus === 'running' || jobStatus === 'pending'}
                className="h-10 px-6 font-bold select-none text-xs rounded-lg uppercase tracking-widest bg-brand hover:bg-brand-hover disabled:bg-gray-800 disabled:opacity-40 border border-brand/25 disabled:cursor-not-allowed text-white flex items-center justify-center space-x-2 transition-all shadow-md active:scale-95 shrink-0"
              >
                {jobStatus === 'running' || jobStatus === 'pending' ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin text-white" />
                    <span>转谱处理中...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 text-white fill-white" />
                    <span>开始 AI 转谱</span>
                  </>
                )}
              </button>
            </div>

            {/* Active Progress Monitor / Fallback log wrapper */}
            {jobStatus && (
              <JobProgress
                status={jobStatus}
                progress={progress}
                message={message}
                resultFiles={resultFiles}
                onCancel={handleCancelTask}
              />
            )}
          </div>

          {/* Interactive Debug Terminal Log */}
          <LogPanel
            logs={logs}
            onClearLogs={handleClearLogs}
          />
        </section>
      </main>

      {/* Aesthetic minimalistic credit footer */}
      <footer className="border-t border-border-subtle bg-sidebar py-3 px-6 text-center text-[10px] text-gray-600 font-mono tracking-wider mt-auto" id="applet-footer">
        POWERED BY DEEP NEURAL PIANO TRANSCRIPTION AND BASIC PITCH • FULLY DESK INTEGRATED
      </footer>
    </div>
  );
}
