/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Download, Loader2, CheckCircle2, AlertTriangle, XCircle, FileSpreadsheet, PlayCircle } from 'lucide-react';
import { JobStatus } from '../types';

interface ResultFile {
  name: string;
  url: string; // Blob URL or API download URL
}

interface JobProgressProps {
  status: JobStatus;
  progress: number;
  message: string;
  resultFiles: ResultFile[];
  onCancel: () => void;
}

export default function JobProgress({ status, progress, message, resultFiles, onCancel }: JobProgressProps) {
  const getStatusConfig = () => {
    switch (status) {
      case 'pending':
        return {
          icon: <Loader2 className="h-5 w-5 text-sky-400 animate-spin" />,
          title: '已提交，排队中...',
          bgColor: 'bg-sky-500/10 border-sky-500/20',
          textColor: 'text-sky-400'
        };
      case 'running':
        return {
          icon: <Loader2 className="h-5 w-5 text-brand animate-spin" />,
          title: '正在执行 AI 谱面转录...',
          bgColor: 'bg-brand/10 border-brand/20',
          textColor: 'text-brand'
        };
      case 'done':
        return {
          icon: <CheckCircle2 className="h-5 w-5 text-green-400" />,
          title: '转换已全部完成！',
          bgColor: 'bg-green-500/10 border-green-500/20',
          textColor: 'text-green-400'
        };
      case 'failed':
        return {
          icon: <AlertTriangle className="h-5 w-5 text-red-400" />,
          title: '处理失败',
          bgColor: 'bg-red-500/10 border-red-500/20',
          textColor: 'text-red-400'
        };
      case 'cancelling':
        return {
          icon: <Loader2 className="h-5 w-5 text-yellow-400 animate-spin" />,
          title: '正在强行取消任务...',
          bgColor: 'bg-yellow-500/10 border-yellow-500/20',
          textColor: 'text-yellow-400'
        };
      case 'cancelled':
        return {
          icon: <XCircle className="h-5 w-5 text-gray-400" />,
          title: '任务已取消',
          bgColor: 'bg-gray-500/10 border-gray-500/20',
          textColor: 'text-gray-400'
        };
      default:
        return null;
    }
  };

  const config = getStatusConfig();
  if (!config) return null;

  return (
    <div id="job-progress-container" className="space-y-4">
      {/* Status Bar Banner */}
      <div className={`p-4 rounded-xl border ${config.bgColor} flex flex-col md:flex-row md:items-center justify-between gap-3`}>
        <div className="flex items-center space-x-3">
          {config.icon}
          <div>
            <h3 className="text-sm font-semibold text-gray-200">{config.title}</h3>
            {message && <p className="text-xs text-gray-400 mt-0.5">{message}</p>}
          </div>
        </div>

        {/* Cancel Action Button */}
        {(status === 'running' || status === 'pending') && (
          <button
            type="button"
            id="btn-force-cancel-job"
            onClick={onCancel}
            className="text-xs font-semibold uppercase tracking-wider bg-brand/10 hover:bg-brand text-brand hover:text-white border border-brand/20 px-3.5 py-1.5 rounded-lg transition-all"
          >
            强制取消任务
          </button>
        )}
      </div>

      {/* Progress Slide Bar */}
      <div className="bg-panel/40 border border-border-subtle rounded-xl p-5 space-y-3" id="progress-indicator-box">
        <div className="flex justify-between items-center text-xs">
          <span className="text-gray-400 font-bold uppercase tracking-wider">阶段处理进度</span>
          <span className="text-brand font-mono font-bold text-sm">{progress.toFixed(0)}%</span>
        </div>

        {/* Progress Bar Track */}
        <div className="h-2 w-full bg-terminal rounded-full overflow-hidden border border-border-subtle/50">
          <div
            id="active-progress-fill"
            className="h-full bg-brand rounded-full transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      </div>

      {/* Expiry download list */}
      {resultFiles.length > 0 && (
        <div className="space-y-2.5" id="download-results-section">
          <h3 className="text-xs font-bold text-gray-400 px-1 uppercase tracking-wider">
            转换成功的 MIDI 文件 ({resultFiles.length})
          </h3>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3" id="midi-results-grid">
            {resultFiles.map((file, index) => (
              <div
                key={index}
                id={`result-item-${index}`}
                className="flex items-center justify-between p-3.5 rounded-xl bg-panel border border-border-subtle hover:border-gray-800 transition-all group scale-100 hover:scale-[1.01]"
              >
                <div className="flex items-center space-x-3 truncate">
                  <div className="p-2 ml-1 rounded-lg bg-green-500/10 text-green-400 border border-green-500/20">
                    <FileSpreadsheet className="h-5 w-5" />
                  </div>
                  <div className="truncate">
                    <p className="text-sm font-semibold text-gray-200 truncate" title={file.name}>
                      {file.name}
                    </p>
                    <p className="text-[10px] text-gray-500 font-mono mt-0.5">MIDI Format 1</p>
                  </div>
                </div>

                <a
                  href={file.url}
                  id={`link-download-midi-${index}`}
                  download={file.name}
                  className="flex items-center justify-center p-2 rounded-lg bg-terminal border border-border-subtle hover:border-brand text-gray-400 hover:text-brand transition-all"
                  title="下载 MIDI 文件"
                >
                  <Download className="h-4 w-4" />
                </a>
              </div>
            ))}
          </div>

          <p className="text-[10px] text-gray-500 text-center mt-2.5 leading-relaxed">
            💡 提示: MIDI 文件是通用格式。下载完成后，您可以直接将其拖入任意 DAW（如 FL Studio, Ableton Live, Logic Pro）或使用外部 MIDI 播放器打开预览。
          </p>
        </div>
      )}
    </div>
  );
}
