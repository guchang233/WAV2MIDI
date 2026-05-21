/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useRef, useState } from 'react';
import { Upload, FileAudio, Trash2, Play, Pause, Music } from 'lucide-react';

interface UploadZoneProps {
  files: File[];
  onAddFiles: (newFiles: File[]) => void;
  onRemoveFile: (index: number) => void;
}

export default function UploadZone({ files, onAddFiles, onRemoveFile }: UploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [playingIndex, setPlayingIndex] = useState<number | null>(null);
  const audioRefs = useRef<{ [key: number]: HTMLAudioElement | null }>({});

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files) {
      const droppedFiles = (Array.from(e.dataTransfer.files) as File[]).filter(file => 
        file.type.startsWith('audio/') || 
        ['.mp3', '.wav', '.flac', '.ogg', '.m4a'].some(ext => file.name.toLowerCase().endsWith(ext))
      );
      if (droppedFiles.length > 0) {
        onAddFiles(droppedFiles);
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      onAddFiles(Array.from(e.target.files));
    }
  };

  const triggerSelect = () => {
    fileInputRef.current?.click();
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const togglePlayAudio = (index: number, file: File) => {
    const existingAudio = audioRefs.current[index];

    if (playingIndex === index) {
      existingAudio?.pause();
      setPlayingIndex(null);
    } else {
      // Pause other playing audios
      if (playingIndex !== null && audioRefs.current[playingIndex]) {
        audioRefs.current[playingIndex]?.pause();
      }

      if (!existingAudio) {
        const url = URL.createObjectURL(file);
        const audio = new Audio(url);
        audio.onended = () => setPlayingIndex(null);
        audioRefs.current[index] = audio;
        audio.play();
      } else {
        existingAudio.play();
      }
      setPlayingIndex(index);
    }
  };

  const handleRemove = (index: number) => {
    if (playingIndex === index) {
      audioRefs.current[index]?.pause();
      setPlayingIndex(null);
    }
    delete audioRefs.current[index];
    onRemoveFile(index);
  };

  return (
    <div className="space-y-4" id="upload-zone-container">
      <div
        id="drag-drop-panel"
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={triggerSelect}
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 flex flex-col items-center justify-center min-h-[200px] ${
          isDragOver
            ? 'border-brand bg-brand/5 scale-[1.01]'
            : 'border-border-subtle hover:border-gray-600 bg-panel/30 hover:bg-panel/50'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a"
          className="hidden"
          onChange={handleFileChange}
        />
        <div className="p-4 rounded-full bg-panel/80 border border-border-subtle mb-4">
          <Upload className={`h-8 w-8 text-brand ${isDragOver ? 'animate-bounce' : ''}`} />
        </div>
        <h3 className="text-sm font-semibold text-gray-200">
          拖拽音频文件到此处，或 <span className="text-brand font-bold hover:underline">浏览文件</span>
        </h3>
        <p className="text-xs text-gray-500 mt-2">
          支持 WAV, FLAC, MP3, OGG, M4A 格式音频文件
        </p>
      </div>

      {files.length > 0 && (
        <div className="space-y-2" id="file-list-section">
          <div className="flex items-center justify-between text-xs text-gray-400 px-1 font-semibold uppercase tracking-wider">
            <span>待处理文件 ({files.length})</span>
            <span className="text-gray-500">双击或点击播放预览</span>
          </div>

          <div className="max-h-[220px] overflow-y-auto space-y-2 pr-1" id="file-items-scrollable">
            {files.map((file, index) => (
              <div
                key={`${file.name}-${index}`}
                id={`file-item-${index}`}
                className="flex items-center justify-between p-3 rounded-lg bg-panel/60 border border-border-subtle hover:border-gray-800 transition-all duration-200"
              >
                <div className="flex items-center space-x-3 truncate">
                  <div className="relative">
                    <button
                      type="button"
                      id={`play-preview-${index}`}
                      onClick={(e) => {
                        e.stopPropagation();
                        togglePlayAudio(index, file);
                      }}
                      className="p-2 rounded-md bg-panel/90 border border-border-subtle text-brand hover:scale-105 transition-all text-xs"
                      title={playingIndex === index ? "暂停预览" : "播放预览"}
                    >
                      {playingIndex === index ? (
                        <Pause className="h-4 w-4 animate-pulse text-brand" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                    </button>
                  </div>

                  <div className="truncate">
                    <p className="text-sm font-medium text-gray-300 truncate" title={file.name}>
                      {file.name}
                    </p>
                    <div className="flex items-center space-x-2 mt-0.5">
                      <span className="text-xs font-mono text-gray-500">
                        {formatSize(file.size)}
                      </span>
                      <span className="text-[10px] bg-panel border border-border-subtle text-gray-400 px-1.5 py-0.5 rounded uppercase">
                        {file.name.split('.').pop()}
                      </span>
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  id={`remove-file-${index}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemove(index);
                  }}
                  className="p-2 text-gray-500 hover:text-brand hover:bg-brand/5 rounded-md transition-colors"
                  title="移除文件"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
