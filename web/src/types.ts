/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface Engine {
  name: string;
}

export type JobStatus = 'pending' | 'running' | 'done' | 'failed' | 'cancelling' | 'cancelled';

export interface JobState {
  job_id: string;
  status: JobStatus;
  progress: number; // 0 to 100
  message: string;
  result_files: string[];
  logs: string[];
}

export interface TranscriptionParams {
  engine: string;
  bpm: number;
  auto_bpm: boolean;
  normalize: boolean;
  preemphasis: boolean;
  velocity_stretch: boolean;
  confidence_threshold: number;
  bp_onset_threshold: number;
  bp_frame_threshold: number;
}
