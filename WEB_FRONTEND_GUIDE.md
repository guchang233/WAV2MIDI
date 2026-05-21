# Web 前端开发指导文档

本文档面向 AI 或开发者，指导构建「音频转MIDI」的 Web 前端界面。后端 API 已就绪，前端只需调用接口即可。

---

## 1. 项目概述

**功能**：用户上传音频文件 → 后端执行 AI 转谱 → 返回 MIDI 文件供下载

**两种使用模式**：
- **同步模式**：单文件，阻塞等待，直接返回 MIDI 二进制
- **异步模式**：多文件批量，非阻塞，通过 WebSocket 实时推送进度

**技术栈**：后端 FastAPI + WebSocket，前端任意（React/Vue/纯 HTML 均可）

---

## 2. 后端启动

```bash
python run_web.py --host 0.0.0.0 --port 8000
```

默认监听 `http://127.0.0.1:8000`，已启用 CORS（允许任意源）。

---

## 3. API 接口完整规范

### 3.1 `GET /api/engines` — 获取可用引擎

**用途**：填充前端引擎下拉框

**响应** `200`：
```json
[
  {"name": "Piano Transcription (Neural)"},
  {"name": "Basic Pitch"},
  {"name": "Ensemble (PT + BP)"},
  {"name": "Harmonic Salience [DEBUG ONLY]"},
  {"name": "Spectral Peaks [DEBUG ONLY]"}
]
```

引擎可用性取决于服务器环境是否安装了对应模型。前端应动态获取此列表，不要硬编码。

---

### 3.2 `POST /api/transcribe` — 同步单文件转谱

**用途**：单文件快速转换，阻塞等待结果

**请求**：`multipart/form-data`

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | ✅ | — | 音频文件（WAV/FLAC/MP3/OGG/M4A） |
| `engine` | string | — | `"Ensemble (PT + BP)"` | 引擎名称，从 `/api/engines` 获取 |
| `bpm` | float | — | `120.0` | 速度，30-400 |
| `auto_bpm` | bool | — | `false` | 自动检测BPM |
| `normalize` | bool | — | `true` | 音量归一化（推荐开启） |
| `preemphasis` | bool | — | `false` | 高频预加重（神经网络引擎会自动忽略） |
| `velocity_stretch` | bool | — | `true` | 力度拉伸 |
| `confidence_threshold` | float | — | `0.2` | 置信度阈值（0.5以下过滤BP补充音符） |
| `bp_onset_threshold` | float | — | `0.35` | Basic Pitch 起音判定（0.1-0.9） |
| `bp_frame_threshold` | float | — | `0.20` | Basic Pitch 持续断点（0.1-0.9） |

**成功响应** `200`：`Content-Type: audio/midi`，直接返回 MIDI 文件二进制

**失败响应** `400`：
```json
{"error": "引擎不可用: xxx"}
```

**cURL 示例**：
```bash
curl -X POST http://localhost:8000/api/transcribe \
  -F "file=@piano.wav" \
  -F "engine=Ensemble (PT + BP)" \
  -F "bpm=120" \
  -F "normalize=true" \
  -o output.mid
```

**JS fetch 示例**：
```js
const fd = new FormData();
fd.append('file', audioFile);
fd.append('engine', 'Ensemble (PT + BP)');
fd.append('bpm', '120');

const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
if (res.ok) {
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'output.mid';
  a.click();
}
```

---

### 3.3 `POST /api/jobs` — 异步批量转谱

**用途**：多文件批量处理，非阻塞，立即返回任务ID

**请求**：`multipart/form-data`，参数同 `/api/transcribe`，但文件字段为 `files`（复数）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | ✅ | 多个音频文件 |
| 其余字段 | 同上 | — | 同 `/api/transcribe` |

**成功响应** `200`：
```json
{"job_id": "a1b2c3d4e5f6"}
```

**JS 示例**：
```js
const fd = new FormData();
audioFiles.forEach(f => fd.append('files', f));
fd.append('engine', 'Ensemble (PT + BP)');
fd.append('bpm', '120');

const res = await fetch('/api/jobs', { method: 'POST', body: fd });
const { job_id } = await res.json();
```

---

### 3.4 `GET /api/jobs/{job_id}` — 查询任务状态

**用途**：轮询方式获取进度（推荐用 WebSocket 替代）

**响应** `200`：
```json
{
  "job_id": "a1b2c3d4e5f6",
  "status": "running",
  "progress": 50,
  "message": "2/3",
  "result_files": ["song1.mid"],
  "logs": [
    "[12:00:01] 处理 [1/3]: song1.wav",
    "[12:00:15] ✅ 完成: song1.mid (342 音符)",
    "[12:00:16] 处理 [2/3]: song2.wav"
  ]
}
```

**status 枚举值**：

| 值 | 含义 |
|----|------|
| `pending` | 刚创建，尚未开始 |
| `running` | 处理中 |
| `done` | 全部完成 |
| `failed` | 失败（message 含原因） |
| `cancelling` | 正在取消 |
| `cancelled` | 已取消 |

**404 响应**：`{"error": "任务不存在"}`

---

### 3.5 `WS /ws/{job_id}` — WebSocket 实时进度

**用途**：实时接收任务状态更新，比轮询更高效

**连接**：
```
ws://localhost:8000/ws/a1b2c3d4e5f6
```

**服务端推送消息**（JSON，格式同 GET 状态）：
```json
{
  "status": "running",
  "progress": 50,
  "message": "2/3",
  "result_files": ["song1.mid"],
  "logs": ["最近3条日志"]
}
```

**客户端发送**：
- `"cancel"` — 终止任务

**JS 示例**：
```js
const ws = new WebSocket(`ws://${location.host}/ws/${jobId}`);

ws.onmessage = (e) => {
  const d = JSON.parse(e.data);
  updateProgressBar(d.progress);
  updateStatusText(d.message);
  appendLogs(d.logs);

  if (d.status === 'done') {
    renderDownloadLinks(jobId, d.result_files);
    ws.close();
  } else if (d.status === 'failed') {
    showError(d.message);
    ws.close();
  }
};

// 取消任务
cancelButton.onclick = () => ws.send('cancel');
```

---

### 3.6 `GET /api/jobs/{job_id}/download/{filename}` — 下载结果

**用途**：下载转谱完成的 MIDI 文件

**URL 示例**：
```
GET /api/jobs/a1b2c3d4e5f6/download/song1.mid
```

**成功响应** `200`：
- `Content-Type: audio/midi`
- `Content-Disposition: attachment; filename="song1.mid"`
- Body: MIDI 二进制

**404 响应**：`{"error": "文件不存在"}` 或 `{"error": "文件已过期"}`

---

## 4. 前端页面设计规范

### 4.1 核心交互流程

```
用户进入页面
  → GET /api/engines（填充引擎下拉框）
  → 用户拖拽/选择音频文件
  → 用户配置参数
  → 点击"开始转谱"
    → 单文件：POST /api/transcribe → 直接下载 MIDI
    → 多文件：POST /api/jobs → WS /ws/{id} → 实时进度 → 下载多个 MIDI
```

### 4.2 必需 UI 组件

| 组件 | 说明 |
|------|------|
| 文件上传区 | 拖拽 + 点击选择，支持多文件，显示文件列表，可删除 |
| 引擎选择 | 下拉框，数据来自 `/api/engines`，默认选中 `Ensemble (PT + BP)` |
| BPM 输入 | 数字输入框 + "自动检测"复选框 |
| 预处理选项 | "音量归一化"复选框（默认开）、"高频预加重"复选框（默认关） |
| 后处理参数 | 力度拉伸、置信度阈值、起音判定、持续断点 |
| 开始/终止按钮 | 开始转谱 + 强行终止 |
| 进度条 | 百分比 + 文字状态 |
| 日志面板 | 实时滚动日志，按类型着色（✅绿 ❌红 ⚠️黄 📋蓝） |
| 结果下载 | 每个完成的文件一个下载链接 |

### 4.3 参数默认值

| 参数 | 默认值 | 范围 |
|------|--------|------|
| engine | `Ensemble (PT + BP)` | 动态获取 |
| bpm | `120` | 30-400 |
| auto_bpm | `false` | — |
| normalize | `true` | — |
| preemphasis | `false` | — |
| velocity_stretch | `true` | — |
| confidence_threshold | `0.2` | 0-1 |
| bp_onset_threshold | `0.35` | 0.1-0.9 |
| bp_frame_threshold | `0.20` | 0.1-0.9 |

### 4.4 视觉风格参考

桌面端使用暗色主题，Web 端建议保持一致：
- 背景：`#0A0A0A`
- 面板：`#121212`
- 边框：`#222222`
- 强调色：`#E03E3E`（红色）
- 成功色：`#4ade80`
- 文字：`#CCCCCC` / `#888888`
- 日志区：`#050505`，等宽字体

### 4.5 注意事项

1. **神经网络引擎处理较慢**：PT 引擎单文件可能需要 30s-3min（取决于音频长度和是否有 GPU），前端应显示进度反馈，避免用户以为卡死
2. **同步接口超时**：`/api/transcribe` 长音频可能超过默认 HTTP 超时，前端应设置较长超时（建议 5 分钟）
3. **文件大小限制**：FastAPI 默认无限制，但 nginx/反向代理可能有 body size 限制
4. **WebSocket 重连**：网络不稳定时 WS 可能断开，前端应实现自动重连 + 回退到 GET 轮询
5. **结果文件临时性**：MIDI 文件存储在系统临时目录，服务器重启后失效，前端应在任务完成后立即提供下载

---

## 5. 完整调用示例（纯 HTML + JS）

```html
<!-- 单文件同步转谱 -->
<script>
async function transcribe(file) {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('engine', 'Ensemble (PT + BP)');

  const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
</script>

<!-- 多文件异步转谱 -->
<script>
async function batchTranscribe(files) {
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  fd.append('engine', 'Ensemble (PT + BP)');

  const res = await fetch('/api/jobs', { method: 'POST', body: fd });
  const { job_id } = await res.json();

  return new Promise((resolve) => {
    const ws = new WebSocket(`ws://${location.host}/ws/${job_id}`);
    const results = [];

    ws.onmessage = (e) => {
      const d = JSON.parse(e.data);
      console.log(`${d.progress}% - ${d.message}`);

      if (d.status === 'done') {
        ws.close();
        resolve(d.result_files);
      }
    };
  });
}
</script>
```

---

## 6. 错误处理

| 场景 | HTTP 状态码 | 处理方式 |
|------|-------------|----------|
| 引擎不可用 | 400 | 提示用户选择其他引擎 |
| 任务不存在 | 404 | 任务已过期或ID错误 |
| 文件已过期 | 404 | 提示重新转谱 |
| 网络中断 | — | WebSocket 断开，回退轮询 |
| 处理异常 | — | 任务 status=failed，message 含错误信息 |

---

## 7. FastAPI 自动文档

后端启动后访问 `http://localhost:8000/docs` 可查看 Swagger UI 交互式文档，支持在线测试所有接口。
