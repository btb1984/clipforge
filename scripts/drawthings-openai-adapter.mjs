#!/usr/bin/env node
/**
 * Draw Things → OpenAI 兼容适配器
 *
 * Draw Things 的 API Server 只提供 A1111 协议（POST /sdapi/v1/txt2img、/sdapi/v1/img2img、
 * GET /sdapi/v1/options），而 ClipForge 的生图链路走的是 OpenAI 协议
 * （POST /v1/images/generations、/v1/images/edits、GET /v1/models）。这个脚本在本地做协议翻译，
 * 让「设置 → 平台 Key → OpenAI / Compatible」直接驱动本机 Draw Things 出图。
 *
 * 用法：
 *   node scripts/drawthings-openai-adapter.mjs            # 启动适配器（默认 127.0.0.1:8787）
 *   node scripts/drawthings-openai-adapter.mjs --probe    # 探测 Draw Things 的 HTTP API 端口
 *
 * 然后在 ClipForge 里：
 *   平台 Key → OpenAI → 启用 → API Key 填 sk-local → 自定义端点 http://127.0.0.1:8787/v1
 *   生成设置 → 自定义模型 → 平台 OpenAI / Compatible、模型 ID 填 drawthings、类型图片
 *
 * 环境变量：
 *   PORT            适配器端口，默认 8787
 *   HOST            监听地址，默认 127.0.0.1（ClipForge 跑在 Docker 里时改成 0.0.0.0）
 *   DT_URL          Draw Things 地址，默认 http://127.0.0.1:7859（--probe 可确认实际端口）
 *   DT_STEPS        采样步数，默认 24；设为 0 则不发送（沿用 Draw Things 当前设置）
 *   DT_CFG          CFG scale，默认 7；设为 0 则不发送
 *   DT_SAMPLER      采样器名（如 "DPM++ 2M Karras"），默认不发送
 *   DT_NEGATIVE     默认负向提示词（ClipForge 的 OpenAI 通道不传负向词，只能在这里兜底）
 *   DT_MAX_SIDE     输出长边上限，默认 1024（0 = 不限制）。Apple Silicon 上超过 1024 容易超时
 *   DT_ASPECT       强制画面比例，如 9:16；留空则跟随 ClipForge 传入的 size
 *   DT_DENOISE      图生图重绘强度 0~1，默认 0.6
 *   DT_TIMEOUT      上游超时毫秒，默认 600000
 *   DT_INSECURE_TLS 设为 1 时接受自签证书（仅在 Draw Things 打开了 TLS 时需要）
 */

import { createServer } from "node:http";

const PORT = Number(process.env.PORT || 8787);
const HOST = process.env.HOST || "127.0.0.1";
const DT_URL = (process.env.DT_URL || "http://127.0.0.1:7859").replace(/\/+$/, "");
const STEPS = Number(process.env.DT_STEPS ?? 24);
const CFG = Number(process.env.DT_CFG ?? 7);
const SAMPLER = process.env.DT_SAMPLER || "";
const NEGATIVE = process.env.DT_NEGATIVE || "";
const MAX_SIDE = Number(process.env.DT_MAX_SIDE ?? 1024);
const ASPECT = process.env.DT_ASPECT || "";
const DENOISE = Number(process.env.DT_DENOISE ?? 0.6);
const DT_TIMEOUT = Number(process.env.DT_TIMEOUT ?? 600_000);
const MAX_BODY = 64 * 1024 * 1024;

if (process.env.DT_INSECURE_TLS === "1") process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";

const log = (...args) => console.log(`[drawthings] ${new Date().toLocaleTimeString("zh-CN")}`, ...args);

// ==================== 尺寸换算 ====================

/**
 * ClipForge 传进来的 size 对自定义模型是硬编码的三档（如 1024x1536），并不等于真实的 9:16。
 * 这里按比例还原：长边压到 MAX_SIDE，短边按比例算，两边都对齐到 64 的倍数（扩散模型的友好尺寸）。
 */
function fitSize(raw) {
  const match = /^(\d{2,5})\s*[x×]\s*(\d{2,5})$/i.exec(String(raw || "").trim());
  let w = match ? Number(match[1]) : 1024;
  let h = match ? Number(match[2]) : 1024;
  const snap = (value) => Math.max(256, Math.round(value / 64) * 64);

  let ratio;
  if (ASPECT) {
    const [aw, ah] = ASPECT.split(":").map(Number);
    if (aw > 0 && ah > 0) ratio = Math.min(aw, ah) / Math.max(aw, ah);
    // 强制比例时，用请求的长边作为输出长边
    const forceWide = aw > ah;
    const long = Math.max(MAX_SIDE > 0 ? Math.min(Math.max(w, h), MAX_SIDE) : Math.max(w, h), 512);
    const snappedLong = snap(long);
    const snappedShort = snap(snappedLong * ratio);
    return forceWide ? { width: snappedLong, height: snappedShort } : { width: snappedShort, height: snappedLong };
  }

  const wide = w >= h;
  ratio = Math.min(w, h) / Math.max(w, h);
  let long = Math.max(w, h);
  if (MAX_SIDE > 0) long = Math.min(long, MAX_SIDE);
  const snappedLong = snap(long);
  const snappedShort = snap(snappedLong * ratio);
  return wide ? { width: snappedLong, height: snappedShort } : { width: snappedShort, height: snappedLong };
}

// ==================== 上游调用 ====================

async function postDrawThings(path, payload) {
  let body = { ...payload };
  for (let attempt = 0; attempt < 2; attempt++) {
    let response;
    try {
      response = await fetch(`${DT_URL}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(DT_TIMEOUT),
      });
    } catch (error) {
      const reason = error?.name === "TimeoutError" ? `超时（${DT_TIMEOUT}ms）` : error?.message || String(error);
      throw new Error(`连不上 Draw Things（${DT_URL}）：${reason}。确认 App 已启动、HTTP API 服务器已开启、端口与 DT_URL 一致。`);
    }

    const text = await response.text();
    if (response.ok) {
      try {
        return JSON.parse(text);
      } catch {
        throw new Error(`Draw Things 返回了非 JSON 响应：${text.slice(0, 200)}`);
      }
    }

    // Draw Things 的 A1111 兼容层会拒收不认识的字段（"Unrecognized keys: xxx"）。
    // 老版本没有某些可选字段时，去掉后重试一次，而不是直接把失败抛给用户。
    const unrecognized = /Unrecognized keys?:\s*(.+)/i.exec(text);
    if (attempt === 0 && unrecognized) {
      // 上游把字段名夹在 JSON 里返回（"Unrecognized keys: denoising_strength"}），
      // 只取标识符本体，避免把引号括号当成字段名
      const keys = unrecognized[1].match(/[A-Za-z0-9_]+/g) ?? [];
      const removed = keys.filter((key) => key in body);
      for (const key of removed) delete body[key];
      if (removed.length > 0) {
        log(`上游不认识字段 ${removed.join(", ")}，已去掉后重试`);
        continue;
      }
    }
    throw new Error(`Draw Things ${response.status}：${text.slice(0, 400) || response.statusText}`);
  }
  throw new Error("Draw Things 请求失败");
}

async function currentModel() {
  try {
    const response = await fetch(`${DT_URL}/sdapi/v1/options`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) return null;
    const options = await response.json();
    return typeof options?.sd_model_checkpoint === "string" ? options.sd_model_checkpoint : null;
  } catch {
    return null;
  }
}

function buildPayload({ prompt, negative, width, height, count, initImages }) {
  const payload = {
    prompt: prompt || "",
    negative_prompt: negative || NEGATIVE,
    width,
    height,
    batch_size: count,
    seed: -1,
  };
  if (STEPS > 0) payload.steps = STEPS;
  if (CFG > 0) payload.cfg_scale = CFG;
  if (SAMPLER) payload.sampler_name = SAMPLER;
  if (initImages?.length) {
    payload.init_images = initImages;
    payload.denoising_strength = DENOISE;
  }
  return payload;
}

function toOpenAIResponse(json, model) {
  const images = Array.isArray(json?.images) ? json.images : [];
  const data = images
    .filter((image) => typeof image === "string" && image.length > 0)
    .map((image) => ({ b64_json: image.replace(/^data:image\/\w+;base64,/, "") }));
  if (data.length === 0) throw new Error("Draw Things 返回成功但没有图片数据，请检查 App 里的模型与画布设置");
  return { created: Math.floor(Date.now() / 1000), data, model };
}

// ==================== multipart 解析（生图编辑接口用） ====================

function parseMultipart(buffer, contentType) {
  const boundaryMatch = /boundary=(?:"([^"]+)"|([^;\s]+))/i.exec(contentType || "");
  if (!boundaryMatch) return [];
  const boundary = Buffer.from(`--${boundaryMatch[1] || boundaryMatch[2]}`);
  const parts = [];
  let cursor = buffer.indexOf(boundary);
  while (cursor !== -1) {
    const next = buffer.indexOf(boundary, cursor + boundary.length);
    if (next === -1) break;
    let chunk = buffer.subarray(cursor + boundary.length, next);
    if (chunk.subarray(0, 2).toString() === "\r\n") chunk = chunk.subarray(2);
    if (chunk.subarray(chunk.length - 2).toString() === "\r\n") chunk = chunk.subarray(0, chunk.length - 2);
    const separator = chunk.indexOf("\r\n\r\n");
    if (separator !== -1) {
      const head = chunk.subarray(0, separator).toString("utf8");
      parts.push({
        name: /name="([^"]*)"/i.exec(head)?.[1] ?? "",
        filename: /filename="([^"]*)"/i.exec(head)?.[1],
        type: /content-type:\s*([^\r\n]+)/i.exec(head)?.[1]?.trim(),
        body: chunk.subarray(separator + 4),
      });
    }
    cursor = next;
  }
  return parts;
}

// ==================== HTTP 路由 ====================

async function readBody(req) {
  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY) throw new Error("请求体过大（>64MB）");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

function send(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Content-Length": Buffer.byteLength(body) });
  res.end(body);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://127.0.0.1:${PORT}`);
  const path = url.pathname.replace(/\/+$/, "") || "/";

  try {
    if (req.method === "GET" && path === "/healthz") {
      const model = await currentModel();
      return send(res, 200, { ok: Boolean(model), upstream: DT_URL, model, adapter: "drawthings-openai" });
    }

    if (req.method === "GET" && (path === "/v1/models" || path === "/models")) {
      const model = await currentModel();
      const data = [model ?? "drawthings"].map((id) => ({ id, object: "model", owned_by: "drawthings" }));
      return send(res, 200, { object: "list", data });
    }

    if (req.method === "POST" && path === "/v1/images/generations") {
      const raw = await readBody(req);
      let body;
      try {
        body = JSON.parse(raw.toString("utf8") || "{}");
      } catch {
        return send(res, 400, { error: { message: "请求不是合法 JSON" } });
      }
      const { width, height } = fitSize(body.size);
      const count = Math.min(Math.max(Number(body.n) || 1, 1), 4);
      log(`文生图 ${width}x${height} × ${count}：${String(body.prompt || "").slice(0, 60)}`);
      const json = await postDrawThings("/sdapi/v1/txt2img", buildPayload({
        prompt: body.prompt,
        negative: body.negative_prompt,
        width,
        height,
        count,
      }));
      return send(res, 200, toOpenAIResponse(json, body.model || "drawthings"));
    }

    if (req.method === "POST" && path === "/v1/images/edits") {
      const raw = await readBody(req);
      const parts = parseMultipart(raw, req.headers["content-type"]);
      const field = (name) => parts.find((part) => part.name === name && !part.filename)?.body.toString("utf8");
      const images = parts
        .filter((part) => part.filename && part.body.length > 0 && (part.type || "").startsWith("image/"))
        .map((part) => part.body.toString("base64"));
      if (images.length === 0) {
        return send(res, 400, { error: { message: "图生图缺少参考图（multipart 中未找到 image 字段）" } });
      }
      const { width, height } = fitSize(field("size"));
      const count = Math.min(Math.max(Number(field("n")) || 1, 1), 4);
      log(`图生图 ${width}x${height} × ${count}，参考图 ${images.length} 张`);
      const json = await postDrawThings("/sdapi/v1/img2img", buildPayload({
        prompt: field("prompt"),
        negative: field("negative_prompt"),
        width,
        height,
        count,
        initImages: images,
      }));
      return send(res, 200, toOpenAIResponse(json, field("model") || "drawthings"));
    }

    return send(res, 404, { error: { message: `未实现的端点：${req.method} ${path}` } });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    log(`失败：${message}`);
    return send(res, 502, { error: { message } });
  }
});

// ==================== 启动 / 探测 ====================

async function probe() {
  const candidates = [
    process.env.DT_URL,
    "http://127.0.0.1:7859",
    "http://127.0.0.1:7860",
    "http://127.0.0.1:7861",
    "http://127.0.0.1:8000",
  ].filter(Boolean);
  const seen = new Set();
  for (const base of candidates) {
    const url = base.replace(/\/+$/, "");
    if (seen.has(url)) continue;
    seen.add(url);
    try {
      const response = await fetch(`${url}/sdapi/v1/options`, { signal: AbortSignal.timeout(1500) });
      if (!response.ok) {
        console.log(`✗ ${url} → HTTP ${response.status}`);
        continue;
      }
      const options = await response.json().catch(() => ({}));
      console.log(`✓ ${url} → 可用，当前模型：${options?.sd_model_checkpoint ?? "未知"}`);
      console.log(`  把它写进环境变量：DT_URL=${url}`);
      return;
    } catch (error) {
      console.log(`✗ ${url} → ${error?.name === "TimeoutError" ? "超时" : "连接失败"}`);
    }
  }
  console.log("\n没有发现可用的 Draw Things HTTP API。请检查：");
  console.log("  Draw Things → 设置 → 高级 → HTTP API 服务器：开启，协议选 HTTP，关掉 TLS");
  console.log("  然后再看一次端口号，用 DT_URL=http://127.0.0.1:<端口> 启动适配器");
}

if (process.argv.includes("--probe")) {
  await probe();
  process.exit(0);
}

server.listen(PORT, HOST, async () => {
  const model = await currentModel();
  log(`适配器已启动：http://${HOST}:${PORT}/v1`);
  log(model ? `已连上 Draw Things（${DT_URL}），当前模型：${model}` : `暂时连不上 Draw Things（${DT_URL}），生成时会再试`);
  log(`参数：steps=${STEPS || "默认"} cfg=${CFG || "默认"} 长边上限=${MAX_SIDE || "不限"} 比例=${ASPECT || "跟随请求"}`);
});
