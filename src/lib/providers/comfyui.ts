/**
 * ComfyUI Provider — 本地（或局域网）ComfyUI 的 Wan2.2-TI2V-5B 接入
 *
 * 和其余 7 家平台的根本区别：ComfyUI 没有 API Key、没有计费、没有公网任务队列，
 * 它只是一个跑在自己机器上的 HTTP 服务。所以 provider 的职责不是「调一个生成接口」，
 * 而是把 ClipForge 的统一参数翻译成一张 API 格式的 workflow 图，再用 5 个端点驱动它：
 *
 *   POST /upload/image   首帧落到 ComfyUI 的 input 目录（LoadImage 只认本地文件名）
 *   POST /prompt         提交图，立刻返回 prompt_id —— 这个 id 就是我们的 taskId
 *   GET  /history/{id}   完成后才有记录（含产物文件名 + 真实执行耗时）
 *   GET  /queue          未完成时的唯一线索（在排队 / 在执行）
 *   GET  /view           取产物二进制
 *
 * 三个特性决定了下面的实现，也是这个 provider 与「照抄一个云端 provider」的区别：
 *
 * 1. 单任务以十分钟计（本机 M5 实测 480×832 / 121 帧 / 12 步 = 434s），而基类
 *    pollTaskStatus 默认 5s × 200 ≈ 16.7 分钟 —— 一条 8 秒镜头就会超时白跑。
 *    这里把 waitForTask 按「总时长 ≥ 60 分钟」重算 maxAttempts。
 * 2. /history 为空有两种完全不同的含义：还在排队，或者 ComfyUI 重启把队列丢了。
 *    单独看 history 会误判，必须和 /queue 交叉验证，并且连续多次查不到才判失败
 *    （见 getTaskStatus 里的 lostChecks 计数）。
 * 3. 产物就在本机磁盘上。完成后立刻用 downloadProviderVideo 落成 /api/files 地址，
 *    而不是把 ComfyUI 的 URL 交回上层：ComfyUI 一关视频就播不了，而且上层
 *    asset-persistence 会为它多走一遍 120s 的公网下载。
 */

import { readFile } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { BaseProvider, ProviderError } from './base'
import { downloadProviderVideo } from './video-download'
import type {
  ProviderConfig,
  ImageOptions,
  ImageResult,
  VideoOptions,
  VideoResult,
  TaskStatus,
  Model,
  MediaType,
} from './types'

// ==================== 常量 ====================

// 无 baseUrl 时的默认值：ComfyUI 本地服务
const DEFAULT_BASE_URL = 'http://127.0.0.1:8188'

// Wan2.2 的原生帧率。实测 121 帧 @24fps = 5.04s，成片时长由帧数决定，
// 所以这里不让上层 fps 改写它 —— 否则 duration 会随 fps 漂移。
const FPS = 24

/**
 * 模型档位。这里的「模型」是「同一份 Wan2.2 权重的不同推理配置」，不是不同权重文件 ——
 * /object_info 里 UNETLoader 只列得出一个 safetensors。
 *
 * 做成多个可选 id 有两个实际原因：
 * 1. /api/ai/video 是同步长连接，用户需要能在「先看个大概」和「正式出片」之间选，
 *    而不是每次改环境变量重启服务；
 * 2. 让设置页「自定义模型接入点」对 ComfyUI 真正有意义：填 `wan2.2-ti2v-5b@8`
 *    就是一个能跑的 8 步档位，而不是一个「名字不同、结果相同」的空别名。
 */
interface VideoPreset {
  steps: number
  pixelBudget: number
  name: string
  description: string
}

const PRESET_MODEL_ID = 'wan2.2-ti2v-5b'

// 像素预算把任意请求尺寸压到 5B 的训练分布附近：720×1280 与 1080×1920 都落到 480×832。
// 实测 960×960 要 38 分钟，尺寸不压下来单镜就超出整片时间预算。
const VIDEO_PRESETS: Record<string, VideoPreset> = {
  [PRESET_MODEL_ID]: {
    steps: 12,
    pixelBudget: 400_000,
    name: 'Wan2.2 TI2V 5B · 标准 (12 步)',
    description: '480×832 · 12 步 · 本机 M5 实测 5 秒镜头 434 秒',
  },
  'wan2.2-ti2v-5b-fast': {
    steps: 8,
    pixelBudget: 250_000, // 9:16 → 384×672（32 的倍数）
    name: 'Wan2.2 TI2V 5B · 快速预览 (8 步)',
    description: '384×672 · 8 步 · 本机 M5 实测 5 秒镜头 262 秒，先看构图与运动',
  },
  'wan2.2-ti2v-5b-hq': {
    steps: 20,
    pixelBudget: 400_000,
    name: 'Wan2.2 TI2V 5B · 精修 (20 步)',
    description: '480×832 · 20 步 · 按步数外推约 12 分钟，细节更稳',
  },
}

// ModelSamplingSD3 的 shift。Wan2.2 官方示例是 8.0；只有用户显式传了
// motionStrength 才按 4~12 映射，未传时严格保持官方值（见 motionShift）。
const DEFAULT_SHIFT = 8.0

// Wan22ImageToVideoLatent 的 width/height step 是 32（不是常见的 16），length step 是 4，
// 且 length 必须是 4k+1。写错这两个数字，ComfyUI 会直接回 node_errors 而不排队。
const SIZE_STEP = 32
const LENGTH_STEP = 4

// 单镜时长上下限，与前端 assets 页的 clamp(4, 15) 保持一致
const MIN_DURATION = 4
const MAX_DURATION = 15

// 轮询总时长下限：一条 6 镜的片子是串行执行的，最后一镜要等前面的全部跑完
const MIN_WAIT_MS = 60 * 60 * 1000

// 默认负向词（与 Wan2.2 官方示例一致）：抑制过曝、静态感、字幕水印与插画风
const DEFAULT_NEGATIVE = [
  'vivid tones, overexposed, static still image, blurry details, subtitles, text, watermark, logo,',
  'stylized, illustration, painting, cartoon, overall gray, worst quality, low quality,',
  'JPEG compression artifacts, deformed, extra limbs, bad anatomy',
].join(' ')

// API 图里的固定节点 ID。自己构造图（而不是去读用户的 .json 工作流）是关键决策：
// 用户在工作流编辑器里挪一下节点，节点 ID 就会变，读文件的做法必然在某次升级后静默错位。
const NODE = {
  unet: '1',
  clip: '2',
  vae: '3',
  shift: '4',
  positive: '5',
  negative: '6',
  latent: '7',
  sampler: '8',
  decode: '9',
  video: '10',
  save: '11',
  loadImage: '12',
} as const

/** 模块级「查不到任务」计数：跨多次轮询累积，用来区分「刚提交」和「队列丢了」 */
const lostChecks = new Map<string, number>()

// ==================== 图结构 ====================

type ApiNode = { class_type: string; inputs: Record<string, unknown> }
type ApiGraph = Record<string, ApiNode>

/** /history 里 SaveVideo 的产物描述 */
interface ComfyOutputFile {
  filename: string
  subfolder?: string
  type?: string
  metadata?: { duration?: number; fps?: number; frame_count?: number; width?: number; height?: number }
}

interface ComfyHistoryRecord {
  outputs?: Record<string, { images?: ComfyOutputFile[]; gifs?: ComfyOutputFile[]; videos?: ComfyOutputFile[] }>
  status?: {
    status_str?: string
    completed?: boolean
    messages?: Array<[string, Record<string, unknown>]>
  }
}

interface ComfyQueueResponse {
  queue_running?: Array<unknown[]>
  queue_pending?: Array<unknown[]>
}

interface ComfyPromptResponse {
  prompt_id?: string
  node_errors?: Record<string, { errors?: Array<{ message?: string; details?: string }> }>
  error?: { message?: string; details?: string; type?: string }
}

// ==================== Provider ====================

export class ComfyUIProvider extends BaseProvider {
  readonly name = 'comfyui'
  readonly displayName = 'ComfyUI (本地)'

  // 模型文件名允许用环境变量覆盖：同一份代码要能跑在只装了 fp8 版权重、
  // 或者换了 VAE 的机器上，否则第一次换模型就得改源码。
  private readonly unetName = process.env.COMFYUI_UNET || 'wan2.2_ti2v_5B_fp16.safetensors'
  private readonly clipName = process.env.COMFYUI_CLIP || 'umt5_xxl_fp8_e4m3fn_scaled.safetensors'
  private readonly vaeName = process.env.COMFYUI_VAE || 'wan2.2_vae.safetensors'
  // 环境变量是「全站强制覆盖」，优先级高于所选档位；留空则完全由 UI 里选的档位决定。
  // 这样既能给整台机器定死一个档，也能在项目里逐次切档，两者不打架。
  private readonly forcedSteps = Number(process.env.COMFYUI_VIDEO_STEPS) || undefined
  private readonly forcedPixelBudget = Number(process.env.COMFYUI_PIXEL_BUDGET) || undefined

  constructor(config: ProviderConfig) {
    super({
      ...config,
      // 结尾的斜杠必须去掉：基类用字符串拼接 `${baseUrl}${path}`，
      // "http://127.0.0.1:8188/" + "/prompt" 会变成双斜杠，某些反代直接 404
      baseUrl: (config.baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, ''),
      // 提交图是 POST 且无幂等性；冷启动时 ComfyUI 可能 20s 才回第一个字节，
      // 默认 30s 偶尔不够，这里放宽到 90s（仍低于路由层的容忍度）
      timeout: config.timeout ?? 90_000,
    })
  }

  /** ComfyUI 无鉴权：显式返回空对象，避免把 apiKey 当 Bearer 发出去被反代拒绝 */
  protected getAuthHeaders(): Record<string, string> {
    return {}
  }

  // ==================== 图像 ====================

  /**
   * 图像生成未实现。
   *
   * 这不是偷懒：ClipForge 的生图走的是 SDXL/FLUX 一类 checkpoint，而本机 ComfyUI
   * 只装了 Wan2.2 的视频权重（/object_info 里 CheckpointLoaderSimple 的候选是空数组）。
   * 硬写一条「假设你有 checkpoint」的路径，只会在用户点生成时抛一个看不懂的节点错误；
   * 明确说清缺什么、怎么补，比一个看起来支持但必然失败的实现更有用。
   */
  async generateImage(options: ImageOptions): Promise<ImageResult> {
    throw new ProviderError(
      `ComfyUI 暂无可用图像模型：本机只装了视频权重（${this.unetName}），` +
        `请求的 ${options.modelId || '未指定模型'} 无法在本地出图。` +
        `请把 SDXL/FLUX 检查点放进 ComfyUI/models/checkpoints，或在「生图模型」里改用其他平台`,
      'NO_IMAGE_MODEL',
      this.name
    )
  }

  // ==================== 视频（两阶段） ====================

  /** 阶段一：上传首帧 + 提交图，返回 prompt_id 作为 taskId */
  async submitVideoTask(options: VideoOptions): Promise<{ taskId: string; modelId: string }> {
    try {
      return await this.submitToComfy(options)
    } catch (error) {
      // ComfyUI 没启动时，fetch 抛的是一行 "fetch failed"，用户无从判断是模型问题还是服务没开
      throw this.explainConnectivity(error)
    }
  }

  private async submitToComfy(options: VideoOptions): Promise<{ taskId: string; modelId: string }> {
    const prompt = (options.prompt ?? '').trim()
    if (!prompt) {
      // 空 prompt 在 ComfyUI 里不会报错，只会安静地出一段无意义的画面 —— 提前拦住
      throw new ProviderError('缺少画面提示词，无法提交视频任务', 'NO_PROMPT', this.name)
    }

    // 档位决定步数与像素预算，motionStrength 决定采样偏移。三者在提交前一次性定死，
    // 之后 7 分钟的轮询期间不再读任何可变的 store 状态。
    const preset = this.resolvePreset(options.modelId)
    const steps = this.forcedSteps ?? preset.steps
    const pixelBudget = this.forcedPixelBudget ?? preset.pixelBudget
    const shift = this.motionShift(options.motionStrength)

    const { width, height } = this.fitSize(options.width, options.height, pixelBudget)
    const length = this.fitLength(options.duration)
    const seed = options.seed ?? Math.floor(Math.random() * 2 ** 47)

    // 首帧：/api/ai/video 已经把本地 /api/files 图转成了 data URI，但产品图等场景
    // 仍可能是 http 地址。两种都要先落到 ComfyUI 的 input 目录，LoadImage 只认文件名。
    const imageName = options.firstFrameUrl ? await this.uploadImage(options.firstFrameUrl) : undefined

    const graph = this.buildVideoGraph({
      width,
      height,
      length,
      seed,
      steps,
      shift,
      positive: this.clampPrompt(prompt),
      negative: options.negativePrompt?.trim() || DEFAULT_NEGATIVE,
      imageName,
    })

    // 长任务排错时最需要知道「当时到底提交的是什么参数」；prompt_id 会同时出现在
    // 任务中心与 ComfyUI 队列里，用它可以把两边对上。
    console.info(
      `[comfyui] model=${options.modelId} preset=${preset.name} ` +
        `${width}x${height} frames=${length} steps=${steps} shift=${shift} i2v=${Boolean(imageName)}`
    )

    const result = await this.request<ComfyPromptResponse>('/prompt', {
      method: 'POST',
      body: { prompt: graph, client_id: `clipforge-${seed}` },
      // 创建任务的 POST 不允许自动重试：重复提交会让同一镜头在队列里跑两遍
      idempotent: false,
    })

    // 节点参数错（尺寸不整除、模型文件不存在）在 ComfyUI 里是 HTTP 200 + node_errors，
    // 不解析就会拿到一个空的 prompt_id 然后空轮询到超时
    const nodeError = this.firstNodeError(result)
    if (nodeError) throw new ProviderError(`ComfyUI 拒绝了工作流：${nodeError}`, 'INVALID_GRAPH', this.name)
    if (!result.prompt_id) {
      throw new ProviderError(
        `ComfyUI 未返回任务 ID：${result.error?.message ?? '响应缺少 prompt_id'}`,
        'NO_PROMPT_ID',
        this.name
      )
    }

    lostChecks.delete(result.prompt_id)
    return { taskId: result.prompt_id, modelId: options.modelId }
  }

  /**
   * 把底层网络异常翻译成用户能行动的提示。
   *
   * 这里只处理「根本没连上」这一类（fetch failed / TypeError / 超时）—— 它们是本地服务
   * 最常见的故障，且原始报错完全指向不到「ComfyUI 没开」。业务错误（NO_PROMPT、
   * INVALID_GRAPH）原样抛出，否则真正的参数问题会被这条提示盖掉。
   */
  private explainConnectivity(error: unknown): unknown {
    const isNetwork =
      (error instanceof ProviderError && (error.code === 'NETWORK_ERROR' || error.code === 'TIMEOUT')) ||
      (error instanceof TypeError && /fetch/i.test(error.message))
    if (!isNetwork) return error
    return new ProviderError(
      `连不上 ComfyUI（${this.config.baseUrl}）：请确认 ComfyUI 已启动；` +
        `若是局域网/远程机器，检查设置里的接入地址与端口是否正确`,
      'COMFYUI_UNREACHABLE',
      this.name
    )
  }

  /** 阶段二：等待。超时窗口按「一镜可能排在 5 镜后面」来定，而不是按单镜耗时 */
  async waitForTask(
    taskId: string,
    options: { interval?: number; maxAttempts?: number } = {}
  ): Promise<TaskStatus> {
    const interval = options.interval ?? 10_000
    // 路由层固定传 interval:5000，如果沿用基类默认的 200 次就只有 16 分钟；
    // 这里按时间窗口反算次数，调用方显式给了 maxAttempts 才以它为准
    const maxAttempts = options.maxAttempts ?? Math.max(360, Math.ceil(MIN_WAIT_MS / interval))
    return this.pollTaskStatus(taskId, { interval, maxAttempts })
  }

  /** 非两阶段路径的兜底（submit + wait），保证 generateVideo 单独调用时也能用 */
  async generateVideo(options: VideoOptions): Promise<VideoResult> {
    const { taskId } = await this.submitVideoTask(options)
    const status = await this.waitForTask(taskId, { interval: 10_000 })
    return this.requireResult(status.result as VideoResult | undefined) as VideoResult
  }

  // ==================== 状态查询 ====================

  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const history = await this.request<Record<string, ComfyHistoryRecord>>(
      `/history/${encodeURIComponent(taskId)}`
    )
    const record = history?.[taskId]

    if (record) {
      lostChecks.delete(taskId)

      const statusStr = record.status?.status_str
      if (statusStr === 'error') {
        return { taskId, status: 'failed', error: this.executionError(record), errorCode: 'GENERATION_FAILED' }
      }

      const file = this.findOutputFile(record)
      if (file) {
        // 完成后立刻本地化：返回 /api/files 地址，上层 asset-persistence 会直接接受，
        // 不再触发一次公网下载；ComfyUI 之后被关掉也不影响已生成的素材
        const viewUrl = this.viewUrl(file)
        const localUrl = await downloadProviderVideo(viewUrl, {})
        const meta = file.metadata
        return {
          taskId,
          status: 'completed',
          progress: 100,
          result: {
            taskId,
            videoUrls: [localUrl],
            modelId: '',
            // metadata.duration 已经带了精确时长；缺失时用 帧数/帧率 反算（121/24 = 5.04s）
            duration:
              meta?.duration ??
              (meta?.frame_count && meta?.fps ? meta.frame_count / meta.fps : undefined),
            extra: { comfyuiFile: file.filename, width: meta?.width, height: meta?.height, fps: meta?.fps },
          },
        }
      }

      // 执行成功但没找到产物：SaveVideo 被删掉或改成了 PreviewImage，
      // 继续等下去只会耗到轮询超时，直接给出可执行的错误
      if (record.status?.completed) {
        return {
          taskId,
          status: 'failed',
          error: 'ComfyUI 执行完成但没有输出视频文件（检查工作流是否保留了 SaveVideo 节点）',
          errorCode: 'NO_OUTPUT',
        }
      }
      return { taskId, status: 'processing' }
    }

    // history 为空：在排队、在执行，或者队列已丢失。必须用 /queue 交叉判断。
    const queue = await this.request<ComfyQueueResponse>('/queue')
    const inQueue = this.queueContains(queue, taskId)
    if (inQueue) {
      lostChecks.delete(taskId)
      return { taskId, status: inQueue === 'running' ? 'processing' : 'pending' }
    }

    // 两边都查不到。刚提交的瞬间 history 和 queue 都可能还没有记录，
    // 所以先容忍几轮，连续 3 次（默认间隔 10s ≈ 30s）才判定队列丢失。
    const misses = (lostChecks.get(taskId) ?? 0) + 1
    lostChecks.set(taskId, misses)
    if (misses >= 3) {
      lostChecks.delete(taskId)
      return {
        taskId,
        status: 'failed',
        error: 'ComfyUI 队列中已找不到该任务（服务可能重启过），请重新生成',
        errorCode: 'TASK_LOST',
      }
    }
    return { taskId, status: 'pending', extra: { missedChecks: misses } }
  }

  // ==================== 模型清单 ====================

  /**
   * 静态返回，不查 /object_info。
   *
   * /api/ai/models 给 listModels 的超时只有 12 秒，而 ComfyUI 冷启动扫描模型目录
   * （几十 GB 的 safetensors）经常超过这个时间。超时的直接后果是设置页下拉框空白，
   * 用户会以为「没有可用模型」而不是「还在加载」。
   */
  async listModels(mediaType?: MediaType): Promise<Model[]> {
    // 标准档放第一位：设置页在「未选过默认模型」时会自动取 options[0]
    // （page.tsx:371），把 12 步放在首位才不会让新手一上来就选到精修档。
    const order = [PRESET_MODEL_ID, 'wan2.2-ti2v-5b-fast', 'wan2.2-ti2v-5b-hq']
    const models: Model[] = order.map((id) => {
      const preset = VIDEO_PRESETS[id]
      return {
        id,
        name: preset.name,
        description: `${preset.description}；${FPS}fps，无原生音频`,
        modes: ['image-to-video', 'text-to-video'],
        mediaType: 'video',
        provider: this.name,
        supportsAudio: false,
      }
    })
    // 本机没有图像 checkpoint，这里如实返回空数组 —— 用户在「生图模型」里就看不到
    // ComfyUI，而不是选了之后在生成时收到一个 NO_IMAGE_MODEL 错误
    return mediaType ? models.filter((m) => m.mediaType === mediaType) : models
  }

  // ==================== 私有：图构造 ====================

  private buildVideoGraph(params: {
    width: number
    height: number
    length: number
    seed: number
    steps: number
    shift: number
    positive: string
    negative: string
    imageName?: string
  }): ApiGraph {
    const { width, height, length, seed, steps, shift, positive, negative, imageName } = params

    const graph: ApiGraph = {
      [NODE.unet]: {
        class_type: 'UNETLoader',
        inputs: { unet_name: this.unetName, weight_dtype: 'default' },
      },
      [NODE.clip]: {
        class_type: 'CLIPLoader',
        inputs: { clip_name: this.clipName, type: 'wan', device: 'default' },
      },
      [NODE.vae]: {
        class_type: 'VAELoader',
        inputs: { vae_name: this.vaeName },
      },
      // shift 改变噪声调度：值越大，采样轨迹越倾向于保留高频细节与更明显的帧间变化。
      // 未显式设置 motionStrength 时保持官方 8.0。
      [NODE.shift]: {
        class_type: 'ModelSamplingSD3',
        inputs: { shift, model: [NODE.unet, 0] },
      },
      [NODE.positive]: {
        class_type: 'CLIPTextEncode',
        inputs: { text: positive, clip: [NODE.clip, 0] },
      },
      [NODE.negative]: {
        class_type: 'CLIPTextEncode',
        inputs: { text: negative, clip: [NODE.clip, 0] },
      },
      [NODE.latent]: {
        class_type: 'Wan22ImageToVideoLatent',
        inputs: {
          width,
          height,
          length,
          batch_size: 1,
          vae: [NODE.vae, 0],
          // start_image 是可选输入：连上就是图生视频，不连就是纯文本生视频
          ...(imageName && { start_image: [NODE.loadImage, 0] }),
        },
      },
      [NODE.sampler]: {
        class_type: 'KSampler',
        inputs: {
          seed,
          steps,
          cfg: 5.0,
          sampler_name: 'euler',
          scheduler: 'normal',
          denoise: 1.0,
          model: [NODE.shift, 0],
          positive: [NODE.positive, 0],
          negative: [NODE.negative, 0],
          latent_image: [NODE.latent, 0],
        },
      },
      [NODE.decode]: {
        class_type: 'VAEDecode',
        inputs: { samples: [NODE.sampler, 0], vae: [NODE.vae, 0] },
      },
      [NODE.video]: {
        class_type: 'CreateVideo',
        inputs: {
          fps: FPS,
          bit_depth: 'auto',
          color_space: 'sRGB',
          codec: 'none',
          images: [NODE.decode, 0],
        },
      },
      [NODE.save]: {
        class_type: 'SaveVideo',
        inputs: {
          filename_prefix: `clipforge/${Date.now()}`,
          format: 'auto',
          'format.codec': 'auto',
          codec: 'auto',
          'video-preview': '',
          video: [NODE.video, 0],
        },
      },
    }

    if (imageName) {
      graph[NODE.loadImage] = { class_type: 'LoadImage', inputs: { image: imageName } }
    }
    return graph
  }

  // ==================== 私有：首帧上传 ====================

  /**
   * 把 data URI / 本地 /api/files 路径 / 远程 URL 统一变成 ComfyUI input 目录里的文件名。
   *
   * 文件名用内容哈希而不是随机名：同一个镜头重试时（超时重提、改提示词重跑）首帧往往不变，
   * 哈希让 ComfyUI 复用同一份文件，也顺带避免了 LoadImage 因文件名变化而重复读盘。
   */
  private async uploadImage(ref: string): Promise<string> {
    const { buffer, mime } = await this.readImageBytes(ref)
    const digest = createHash('sha256').update(buffer).digest('hex').slice(0, 16)
    const ext = mime.includes('jpeg') ? 'jpg' : mime.includes('webp') ? 'webp' : 'png'
    const filename = `cf-${digest}.${ext}`

    const form = new FormData()
    form.append('image', new Blob([new Uint8Array(buffer)], { type: mime }), filename)
    form.append('overwrite', 'true')
    form.append('type', 'input')

    const response = await fetch(`${this.config.baseUrl}/upload/image`, {
      method: 'POST',
      body: form,
      signal: AbortSignal.timeout(120_000),
    })
    if (!response.ok) {
      throw new ProviderError(
        `首帧上传失败（${response.status}）：检查 ComfyUI 是否在 ${this.config.baseUrl} 运行`,
        'UPLOAD_FAILED',
        this.name,
        response.status
      )
    }

    const uploaded = (await response.json()) as { name?: string; subfolder?: string }
    if (!uploaded.name) throw new ProviderError('首帧上传后未返回文件名', 'UPLOAD_NO_NAME', this.name)
    // 传入子目录时 LoadImage 需要 "subfolder/name"，只给 name 会找不到文件
    return uploaded.subfolder ? `${uploaded.subfolder}/${uploaded.name}` : uploaded.name
  }

  private async readImageBytes(ref: string): Promise<{ buffer: Buffer; mime: string }> {
    if (ref.startsWith('data:')) {
      const comma = ref.indexOf(',')
      if (comma === -1) throw new ProviderError('首帧 data URI 格式不合法', 'INVALID_IMAGE', this.name)
      const meta = ref.slice(5, comma)
      const payload = ref.slice(comma + 1)
      const mime = meta.split(';')[0] || 'image/png'
      const buffer = /;base64/i.test(meta)
        ? Buffer.from(payload, 'base64')
        : Buffer.from(decodeURIComponent(payload), 'utf8')
      if (!buffer.byteLength) throw new ProviderError('首帧数据为空', 'INVALID_IMAGE', this.name)
      return { buffer, mime }
    }

    if (ref.startsWith('/api/files/')) {
      // 动态 import：remote-image 依赖 @/lib/paths（会读 process.cwd），
      // 放在函数内可以避免模块加载顺序影响静态分析，也保持 provider 层对上层路径工具的弱依赖
      const { resolveUploadFilePath } = await import('@/lib/remote-image')
      const local = resolveUploadFilePath(ref)
      if (!local) throw new ProviderError('首帧路径不合法（疑似越过 uploads 目录）', 'INVALID_IMAGE', this.name)
      try {
        return { buffer: await readFile(local), mime: 'image/png' }
      } catch {
        throw new ProviderError('首帧文件不存在，请重新生成或上传', 'IMAGE_NOT_FOUND', this.name)
      }
    }

    if (/^https?:\/\//i.test(ref)) {
      // ComfyUI 的 LoadImage 只能读本地文件，远程图必须先下载再上传
      const response = await fetch(ref, { signal: AbortSignal.timeout(60_000) })
      if (!response.ok) {
        throw new ProviderError(`首帧下载失败（${response.status}）`, 'IMAGE_DOWNLOAD_FAILED', this.name, response.status)
      }
      return {
        buffer: Buffer.from(await response.arrayBuffer()),
        mime: response.headers.get('content-type')?.split(';')[0] || 'image/png',
      }
    }

    throw new ProviderError('不支持的首帧地址格式', 'INVALID_IMAGE', this.name)
  }

  // ==================== 私有：参数换算 ====================

  /**
   * 把 modelId 解析成具体的推理档位。
   *
   * 内置三个 id 之外还支持 `wan2.2-ti2v-5b@N`（N = 4–30 步）：这是设置页
   * 「自定义模型接入点」对 ComfyUI 唯一有意义的写法 —— 填 `@6` 会真的跑 6 步，
   * 而不是只给同一张图换个显示名。
   *
   * 其它任意字符串回退到标准档：用户可能在「自定义模型」里留着旧别名或误填的值，
   * 直接报错会让整条片子卡住，静默回退则至少能出片（日志里会打印实际档位）。
   */
  private resolvePreset(modelId: string | undefined): VideoPreset {
    const id = (modelId ?? '').trim()
    const builtin = VIDEO_PRESETS[id]
    if (builtin) return builtin

    const custom = /^wan2\.2-ti2v-5b@(\d{1,2})$/i.exec(id)
    if (custom) {
      const steps = Math.min(30, Math.max(4, Number(custom[1])))
      const standard = VIDEO_PRESETS[PRESET_MODEL_ID]
      return {
        steps,
        pixelBudget: standard.pixelBudget,
        name: `自定义 ${steps} 步`,
        description: `${steps} 步 · 标准像素预算`,
      }
    }
    return VIDEO_PRESETS[PRESET_MODEL_ID]
  }

  /**
   * motionStrength(0–1) → ModelSamplingSD3.shift(4–12)。
   *
   * 如实说明局限：shift 改的是噪声调度，不是物理意义上的「运动幅度」。
   * 调高会让帧间变化更活跃、细节更锐，但不会出现「镜头推得更快」。真正决定运镜的
   * 仍是 prompt（素材页的「运镜 轻/中/强」）。未传时返回官方 8.0，避免为了接一个
   * UI 旋钮而把已经跑通的默认画质改掉。
   *
   * 生产档位会传 0.35 / 0.55 / 0.72，对应 shift 6.8 / 8.4 / 9.8 —— 这正好让
   * 「极速试片 / 智能均衡 / 品牌大片」在本地推理上也有了实际差别。
   */
  private motionShift(motionStrength?: number): number {
    if (motionStrength == null || !Number.isFinite(motionStrength)) return DEFAULT_SHIFT
    const strength = Math.max(0, Math.min(1, motionStrength))
    return Math.round((4 + strength * 8) * 10) / 10
  }

  /**
   * 裁掉超出文本编码器容量的尾部。
   *
   * umt5_xxl 的上限是 512 token，超出部分会被 CLIPTextEncode 静默截断 —— 不报错，
   * 只是画面莫名丢失运动描述。ClipForge 拼的 prompt 往往上千字符（运动描述 +
   * Project direction + 控制计划后缀），一定会撞上这个上限。
   *
   * 保留「前 70% + 后 25%」而不是直接切尾：开头是主体与构图，结尾通常是
   * 项目级的保真/风格约束（buildMotionPrompt 把 projectDirection 追加在最后），
   * 两头都比中间重要。头部退到最近的句子边界，避免半句话被切断。
   */
  private clampPrompt(text: string, maxChars = 900): string {
    if (text.length <= maxChars) return text
    const headLimit = Math.floor(maxChars * 0.7)
    const head = text.slice(0, headLimit)
    const boundary = Math.max(head.lastIndexOf('. '), head.lastIndexOf('; '), head.lastIndexOf(', '))
    // 找不到像样的断点（比如整段没有标点）时才硬切，否则至少保留 40% 才值得回退
    const headClean = boundary > maxChars * 0.4 ? head.slice(0, boundary + 1) : head
    const tail = text.slice(-Math.floor(maxChars * 0.25))
    return `${headClean.trim()} ... ${tail.trim()}`
  }

  /**
   * 把请求尺寸压到像素预算内，并对齐到 32 的倍数。
   *
   * 按「面积」而不是「长边」缩放，是为了让 16:9 和 9:16 消耗同样的算力：
   * 只缩长边的话，竖屏 720×1280 会比横屏 1280×720 慢一倍。
   */
  private fitSize(width?: number, height?: number, pixelBudget?: number): { width: number; height: number } {
    const w = width && width > 0 ? width : 720
    const h = height && height > 0 ? height : 1280
    const budget = pixelBudget && pixelBudget > 0 ? pixelBudget : VIDEO_PRESETS[PRESET_MODEL_ID].pixelBudget
    const area = w * h
    const scale = Math.min(1, Math.sqrt(budget / area))
    const align = (value: number) =>
      Math.max(SIZE_STEP, Math.round((value * scale) / SIZE_STEP) * SIZE_STEP)
    return { width: align(w), height: align(h) }
  }

  /** 秒 → 帧数：Wan 的时间维压缩是 4，length 必须是 4k+1（121 帧 = 5.04s） */
  private fitLength(duration?: number): number {
    const seconds = Math.min(MAX_DURATION, Math.max(MIN_DURATION, Math.round(duration ?? 5)))
    return Math.round((seconds * FPS) / LENGTH_STEP) * LENGTH_STEP + 1
  }

  // ==================== 私有：响应解析 ====================

  private firstNodeError(result: ComfyPromptResponse): string | undefined {
    if (result.error?.message) return result.error.message
    for (const node of Object.values(result.node_errors ?? {})) {
      const first = node.errors?.[0]
      if (first?.message) return first.details ? `${first.message}（${first.details}）` : first.message
    }
    return undefined
  }

  private executionError(record: ComfyHistoryRecord): string {
    for (const message of record.status?.messages ?? []) {
      if (Array.isArray(message) && message[0] === 'execution_error') {
        const text = message[1]?.exception_message
        if (typeof text === 'string' && text.trim()) return text
      }
    }
    return 'ComfyUI 执行失败（未返回具体原因）'
  }

  /** outputs 的 key 是节点 ID，值形如 { images: [{ filename, subfolder, type }] } */
  private findOutputFile(record: ComfyHistoryRecord): ComfyOutputFile | undefined {
    let found: ComfyOutputFile | undefined
    for (const output of Object.values(record.outputs ?? {})) {
      for (const key of ['videos', 'gifs', 'images'] as const) {
        const files = output?.[key]
        if (!Array.isArray(files)) continue
        for (const file of files) {
          if (file?.filename) found = file
        }
      }
    }
    return found
  }

  private viewUrl(file: ComfyOutputFile): string {
    const query = new URLSearchParams({
      filename: file.filename,
      subfolder: file.subfolder ?? '',
      type: file.type ?? 'output',
    })
    return `${this.config.baseUrl}/view?${query.toString()}`
  }

  /**
   * /queue 的结构是 { queue_running: [[number, prompt_id, graph, ...]], queue_pending: [...] }，
   * 第二项才是 prompt_id。只匹配数组里任意位置会出现误判（seed 恰好等于 id 前缀），
   * 所以只取索引 1。
   */
  private queueContains(queue: ComfyQueueResponse, taskId: string): 'running' | 'pending' | undefined {
    const hit = (list: Array<unknown>[] | undefined) =>
      Array.isArray(list) && list.some((entry) => Array.isArray(entry) && entry[1] === taskId)
    if (hit(queue?.queue_running)) return 'running'
    if (hit(queue?.queue_pending)) return 'pending'
    return undefined
  }
}
