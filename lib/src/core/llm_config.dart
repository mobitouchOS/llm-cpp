import 'package:llamadart/llamadart.dart'
    show
        FlashAttention,
        GenerationGrammarTrigger,
        GpuBackend,
        KvCacheType,
        LoraAdapterConfig,
        ModelParams,
        SpeculativeDecodingConfig,
        ThinkingBudget;

class LlmConfig {
  // ── ModelParams ────────────────────────────────────────────────────────────

  /// Model layers offloaded to the GPU. `null` offloads everything llamadart
  /// can (`ModelParams.maxGpuLayers`); use `0` to force CPU-only inference.
  final int? nGpuLayers;

  final int? nCtx;

  /// Logical batch size (n_batch). `null` lets llama.cpp size it —
  /// `min(nCtx, 2048)` — which is what mobile devices want.
  final int? nBatch;

  /// CPU threads for inference. `null`/`0` = auto (llama.cpp picks).
  final int? nThreads;

  /// Number of threads for batch processing (n_threads_batch). 0 = auto.
  final int? numberOfThreadsBatch;

  /// Physical micro-batch size (n_ubatch). `null`/`0` lets llama.cpp size it —
  /// `min(nBatch, 512)`.
  final int? microBatchSize;

  /// Maximum parallel sequence slots in context memory (n_seq_max).
  final int? maxParallelSequences;

  /// LoRA adapters to load with the model.
  final List<LoraAdapterConfig>? loras;

  /// Custom chat template to override the model's built-in template.
  final String? chatTemplate;

  /// GPU backend to use for inference. Defaults to [GpuBackend.auto] which
  /// tries Vulkan → Metal → CUDA → CPU in order. Use [GpuBackend.cpu] to
  /// disable GPU acceleration entirely (useful when Vulkan causes crashes).
  final GpuBackend? gpuBackend;

  /// Path to the multimodal projector GGUF file (e.g. `mmproj-model-f16.gguf`).
  ///
  /// Required when using vision models (LLaVA, Gemma 3, Qwen VL, etc.).
  /// When set, pass images via the `attachments` parameter of
  /// [LocalModel.sendPromptStream] and related methods to enable vision
  /// inference.
  final String? mmprojPath;

  /// Flash attention mode. Required to be `auto` or `enabled` when either KV
  /// cache type is quantized — see [cacheTypeK].
  final FlashAttention? flashAttention;

  /// Quantization of the K side of the KV cache.
  ///
  /// [KvCacheType.q8_0] halves and [KvCacheType.q4_0] quarters KV memory,
  /// which is the single largest memory saving available on a phone at long
  /// context lengths. Both require flash attention (see [flashAttention]).
  final KvCacheType? cacheTypeK;

  /// Quantization of the V side of the KV cache. See [cacheTypeK].
  final KvCacheType? cacheTypeV;

  /// Whether to use a unified KV cache across sequences.
  final bool? kvUnified;

  /// Memory-maps the model file instead of reading it into RAM. On by default
  /// in llama.cpp; turning it off raises resident memory sharply.
  final bool? useMmap;

  /// Locks the model in RAM so the OS cannot page it out.
  final bool? useMlock;

  /// RoPE base frequency override (for context-extended models).
  final double? ropeFrequencyBase;

  /// RoPE frequency scaling override (for context-extended models).
  final double? ropeFrequencyScale;

  // ── GenerationParams ───────────────────────────────────────────────────────

  final int? nPredict;
  final double? temp;
  final int? topK;
  final double? topP;

  /// Min-P sampling threshold. Set to 0.0 to disable.
  final double? minP;

  final double? penaltyRepeat;

  /// Presence penalty applied to tokens already present in the output.
  final double? presencePenalty;

  /// Random seed for the sampler. null = time-based seed.
  final int? seed;

  /// Strings that immediately stop generation when encountered.
  final List<String>? stopSequences;

  /// GBNF grammar string for structured output (e.g. `'root ::= "yes" | "no"'`).
  final String? grammar;

  /// Whether grammar should be lazily activated by [grammarTriggers].
  final bool? grammarLazy;

  /// Lazy grammar activation triggers. Used together with [grammarLazy].
  final List<GenerationGrammarTrigger>? grammarTriggers;

  /// Tokens to preserve during constrained decoding.
  final List<String>? preservedTokens;

  /// Grammar start symbol. Defaults to `'root'`.
  final String? grammarRoot;

  /// Reuse matching prompt prefixes from previous requests to reduce latency.
  final bool? reusePromptPrefix;

  /// Chunk flush threshold by token pieces (lower = finer stream granularity).
  final int? streamBatchTokenThreshold;

  /// Chunk flush threshold by byte size (lower = finer stream granularity).
  final int? streamBatchByteThreshold;

  /// Whether reasoning models may emit a thinking block. Defaults to `true`
  /// in llamadart.
  ///
  /// Reasoning tokens arrive as [StreamingChunk.thinking], separate from the
  /// answer text, and they are decoded at full cost. Turn this off for
  /// short-answer workloads on a reasoning model, or bound it with
  /// [thinkingBudget].
  final bool? enableThinking;

  /// Caps how many tokens each reasoning block may use before the closing tag
  /// is forced. Only meaningful while [enableThinking] is on.
  final ThinkingBudget? thinkingBudget;

  /// Backend-native speculative decoding. `null` uses llamadart's default
  /// (off); a config selects the strategy (n-gram, MTP, draft model, …).
  final SpeculativeDecodingConfig? speculativeDecoding;

  const LlmConfig({
    this.nGpuLayers,
    this.nCtx,
    this.nBatch,
    this.nThreads,
    this.numberOfThreadsBatch,
    this.microBatchSize,
    this.maxParallelSequences,
    this.loras,
    this.chatTemplate,
    this.gpuBackend,
    this.mmprojPath,
    this.flashAttention,
    this.cacheTypeK,
    this.cacheTypeV,
    this.kvUnified,
    this.useMmap,
    this.useMlock,
    this.ropeFrequencyBase,
    this.ropeFrequencyScale,
    this.nPredict,
    this.temp,
    this.topK,
    this.topP,
    this.minP,
    this.penaltyRepeat,
    this.presencePenalty,
    this.seed,
    this.stopSequences,
    this.grammar,
    this.grammarLazy,
    this.grammarTriggers,
    this.preservedTokens,
    this.grammarRoot,
    this.reusePromptPrefix,
    this.streamBatchTokenThreshold,
    this.streamBatchByteThreshold,
    this.enableThinking,
    this.thinkingBudget,
    this.speculativeDecoding,
  });

  // ModelParams defaults
  //
  // Unset values resolve to llamadart's own auto-sizing rather than to fixed
  // numbers: `gpuLayers = maxGpuLayers` (full offload), `batchSize = 0` →
  // `min(nCtx, 2048)`, `microBatchSize = 0` → `min(nBatch, 512)`,
  // `numberOfThreads = 0` → llama.cpp's thread heuristic.
  int get nGpuLayersDefault => nGpuLayers ?? ModelParams.maxGpuLayers;
  int get nCtxDefault => nCtx ?? 8192;
  int get nBatchDefault => nBatch ?? 0;
  int get nThreadsDefault => nThreads ?? 0;
  int get numberOfThreadsBatchDefault => numberOfThreadsBatch ?? 0;
  int get microBatchSizeDefault => microBatchSize ?? 0;
  int get maxParallelSequencesDefault => maxParallelSequences ?? 1;
  List<LoraAdapterConfig> get lorasDefault => loras ?? const [];
  GpuBackend get gpuBackendDefault => gpuBackend ?? GpuBackend.auto;
  FlashAttention get flashAttentionDefault =>
      flashAttention ?? FlashAttention.auto;
  KvCacheType get cacheTypeKDefault => cacheTypeK ?? KvCacheType.f16;
  KvCacheType get cacheTypeVDefault => cacheTypeV ?? KvCacheType.f16;
  bool get useMmapDefault => useMmap ?? true;
  bool get useMlockDefault => useMlock ?? false;

  // GenerationParams defaults
  int get nPredictDefault => nPredict ?? 8192;
  double get tempDefault => temp ?? 0.72;
  int get topKDefault => topK ?? 64;
  double get topPDefault => topP ?? 0.95;
  double get minPDefault => minP ?? 0.0;
  double get penaltyRepeatDefault => penaltyRepeat ?? 1.1;
  double get presencePenaltyDefault => presencePenalty ?? 0.0;
  List<String> get stopSequencesDefault => stopSequences ?? const [];
  bool get grammarLazyDefault => grammarLazy ?? false;
  List<GenerationGrammarTrigger> get grammarTriggersDefault =>
      grammarTriggers ?? const [];
  List<String> get preservedTokensDefault => preservedTokens ?? const [];
  String get grammarRootDefault => grammarRoot ?? 'root';
  bool get reusePromptPrefixDefault => reusePromptPrefix ?? true;
  int get streamBatchTokenThresholdDefault => streamBatchTokenThreshold ?? 1;
  int get streamBatchByteThresholdDefault => streamBatchByteThreshold ?? 512;
  bool get enableThinkingDefault => enableThinking ?? true;

  /// Throws [ArgumentError] for combinations llama.cpp rejects.
  ///
  /// Called by the model backends before loading, so a bad combination fails
  /// with a readable message instead of deep inside the worker isolate.
  void validate() {
    if ((cacheTypeKDefault != KvCacheType.f16 ||
            cacheTypeVDefault != KvCacheType.f16) &&
        flashAttentionDefault == FlashAttention.disabled) {
      throw ArgumentError(
        'A quantized KV cache (cacheTypeK=$cacheTypeKDefault, '
        'cacheTypeV=$cacheTypeVDefault) requires flash attention. Set '
        'flashAttention to FlashAttention.auto or FlashAttention.enabled, or '
        'use KvCacheType.f16 for both.',
      );
    }
  }
}
