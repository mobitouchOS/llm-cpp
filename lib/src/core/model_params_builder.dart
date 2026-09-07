// lib/src/core/model_params_builder.dart

import 'package:llamadart/llamadart.dart';

import 'llm_config.dart';

/// Translates [LlmConfig] into llamadart's [ModelParams].
///
/// Shared by both backends and the RAG coordinator so a new knob only has to
/// be wired once.
ModelParams buildModelParams(LlmConfig config) => ModelParams(
  contextSize: config.nCtxDefault,
  gpuLayers: config.nGpuLayersDefault,
  batchSize: config.nBatchDefault,
  numberOfThreads: config.nThreadsDefault,
  numberOfThreadsBatch: config.numberOfThreadsBatchDefault,
  microBatchSize: config.microBatchSizeDefault,
  maxParallelSequences: config.maxParallelSequencesDefault,
  loras: config.lorasDefault,
  chatTemplate: config.chatTemplate,
  preferredBackend: config.gpuBackendDefault,
  flashAttention: config.flashAttentionDefault,
  cacheTypeK: config.cacheTypeKDefault,
  cacheTypeV: config.cacheTypeVDefault,
  kvUnified: config.kvUnified,
  useMmap: config.useMmapDefault,
  useMlock: config.useMlockDefault,
  ropeFrequencyBase: config.ropeFrequencyBase,
  ropeFrequencyScale: config.ropeFrequencyScale,
);

/// Translates [LlmConfig] into llamadart's [GenerationParams].
GenerationParams buildGenerationParams(LlmConfig config) => GenerationParams(
  maxTokens: config.nPredictDefault,
  temp: config.tempDefault,
  topK: config.topKDefault,
  topP: config.topPDefault,
  minP: config.minPDefault,
  penalty: config.penaltyRepeatDefault,
  presencePenalty: config.presencePenaltyDefault,
  seed: config.seed,
  stopSequences: config.stopSequencesDefault,
  grammar: config.grammar,
  grammarLazy: config.grammarLazyDefault,
  thinkingBudget: config.thinkingBudget,
  grammarTriggers: config.grammarTriggersDefault,
  preservedTokens: config.preservedTokensDefault,
  grammarRoot: config.grammarRootDefault,
  speculativeDecodingConfig: config.speculativeDecoding,
  reusePromptPrefix: config.reusePromptPrefixDefault,
  streamBatchTokenThreshold: config.streamBatchTokenThresholdDefault,
  streamBatchByteThreshold: config.streamBatchByteThresholdDefault,
);

/// Builds the chat messages for one request.
///
/// A [systemPrompt] becomes a real `system` message so the model's chat
/// template can place it correctly, instead of being prepended to the user
/// turn where the template would treat it as user text.
List<LlamaChatMessage> buildMessages(
  String prompt, {
  String? systemPrompt,
  List<LlamaContentPart>? attachments,
}) => [
  if (systemPrompt != null && systemPrompt.isNotEmpty)
    LlamaChatMessage.fromText(role: LlamaChatRole.system, text: systemPrompt),
  if (attachments != null && attachments.isNotEmpty)
    LlamaChatMessage.withContent(
      role: LlamaChatRole.user,
      content: [LlamaTextContent(prompt), ...attachments],
    )
  else
    LlamaChatMessage.fromText(role: LlamaChatRole.user, text: prompt),
];
