// lib/src/core/generation_overrides.dart

import 'package:llamadart/llamadart.dart'
    show GenerationParams, ThinkingBudget, ToolChoice;

import 'tools.dart';

/// Per-request sampling overrides.
///
/// Sampling settings otherwise come from [LlmConfig] and are fixed when the
/// model is loaded, so changing the temperature for a single request used to
/// mean reloading the model. Pass one of these to `sendPrompt*` instead — for
/// example a low temperature for a RAG answer and a higher one for chat, on
/// one loaded model.
///
/// Only sampling knobs that meaningfully vary per request are here. Grammar
/// triggers and preserved tokens stay in [LlmConfig]: they describe the model's
/// decoding setup, not one request.
class GenerationOverrides {
  final int? maxTokens;
  final double? temp;
  final int? topK;
  final double? topP;
  final double? minP;
  final double? penaltyRepeat;
  final double? presencePenalty;
  final int? seed;
  final List<String>? stopSequences;
  final String? grammar;
  final String? grammarRoot;
  final bool? enableThinking;
  final ThinkingBudget? thinkingBudget;

  /// Tools the model may call for this request. Declaring tools constrains
  /// decoding to their schema; completed calls arrive on
  /// [StreamingChunk.toolCalls].
  final List<LlmTool>? tools;

  /// Whether the model may, must, or must not call a tool. Ignored when
  /// [tools] is empty.
  final ToolChoice? toolChoice;

  /// Whether several tools may be called in one turn.
  final bool? parallelToolCalls;

  /// JSON-schema response format for structured output.
  final Map<String, dynamic>? responseFormat;

  const GenerationOverrides({
    this.maxTokens,
    this.temp,
    this.topK,
    this.topP,
    this.minP,
    this.penaltyRepeat,
    this.presencePenalty,
    this.seed,
    this.stopSequences,
    this.grammar,
    this.grammarRoot,
    this.enableThinking,
    this.thinkingBudget,
    this.tools,
    this.toolChoice,
    this.parallelToolCalls,
    this.responseFormat,
  });

  /// Applies these overrides on top of [base].
  GenerationParams applyTo(GenerationParams base) => base.copyWith(
    maxTokens: maxTokens,
    temp: temp,
    topK: topK,
    topP: topP,
    minP: minP,
    penalty: penaltyRepeat,
    presencePenalty: presencePenalty,
    seed: seed,
    stopSequences: stopSequences,
    grammar: grammar,
    grammarRoot: grammarRoot,
    thinkingBudget: thinkingBudget,
  );
}
