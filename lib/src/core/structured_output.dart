// lib/src/core/structured_output.dart

import 'package:llamadart/llamadart.dart'
    show LlamaInferenceException, LlamaStructuredOutput;

import 'performance_metrics.dart';
import 'streaming_result.dart';

/// A typed JSON contract for a generation.
///
/// Declaring one constrains decoding to the schema — the model cannot emit
/// anything that would not parse — and decodes the result into `T`.
///
/// This object stays on the calling isolate: only [responseFormat] — a plain
/// map — is sent to the worker, and the generated text is decoded back here.
/// The decoder would technically survive the hop (closures are sendable
/// between isolates of the same group), but running your decoding code, and
/// whatever it captures, inside the inference worker is not something this
/// plugin should decide for you.
class LlmStructuredOutput<T> {
  final LlamaStructuredOutput<T> _inner;

  LlmStructuredOutput._(this._inner);

  /// Any JSON object, with no schema constraint.
  factory LlmStructuredOutput.jsonObject({
    required T Function(Map<String, dynamic> json) decoder,
  }) =>
      LlmStructuredOutput._(LlamaStructuredOutput.jsonObject(decoder: decoder));

  /// An object-shaped JSON Schema.
  ///
  /// Throws [LlamaUnsupportedException] **here, at construction**, when the
  /// schema uses constructs llamadart cannot turn into a GBNF grammar — not
  /// halfway through a generation.
  factory LlmStructuredOutput.jsonSchema({
    required Map<String, dynamic> schema,
    required T Function(Map<String, dynamic> json) decoder,
    String? name,
    String? description,
    bool strict = true,
  }) => LlmStructuredOutput._(
    LlamaStructuredOutput.jsonSchema(
      schema: schema,
      decoder: decoder,
      name: name,
      description: description,
      strict: strict,
    ),
  );

  /// A JSON Schema whose root is not an object (an array, a string, …).
  factory LlmStructuredOutput.jsonValueSchema({
    required Map<String, dynamic> schema,
    required T Function(Object? value) decoder,
    String? name,
    String? description,
    bool strict = true,
  }) => LlmStructuredOutput._(
    LlamaStructuredOutput.jsonValueSchema(
      schema: schema,
      decoder: decoder,
      name: name,
      description: description,
      strict: strict,
    ),
  );

  /// The only part of this object that crosses an isolate boundary.
  Map<String, dynamic> get responseFormat => _inner.responseFormat;

  /// The declared schema, or null for [LlmStructuredOutput.jsonObject].
  Map<String, dynamic>? get schema => _inner.schema;

  /// Decodes model output.
  ///
  /// Throws [LlamaInferenceException] for malformed JSON, output that does not
  /// match [schema], or a decoder that itself throws.
  T parse(String output) => _inner.parse(output);
}

/// A completed structured generation.
class LlmStructuredResult<T> {
  /// The decoded value.
  final T value;

  /// The raw JSON the model produced, before decoding.
  final String rawJson;

  /// Reasoning text, when the model emitted a thinking block. It is never part
  /// of [rawJson].
  final String? thinking;

  final String? finishReason;
  final PerformanceMetrics? metrics;

  const LlmStructuredResult({
    required this.value,
    required this.rawJson,
    this.thinking,
    this.finishReason,
    this.metrics,
  });
}

/// Decodes a streamed generation into the type its [LlmStructuredOutput]
/// declares.
extension LlmStructuredStream on Stream<StreamingChunk> {
  /// Consumes the stream and decodes the accumulated text.
  ///
  /// Only [StreamingChunk.text] is buffered: reasoning arrives on its own
  /// channel and is not part of the JSON.
  Future<LlmStructuredResult<T>> parseStructured<T>(
    LlmStructuredOutput<T> output,
  ) async {
    final json = StringBuffer();
    final thinking = StringBuffer();
    String? finishReason;
    PerformanceMetrics? metrics;

    await for (final chunk in this) {
      json.write(chunk.text);
      if (chunk.thinking != null) thinking.write(chunk.thinking);
      if (chunk.isFinal) {
        finishReason = chunk.finishReason;
        metrics = chunk.metrics;
      }
    }

    // A budget-truncated response is almost always unterminated JSON. Saying so
    // beats letting jsonDecode report a confusing offset in the middle of it.
    if (finishReason == 'length') {
      throw LlamaInferenceException(
        'Structured output was truncated at the token budget, so it is not '
        'valid JSON. Raise nPredict (or GenerationOverrides.maxTokens) or '
        'simplify the schema.',
        json.toString(),
      );
    }

    return LlmStructuredResult<T>(
      value: output.parse(json.toString()),
      rawJson: json.toString(),
      thinking: thinking.isEmpty ? null : thinking.toString(),
      finishReason: finishReason,
      metrics: metrics,
    );
  }
}
