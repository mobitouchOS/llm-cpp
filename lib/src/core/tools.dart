// lib/src/core/tools.dart

import 'dart:convert';

import 'package:llamadart/llamadart.dart'
    show LlamaCompletionChunkToolCall, ToolDefinition, ToolParam;

/// A tool the model may call.
///
/// Declaring a tool constrains decoding to the tool's schema and lets
/// llamadart parse the model's call out of whatever envelope its chat
/// template uses.
///
/// Unlike llamadart's [ToolDefinition] this carries no handler: the plugin
/// runs generation in a worker isolate, and executing your code there would
/// be both surprising and impossible to send across the port. Tool calls come
/// back as data in [StreamingChunk.toolCalls]; you run them and feed the
/// result into the next turn.
class LlmTool {
  /// Unique name the model uses to reference the tool (e.g. `get_weather`).
  final String name;

  /// What the tool does — this is what the model reasons about when deciding
  /// whether to call it.
  final String description;

  /// The tool's input schema.
  final List<ToolParam> parameters;

  const LlmTool({
    required this.name,
    required this.description,
    this.parameters = const [],
  });

  /// Converts to llamadart's shape for template and grammar generation.
  ///
  /// The handler is a stub that throws: nothing inside the worker isolate
  /// ever executes it, and reaching it would mean llamadart changed how tool
  /// declarations are consumed.
  ToolDefinition toToolDefinition() => ToolDefinition(
    name: name,
    description: description,
    parameters: parameters,
    handler: (_) async =>
        throw UnsupportedError('mt_llmkit does not execute tool handlers.'),
  );
}

/// A completed tool call requested by the model.
class LlmToolCall {
  /// Position of the call in the response, for parallel calls.
  final int index;

  /// Call id, when the model's format provides one.
  final String? id;

  /// Name of the tool the model wants to call.
  final String name;

  /// Raw JSON arguments as the model emitted them.
  final String arguments;

  const LlmToolCall({
    required this.index,
    required this.name,
    required this.arguments,
    this.id,
  });

  /// Decoded [arguments], or null when the model produced invalid JSON.
  Map<String, dynamic>? get argumentsJson {
    try {
      final decoded = jsonDecode(arguments);
      return decoded is Map<String, dynamic> ? decoded : null;
    } on FormatException {
      return null;
    }
  }

  Map<String, dynamic> toMap() => {
    'index': index,
    if (id != null) 'id': id,
    'name': name,
    'arguments': arguments,
  };

  static LlmToolCall fromMap(Map<String, dynamic> map) => LlmToolCall(
    index: map['index'] as int,
    id: map['id'] as String?,
    name: map['name'] as String? ?? '',
    arguments: map['arguments'] as String? ?? '',
  );

  @override
  String toString() => 'LlmToolCall($name, args: $arguments)';
}

/// Reassembles streamed tool-call deltas into complete calls.
///
/// llamadart streams a call's name and arguments as fragments across chunks,
/// the same way the OpenAI API does. Accumulating that in every caller would
/// be a trap, so the backends do it and emit finished calls on the final
/// chunk.
class ToolCallAccumulator {
  final _byIndex = <int, _PartialToolCall>{};

  void add(List<LlamaCompletionChunkToolCall> deltas) {
    for (final delta in deltas) {
      final partial = _byIndex.putIfAbsent(
        delta.index,
        () => _PartialToolCall(),
      );
      partial.id ??= delta.id;
      final name = delta.function?.name;
      if (name != null) partial.name.write(name);
      final args = delta.function?.arguments;
      if (args != null) partial.arguments.write(args);
    }
  }

  bool get isEmpty => _byIndex.isEmpty;

  List<LlmToolCall> build() {
    final indices = _byIndex.keys.toList()..sort();
    return [
      for (final index in indices)
        LlmToolCall(
          index: index,
          id: _byIndex[index]!.id,
          name: _byIndex[index]!.name.toString(),
          arguments: _byIndex[index]!.arguments.toString(),
        ),
    ];
  }
}

class _PartialToolCall {
  String? id;
  final StringBuffer name = StringBuffer();
  final StringBuffer arguments = StringBuffer();
}
