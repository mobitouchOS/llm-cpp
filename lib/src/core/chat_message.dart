// lib/src/core/chat_message.dart

import 'dart:convert';

import 'package:llamadart/llamadart.dart'
    show
        LlamaAudioContent,
        LlamaChatMessage,
        LlamaChatRole,
        LlamaContentPart,
        LlamaImageContent,
        LlamaTextContent,
        LlamaThinkingContent,
        LlamaToolCallContent,
        LlamaToolResultContent;

import 'tools.dart';

/// Who produced a message.
enum LlmChatRole { system, user, assistant, tool }

/// The outcome of running a tool the model asked for.
class ToolResult {
  /// The tool's name — must match the [LlmToolCall] being answered.
  final String name;

  /// What the tool returned. A String, or anything `jsonEncode` accepts.
  final Object? result;

  /// The call id, when the model's format provided one.
  final String? id;

  const ToolResult({required this.name, required this.result, this.id});

  /// Answers a specific [call], carrying its id across automatically.
  ToolResult.forCall(LlmToolCall call, this.result)
    : name = call.name,
      id = call.id;

  Map<String, dynamic> toJson() => {
    'name': name,
    if (id != null) 'id': id,
    'result': result,
  };

  factory ToolResult.fromJson(Map<String, dynamic> json) => ToolResult(
    name: json['name'] as String,
    id: json['id'] as String?,
    result: json['result'],
  );

  @override
  String toString() => 'ToolResult($name)';
}

/// One turn of a conversation.
///
/// This is the plugin's own history type rather than llamadart's
/// [LlamaChatMessage]. That one is prompt-shaped and lossy on the way out —
/// its `toJson()` keeps only the first tool result in a message and folds
/// reasoning into a `reasoning_content` string — so it round-trips badly for
/// anything that wants to store a conversation and restore it later.
class LlmChatMessage {
  final LlmChatRole role;

  /// The message text. Empty for a turn that only carries tool calls.
  final String text;

  /// Reasoning the model emitted for this turn, on assistant messages.
  final String? thinking;

  /// Tool calls the model requested, on assistant messages.
  final List<LlmToolCall> toolCalls;

  /// The tool's answer, on `tool` messages.
  final ToolResult? toolResult;

  /// Image or audio parts that accompanied the text.
  ///
  /// Not written by [toJson]: attachments can be megabytes of raw bytes, and
  /// whether to persist them alongside a conversation is the app's call. They
  /// do travel with the object itself, including across an isolate port.
  final List<LlamaContentPart> attachments;

  /// Whether this message continues the previous turn rather than starting a
  /// new one. Tool exchanges use this so context trimming treats a turn and
  /// its tool round trips as one unit.
  final bool continuesPreviousTurn;

  const LlmChatMessage({
    required this.role,
    this.text = '',
    this.thinking,
    this.toolCalls = const [],
    this.toolResult,
    this.attachments = const [],
    this.continuesPreviousTurn = false,
  });

  const LlmChatMessage.user(
    this.text, {
    this.attachments = const [],
    this.continuesPreviousTurn = false,
  }) : role = LlmChatRole.user,
       thinking = null,
       toolCalls = const [],
       toolResult = null;

  const LlmChatMessage.assistant(
    this.text, {
    this.thinking,
    this.toolCalls = const [],
  }) : role = LlmChatRole.assistant,
       toolResult = null,
       attachments = const [],
       continuesPreviousTurn = false;

  LlmChatMessage.tool(ToolResult result)
    : role = LlmChatRole.tool,
      text = '',
      thinking = null,
      toolCalls = const [],
      toolResult = result,
      attachments = const [],
      continuesPreviousTurn = true;

  Map<String, dynamic> toJson() => {
    'role': role.name,
    if (text.isNotEmpty) 'text': text,
    if (thinking != null) 'thinking': thinking,
    if (toolCalls.isNotEmpty)
      'toolCalls': toolCalls.map((c) => c.toMap()).toList(),
    if (toolResult != null) 'toolResult': toolResult!.toJson(),
    if (continuesPreviousTurn) 'continuesPreviousTurn': true,
  };

  factory LlmChatMessage.fromJson(Map<String, dynamic> json) => LlmChatMessage(
    role: LlmChatRole.values.firstWhere(
      (r) => r.name == json['role'],
      orElse: () => LlmChatRole.user,
    ),
    text: json['text'] as String? ?? '',
    thinking: json['thinking'] as String?,
    toolCalls: [
      for (final call in (json['toolCalls'] as List?) ?? const [])
        LlmToolCall.fromMap((call as Map).cast<String, dynamic>()),
    ],
    toolResult: json['toolResult'] == null
        ? null
        : ToolResult.fromJson(
            (json['toolResult'] as Map).cast<String, dynamic>(),
          ),
    continuesPreviousTurn: json['continuesPreviousTurn'] as bool? ?? false,
  );

  @override
  String toString() =>
      'LlmChatMessage(${role.name}, ${text.length} chars'
      '${toolCalls.isEmpty ? '' : ', ${toolCalls.length} tool calls'}'
      '${toolResult == null ? '' : ', tool result'})';
}

// ── llamadart conversion ─────────────────────────────────────────────────────

const _roleMap = {
  LlmChatRole.system: LlamaChatRole.system,
  LlmChatRole.user: LlamaChatRole.user,
  LlmChatRole.assistant: LlamaChatRole.assistant,
  LlmChatRole.tool: LlamaChatRole.tool,
};

/// Converts to llamadart's prompt-shaped message.
///
/// A tool result becomes its own message part: llamadart's `toJson()` keeps
/// only the first [LlamaToolResultContent] in a message, so several results
/// have to be several messages.
LlamaChatMessage toLlamaMessage(LlmChatMessage message) {
  final parts = <LlamaContentPart>[
    if (message.thinking != null && message.thinking!.isNotEmpty)
      LlamaThinkingContent(message.thinking!),
    if (message.text.isNotEmpty) LlamaTextContent(message.text),
    ...message.attachments,
    for (final call in message.toolCalls)
      LlamaToolCallContent(
        id: call.id,
        name: call.name,
        arguments: call.argumentsJson ?? const {},
        rawJson: call.arguments,
      ),
    if (message.toolResult != null)
      LlamaToolResultContent(
        name: message.toolResult!.name,
        result: message.toolResult!.result,
        id: message.toolResult!.id,
      ),
  ];

  return LlamaChatMessage.withContent(
    role: _roleMap[message.role]!,
    content: parts.isEmpty ? [const LlamaTextContent('')] : parts,
    continuesPreviousTurn: message.continuesPreviousTurn,
  );
}

/// Converts back from llamadart's message, which is what [ChatSession] stores.
LlmChatMessage fromLlamaMessage(LlamaChatMessage message) {
  final text = StringBuffer();
  final thinking = StringBuffer();
  final attachments = <LlamaContentPart>[];
  final toolCalls = <LlmToolCall>[];
  ToolResult? toolResult;

  var index = 0;
  for (final part in message.parts) {
    switch (part) {
      case LlamaTextContent():
        text.write(part.text);
      case LlamaThinkingContent():
        thinking.write(part.thinking);
      case LlamaImageContent():
      case LlamaAudioContent():
        attachments.add(part);
      case LlamaToolCallContent():
        toolCalls.add(
          LlmToolCall(
            index: index++,
            id: part.id,
            name: part.name,
            arguments: part.rawJson.isNotEmpty
                ? part.rawJson
                : jsonEncode(part.arguments),
          ),
        );
      case LlamaToolResultContent():
        toolResult ??= ToolResult(
          name: part.name,
          result: part.result,
          id: part.id,
        );
      default:
        break;
    }
  }

  return LlmChatMessage(
    role: _roleMap.entries
        .firstWhere(
          (e) => e.value == message.role,
          orElse: () => const MapEntry(LlmChatRole.user, LlamaChatRole.user),
        )
        .key,
    text: text.toString(),
    thinking: thinking.isEmpty ? null : thinking.toString(),
    toolCalls: toolCalls,
    toolResult: toolResult,
    attachments: attachments,
    continuesPreviousTurn: message.continuesPreviousTurn,
  );
}
