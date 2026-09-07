import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

void main() {
  group('LlmChatMessage JSON', () {
    test('round-trips a plain user turn', () {
      const message = LlmChatMessage.user('What is the capital of Poland?');

      final restored = LlmChatMessage.fromJson(message.toJson());

      expect(restored.role, LlmChatRole.user);
      expect(restored.text, 'What is the capital of Poland?');
    });

    test('keeps reasoning and tool calls on one assistant turn', () {
      // llamadart's own message JSON cannot express both: it folds reasoning
      // into reasoning_content and reshapes tool messages.
      const message = LlmChatMessage.assistant(
        'Let me look that up.',
        thinking: 'The user wants current weather.',
        toolCalls: [
          LlmToolCall(
            index: 0,
            id: 'c1',
            name: 'get_weather',
            arguments: '{"city":"Kraków"}',
          ),
          LlmToolCall(index: 1, name: 'get_time', arguments: '{}'),
        ],
      );

      final restored = LlmChatMessage.fromJson(message.toJson());

      expect(restored.thinking, 'The user wants current weather.');
      expect(restored.toolCalls, hasLength(2));
      expect(restored.toolCalls.first.argumentsJson, {'city': 'Kraków'});
      expect(restored.toolCalls.last.name, 'get_time');
    });

    test('round-trips a tool result', () {
      final message = LlmChatMessage.tool(
        const ToolResult(name: 'get_weather', result: '18°C', id: 'c1'),
      );

      final restored = LlmChatMessage.fromJson(message.toJson());

      expect(restored.role, LlmChatRole.tool);
      expect(restored.toolResult?.name, 'get_weather');
      expect(restored.toolResult?.result, '18°C');
      expect(restored.toolResult?.id, 'c1');
      // Tool exchanges belong to the turn that triggered them.
      expect(restored.continuesPreviousTurn, isTrue);
    });

    test('does not serialize attachments', () {
      // Attachments can be megabytes of raw bytes; whether to persist them
      // alongside a conversation is the app's decision.
      final message = LlmChatMessage.user(
        'look',
        attachments: [LlamaImageContent(path: '/tmp/cat.png')],
      );

      expect(message.toJson().containsKey('attachments'), isFalse);
      expect(message.attachments, hasLength(1));
    });

    test('an unknown role decodes to user rather than throwing', () {
      final restored = LlmChatMessage.fromJson(const {
        'role': 'from-the-future',
        'text': 'hi',
      });

      expect(restored.role, LlmChatRole.user);
    });
  });

  group('ToolResult', () {
    test('forCall carries the call id across', () {
      const call = LlmToolCall(
        index: 0,
        id: 'call_42',
        name: 'search',
        arguments: '{}',
      );

      final result = ToolResult.forCall(call, 'found it');

      expect(result.name, 'search');
      expect(result.id, 'call_42');
      expect(result.result, 'found it');
    });

    test('round-trips through JSON', () {
      const result = ToolResult(name: 'search', result: {'hits': 3}, id: 'c1');

      final restored = ToolResult.fromJson(result.toJson());

      expect(restored.name, 'search');
      expect(restored.result, {'hits': 3});
      expect(restored.id, 'c1');
    });
  });
}
