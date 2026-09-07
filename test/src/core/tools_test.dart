import 'package:flutter_test/flutter_test.dart';
import 'package:llamadart/llamadart.dart'
    show LlamaCompletionChunkFunction, LlamaCompletionChunkToolCall, ToolParams;
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/core/tools.dart';

LlamaCompletionChunkToolCall _delta(
  int index, {
  String? id,
  String? name,
  String? arguments,
}) => LlamaCompletionChunkToolCall(
  index: index,
  id: id,
  function: LlamaCompletionChunkFunction(name: name, arguments: arguments),
);

void main() {
  group('ToolCallAccumulator', () {
    test('reassembles a call streamed as fragments', () {
      final acc = ToolCallAccumulator()
        ..add([_delta(0, id: 'call_1', name: 'get_')])
        ..add([_delta(0, name: 'weather', arguments: '{"loc')])
        ..add([_delta(0, arguments: 'ation":"Kraków"}')]);

      final call = acc.build().single;

      expect(call.id, 'call_1');
      expect(call.name, 'get_weather');
      expect(call.arguments, '{"location":"Kraków"}');
      expect(call.argumentsJson, {'location': 'Kraków'});
    });

    test('keeps parallel calls apart and orders them by index', () {
      final acc = ToolCallAccumulator()
        ..add([
          _delta(1, name: 'second', arguments: '{}'),
          _delta(0, name: 'first', arguments: '{}'),
        ]);

      expect(acc.build().map((c) => c.name), ['first', 'second']);
    });

    test('is empty until a delta arrives', () {
      final acc = ToolCallAccumulator();

      expect(acc.isEmpty, isTrue);
      expect(acc.build(), isEmpty);

      acc.add([_delta(0, name: 'x', arguments: '{}')]);
      expect(acc.isEmpty, isFalse);
    });

    test('returns null arguments rather than throwing on invalid JSON', () {
      final acc = ToolCallAccumulator()
        ..add([_delta(0, name: 'broken', arguments: '{"unterminated')]);

      expect(acc.build().single.argumentsJson, isNull);
    });

    test('survives a map round-trip through the worker port', () {
      const call = LlmToolCall(
        index: 2,
        id: 'call_9',
        name: 'search',
        arguments: '{"q":"rag"}',
      );

      final restored = LlmToolCall.fromMap(call.toMap());

      expect(restored.index, 2);
      expect(restored.id, 'call_9');
      expect(restored.name, 'search');
      expect(restored.argumentsJson, {'q': 'rag'});
    });
  });

  group('LlmTool', () {
    test('converts to a llamadart tool definition with a schema', () {
      const tool = LlmTool(
        name: 'get_weather',
        description: 'Current weather for a city',
        parameters: [],
      );

      final definition = tool.toToolDefinition();

      expect(definition.name, 'get_weather');
      expect(definition.description, 'Current weather for a city');
    });

    test('its stub handler is never meant to run', () {
      final definition = const LlmTool(
        name: 't',
        description: 'd',
      ).toToolDefinition();

      expect(
        () => definition.handler(const ToolParams({})),
        throwsUnsupportedError,
      );
    });
  });
}
