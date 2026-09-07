import 'dart:isolate';

import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';

/// Echoes whatever it is handed back to the main isolate.
///
/// The worker isolates hand `LlmConfig` and `GenerationOverrides` across the
/// port as objects rather than as hand-rolled maps, so these have to stay
/// sendable — a field holding something non-transferable would only fail at
/// runtime, on a device, while loading a model.
Future<void> _echo(List<Object?> args) async {
  (args[0] as SendPort).send(args[1]);
}

Future<Object?> _roundTrip(Object? value) async {
  final port = ReceivePort();
  await Isolate.spawn(_echo, [port.sendPort, value]);
  final result = await port.first;
  port.close();
  return result;
}

void main() {
  group('isolate transport', () {
    test(
      'LlmConfig survives the isolate boundary with every knob set',
      () async {
        const config = LlmConfig(
          nGpuLayers: 24,
          nCtx: 4096,
          gpuBackend: GpuBackend.metal,
          flashAttention: FlashAttention.enabled,
          cacheTypeK: KvCacheType.q8_0,
          cacheTypeV: KvCacheType.q4_0,
          loras: [LoraAdapterConfig(path: '/tmp/a.gguf', scale: 0.7)],
          grammarTriggers: [GenerationGrammarTrigger(type: 1, value: '<tool>')],
          thinkingBudget: ThinkingBudget(maxTokens: 128, endTag: '</think>'),
          stopSequences: ['<|im_end|>'],
          chatTemplate: '{{ messages }}',
        );

        final restored = await _roundTrip(config) as LlmConfig;

        expect(restored.nGpuLayersDefault, 24);
        expect(restored.gpuBackendDefault, GpuBackend.metal);
        expect(restored.flashAttentionDefault, FlashAttention.enabled);
        expect(restored.cacheTypeKDefault, KvCacheType.q8_0);
        expect(restored.cacheTypeVDefault, KvCacheType.q4_0);
        expect(restored.lorasDefault.single.path, '/tmp/a.gguf');
        expect(restored.grammarTriggersDefault.single.value, '<tool>');
        expect(restored.thinkingBudget?.endTag, '</think>');
        expect(restored.stopSequencesDefault, ['<|im_end|>']);
        expect(restored.chatTemplate, '{{ messages }}');
      },
    );

    test('GenerationOverrides survives the isolate boundary', () async {
      const overrides = GenerationOverrides(
        temp: 0.1,
        maxTokens: 256,
        seed: 42,
        stopSequences: ['STOP'],
        enableThinking: false,
        thinkingBudget: ThinkingBudget(maxTokens: 64),
      );

      final restored = await _roundTrip(overrides) as GenerationOverrides;

      expect(restored.temp, 0.1);
      expect(restored.maxTokens, 256);
      expect(restored.seed, 42);
      expect(restored.stopSequences, ['STOP']);
      expect(restored.enableThinking, isFalse);
      expect(restored.thinkingBudget?.maxTokens, 64);
    });

    test('declared tools survive the isolate boundary', () async {
      const overrides = GenerationOverrides(
        tools: [
          LlmTool(
            name: 'get_weather',
            description: 'Weather for a city',
            parameters: [],
          ),
        ],
        toolChoice: ToolChoice.required,
        parallelToolCalls: true,
        responseFormat: {'type': 'json_object'},
      );

      final restored = await _roundTrip(overrides) as GenerationOverrides;

      expect(restored.tools?.single.name, 'get_weather');
      expect(restored.toolChoice, ToolChoice.required);
      expect(restored.parallelToolCalls, isTrue);
      expect(restored.responseFormat, {'type': 'json_object'});
    });

    test('ModelDiagnostics survives the isolate boundary', () async {
      const diagnostics = ModelDiagnostics(
        backendName: 'Metal',
        availableBackends: 'Metal,CPU',
        resolvedGpuLayers: 32,
        gpuSupported: true,
        modelFileType: 'q4_K_M',
        contextSize: 4096,
        supportsVision: false,
        supportsAudio: false,
        supportsStatePersistence: true,
        vramTotal: 8000,
        vramFree: 4000,
      );

      final restored = await _roundTrip(diagnostics) as ModelDiagnostics;

      expect(restored.backendName, 'Metal');
      expect(restored.resolvedGpuLayers, 32);
      expect(restored.supportsStatePersistence, isTrue);
    });

    test(
      'only a structured output\'s response format is sent to the worker',
      () async {
        final output = LlmStructuredOutput.jsonObject(decoder: (json) => json);

        // The decoder *would* survive the hop — closures are sendable between
        // isolates of the same group — but decoding stays on the calling isolate
        // by choice, so the worker only ever receives this map.
        expect(await _roundTrip(output.responseFormat), {
          'type': 'json_object',
        });
      },
    );

    test('conversation history survives the isolate boundary', () async {
      // The session lives in the worker, so history and turn requests cross
      // the port as objects rather than hand-rolled maps.
      final messages = <LlmChatMessage>[
        const LlmChatMessage.user('question'),
        LlmChatMessage.assistant(
          'reply',
          thinking: 'reasoning',
          toolCalls: const [
            LlmToolCall(index: 0, id: 'c1', name: 'search', arguments: '{}'),
          ],
        ),
        LlmChatMessage.tool(
          const ToolResult(name: 'search', result: 'found', id: 'c1'),
        ),
        LlmChatMessage.user(
          'look at this',
          attachments: [LlamaImageContent(path: '/tmp/cat.png')],
        ),
      ];

      final restored = (await _roundTrip(messages) as List)
          .cast<LlmChatMessage>();

      expect(restored[1].thinking, 'reasoning');
      expect(restored[1].toolCalls.single.name, 'search');
      expect(restored[2].toolResult?.result, 'found');
      expect(restored[3].attachments.single, isA<LlamaImageContent>());
    });

    test('attachments survive the isolate boundary', () async {
      final attachments = <LlamaContentPart>[
        LlamaImageContent(path: '/tmp/cat.png'),
      ];

      final restored = await _roundTrip(attachments) as List;

      expect(restored.single, isA<LlamaImageContent>());
    });
  });
}
