import 'package:flutter_test/flutter_test.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:mt_llmkit/src/models/llm_model_base.dart';

/// A model whose stream is scripted, so `sendPromptResult` — which is shared
/// by both real backends — can be exercised without a GGUF file.
class _ScriptedModel extends LlmModelBase {
  @override
  final LlmConfig config;

  final List<StreamingChunk> chunks;

  /// Stands in for the tokenizer: one token per whitespace-separated word.
  int tokenizeCalls = 0;

  _ScriptedModel({required this.chunks, this.config = const LlmConfig()});

  @override
  Stream<StreamingChunk> sendPromptStream(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => Stream.fromIterable(chunks);

  @override
  Future<int> countTokens(String text) async {
    tokenizeCalls++;
    return text.trim().split(RegExp(r'\s+')).length;
  }

  // ── Not exercised here ───────────────────────────────────────────────────
  @override
  Future<Conversation> startConversation({
    String? systemPrompt,
    int? maxContextTokens,
    List<LlmChatMessage>? history,
    ContextOverflowPolicy overflowPolicy = ContextOverflowPolicy.allow,
    bool keepThinkingInHistory = false,
  }) async => throw UnsupportedError('no session');

  @override
  Future<void> loadModel(String localPath) async {}
  @override
  Stream<String> sendPrompt(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) => const Stream.empty();
  @override
  Future<String> sendPromptComplete(
    String prompt, {
    String? systemPrompt,
    List<LlamaContentPart>? attachments,
    GenerationOverrides? overrides,
  }) async => '';
  @override
  Future<void> dispose() async {}
  @override
  Future<void> unload() async {}
  @override
  Future<void> clean({bool resetConversations = true}) async {}
  @override
  Future<List<int>> tokenize(String text, {bool addSpecial = true}) async =>
      const [];
  @override
  Future<String> detokenize(List<int> tokens, {bool special = false}) async =>
      '';
  @override
  Future<int> contextSize() async => 0;
  @override
  Future<Map<String, String>> metadata() async => const {};
  @override
  Future<bool> get supportsStatePersistence async => false;
  @override
  Future<bool> saveState(String path, {required List<int> tokens}) async =>
      false;
  @override
  Future<List<int>> loadState(String path, {int? tokenCapacity}) async =>
      const [];
  @override
  Future<void> setLora(String path, {double scale = 1.0}) async {}
  @override
  Future<void> removeLora(String path) async {}
  @override
  Future<void> clearLoras() async {}
  @override
  Future<ModelDiagnostics> diagnostics() async =>
      throw UnsupportedError('no engine');
}

void main() {
  group('sendPromptResult', () {
    test('keeps reasoning, which sendPromptComplete would discard', () async {
      final model = _ScriptedModel(
        chunks: [
          StreamingChunk(text: '', thinking: 'Poland. Capital. '),
          StreamingChunk(text: '', thinking: 'Warsaw.'),
          StreamingChunk(text: 'The capital is '),
          StreamingChunk(text: 'Warsaw.'),
          StreamingChunk(text: '', isFinal: true, finishReason: 'stop'),
        ],
      );

      final result = await model.sendPromptResult('...');

      expect(result.text, 'The capital is Warsaw.');
      expect(result.thinking, 'Poland. Capital. Warsaw.');
      expect(result.finishReason, 'stop');
      expect(result.isTruncated, isFalse);
    });

    test('reports no reasoning as null, not as an empty string', () async {
      final model = _ScriptedModel(
        chunks: [
          StreamingChunk(text: 'Hi.'),
          StreamingChunk(text: '', isFinal: true, finishReason: 'stop'),
        ],
      );

      expect((await model.sendPromptResult('...')).thinking, isNull);
    });

    test('carries tool calls and truncation off the final chunk', () async {
      final model = _ScriptedModel(
        chunks: [
          StreamingChunk(
            text: '',
            isFinal: true,
            finishReason: 'length',
            toolCalls: const [
              LlmToolCall(index: 0, name: 'search', arguments: '{}'),
            ],
          ),
        ],
      );

      final result = await model.sendPromptResult('...');

      expect(result.needsToolResults, isTrue);
      expect(result.toolCalls.single.name, 'search');
      expect(result.isTruncated, isTrue);
    });
  });

  group('thinking budget truncation', () {
    List<StreamingChunk> reasoning(String thinking) => [
      StreamingChunk(text: '', thinking: thinking),
      StreamingChunk(text: 'answer'),
      StreamingChunk(text: '', isFinal: true, finishReason: 'stop'),
    ];

    test('forcedMessage gives an exact signal, with no tokenizing', () async {
      final model = _ScriptedModel(
        config: const LlmConfig(
          thinkingBudget: ThinkingBudget(
            maxTokens: 8,
            forcedMessage: 'Out of thinking budget.',
          ),
        ),
        chunks: reasoning('Thinking… Out of thinking budget.'),
      );

      final result = await model.sendPromptResult('...');

      expect(result.thinkingTruncated, isTrue);
      expect(result.thinkingTokens, isNull);
      expect(model.tokenizeCalls, 0);
    });

    test(
      'forcedMessage absent from the reasoning means not truncated',
      () async {
        final model = _ScriptedModel(
          config: const LlmConfig(
            thinkingBudget: ThinkingBudget(
              maxTokens: 8,
              forcedMessage: 'Out of thinking budget.',
            ),
          ),
          chunks: reasoning('Short thought.'),
        );

        expect(
          (await model.sendPromptResult('...')).thinkingTruncated,
          isFalse,
        );
      },
    );

    test('without forcedMessage it falls back to counting tokens', () async {
      final model = _ScriptedModel(
        config: const LlmConfig(thinkingBudget: ThinkingBudget(maxTokens: 3)),
        chunks: reasoning('one two three'),
      );

      final result = await model.sendPromptResult('...');

      expect(result.thinkingTokens, 3);
      expect(result.thinkingTruncated, isTrue);
      expect(model.tokenizeCalls, 1);
    });

    test('reasoning under the budget is not reported as truncated', () async {
      final model = _ScriptedModel(
        config: const LlmConfig(thinkingBudget: ThinkingBudget(maxTokens: 10)),
        chunks: reasoning('one two three'),
      );

      final result = await model.sendPromptResult('...');

      expect(result.thinkingTokens, 3);
      expect(result.thinkingTruncated, isFalse);
    });

    test('no budget means no measuring at all', () async {
      final model = _ScriptedModel(chunks: reasoning('one two three'));

      final result = await model.sendPromptResult('...');

      expect(result.thinkingTruncated, isFalse);
      expect(result.thinkingTokens, isNull);
      expect(model.tokenizeCalls, 0);
    });

    test('a per-request budget overrides the config', () async {
      final model = _ScriptedModel(
        config: const LlmConfig(thinkingBudget: ThinkingBudget(maxTokens: 999)),
        chunks: reasoning('one two three'),
      );

      final result = await model.sendPromptResult(
        '...',
        overrides: const GenerationOverrides(
          thinkingBudget: ThinkingBudget(maxTokens: 2),
        ),
      );

      expect(result.thinkingTruncated, isTrue);
    });
  });
}
