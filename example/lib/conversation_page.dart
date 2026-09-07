// example/lib/conversation_page.dart
//
// Multi-turn chat with a local GGUF model:
//
//  1. Reuses the model file downloaded on the LLM tab
//  2. Keeps history, so follow-up questions work without prompt concatenation
//  3. Shows the reasoning channel separately from the answer
//  4. Reports turns permanently dropped to keep the prompt inside nCtx

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:path_provider/path_provider.dart';

class ConversationPage extends StatefulWidget {
  const ConversationPage({super.key});

  @override
  State<ConversationPage> createState() => _ConversationPageState();
}

class _Turn {
  final bool fromUser;
  String text;
  String thinking;

  _Turn({required this.fromUser, this.text = ''}) : thinking = '';
}

class _ConversationPageState extends State<ConversationPage> {
  LocalModel? _model;
  Conversation? _chat;
  StreamSubscription<ConversationChunk>? _subscription;
  StreamSubscription<ContextTrimEvent>? _trimSubscription;

  final List<_Turn> _turns = [];
  final _inputController = TextEditingController();
  final _scrollController = ScrollController();

  bool _isBusy = false;
  String _status = '';
  String? _modelPath;

  @override
  void initState() {
    super.initState();
    _checkModel();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    _trimSubscription?.cancel();
    // Widget teardown is synchronous; let the model release itself.
    unawaited(_chat?.close() ?? Future<void>.value());
    unawaited(_model?.dispose() ?? Future<void>.value());
    _inputController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _checkModel() async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/model.gguf');
    if (file.existsSync()) {
      setState(() => _modelPath = file.path);
    } else {
      setState(() => _status = 'Download a model on the LLM tab first.');
    }
  }

  Future<void> _start() async {
    if (_modelPath == null) return;
    setState(() {
      _isBusy = true;
      _status = 'Loading model…';
    });

    try {
      final model = LocalModel(
        config: const LlmConfig(nCtx: 4096, nPredict: 512, temp: 0.7),
      );
      await model.loadModel(_modelPath!);

      final chat = await model.startConversation(
        systemPrompt: 'You are a concise, helpful assistant.',
      );

      // History is trimmed as it approaches nCtx, and the trim is permanent —
      // without this the only symptom is the model appearing to forget.
      _trimSubscription = chat.trims.listen((event) {
        setState(
          () => _status =
              'Dropped ${event.dropped.length} older messages to fit '
              'the context window.',
        );
      });

      setState(() {
        _model = model;
        _chat = chat;
        _isBusy = false;
        _status = 'Ready';
      });
    } catch (e) {
      setState(() {
        _isBusy = false;
        _status = 'Failed to start: $e';
      });
    }
  }

  Future<void> _send() async {
    final chat = _chat;
    final message = _inputController.text.trim();
    if (chat == null || message.isEmpty || _isBusy) return;

    _inputController.clear();
    final reply = _Turn(fromUser: false);
    setState(() {
      _turns
        ..add(_Turn(fromUser: true, text: message))
        ..add(reply);
      _isBusy = true;
      _status = 'Generating…';
    });

    _subscription = chat.send(message).listen(
      (chunk) {
        setState(() {
          reply.text += chunk.text;
          // Reasoning has its own channel and never mixes into text.
          if (chunk.thinking != null) reply.thinking += chunk.thinking!;
          if (chunk.isFinal) {
            _isBusy = false;
            _status = chunk.isTruncated
                ? 'Stopped: token budget exhausted'
                : 'Ready';
          }
        });
        _scrollToEnd();
      },
      onError: (Object error) => setState(() {
        _isBusy = false;
        _status = 'Error: $error';
      }),
    );
  }

  Future<void> _resetChat() async {
    await _subscription?.cancel();
    await _chat?.reset();
    setState(() {
      _turns.clear();
      _isBusy = false;
      _status = 'History cleared';
    });
  }

  void _scrollToEnd() {
    if (!_scrollController.hasClients) return;
    _scrollController.jumpTo(_scrollController.position.maxScrollExtent);
  }

  @override
  Widget build(BuildContext context) {
    if (_chat == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(_status),
            const SizedBox(height: 12),
            FilledButton(
              onPressed: _modelPath == null || _isBusy ? null : _start,
              child: const Text('Start conversation'),
            ),
          ],
        ),
      );
    }

    return Column(
      children: [
        Container(
          color: Colors.blue.shade50,
          width: double.infinity,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          child: Text(
            _status,
            style: TextStyle(fontSize: 11, color: Colors.blue.shade700),
          ),
        ),
        Expanded(
          child: ListView.builder(
            controller: _scrollController,
            padding: const EdgeInsets.all(16),
            itemCount: _turns.length,
            itemBuilder: (context, index) => _TurnBubble(turn: _turns[index]),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(12),
          child: Row(
            children: [
              IconButton(
                onPressed: _isBusy ? null : _resetChat,
                icon: const Icon(Icons.delete_outline),
                tooltip: 'Clear history',
              ),
              Expanded(
                child: TextField(
                  controller: _inputController,
                  minLines: 1,
                  maxLines: 3,
                  onSubmitted: (_) => _send(),
                  decoration: const InputDecoration(
                    hintText: 'Ask a follow-up…',
                    border: OutlineInputBorder(),
                  ),
                ),
              ),
              const SizedBox(width: 8),
              IconButton.filled(
                onPressed: _isBusy ? null : _send,
                icon: const Icon(Icons.send),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _TurnBubble extends StatelessWidget {
  final _Turn turn;

  const _TurnBubble({required this.turn});

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: turn.fromUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: turn.fromUser ? Colors.blue.shade100 : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (turn.thinking.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  turn.thinking,
                  style: TextStyle(
                    fontSize: 12,
                    fontStyle: FontStyle.italic,
                    color: Colors.grey.shade600,
                  ),
                ),
              ),
            SelectableText(turn.text),
          ],
        ),
      ),
    );
  }
}
