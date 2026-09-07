// example/lib/main.dart

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:path_provider/path_provider.dart';

import 'rag_page.dart';
import 'rest_api_tab.dart';
import 'vision_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'llmcpp Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const MainPage(),
    );
  }
}

// ── Main navigation ──────────────────────────────────────────────────────────

class MainPage extends StatefulWidget {
  const MainPage({super.key});

  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  int _selectedIndex = 0;

  static const _pages = [LlmDemoPage(), VisionPage(), RagPage()];

  static const _labels = [
    NavigationDestination(
      icon: Icon(Icons.chat_outlined),
      selectedIcon: Icon(Icons.chat),
      label: 'LLM',
    ),
    NavigationDestination(
      icon: Icon(Icons.image_outlined),
      selectedIcon: Icon(Icons.image),
      label: 'Vision',
    ),
    NavigationDestination(
      icon: Icon(Icons.search_outlined),
      selectedIcon: Icon(Icons.search),
      label: 'RAG',
    ),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(switch (_selectedIndex) {
          0 => 'llmcpp — LLM Demo',
          1 => 'llmcpp — Vision Demo',
          _ => 'llmcpp — RAG Demo',
        }),
      ),
      body: IndexedStack(index: _selectedIndex, children: _pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (i) => setState(() => _selectedIndex = i),
        destinations: _labels,
      ),
    );
  }
}

// ── LLM page ──────────────────────────────────────────────────────────────────

enum ProviderType { localGGUF, restApi }

class LlmDemoPage extends StatefulWidget {
  const LlmDemoPage({super.key});

  @override
  State<LlmDemoPage> createState() => _LlmDemoPageState();
}

class _LlmDemoPageState extends State<LlmDemoPage> {
  LocalModel? _ggufPlugin;
  StreamSubscription<StreamingChunk>? _streamSubscription;

  ProviderType _selectedProvider = ProviderType.localGGUF;
  String? _modelPath;
  bool _isDownloading = false;
  double _downloadProgress = 0;
  bool _isModelReady = false;
  bool _isGenerating = false;

  final StringBuffer _outputBuffer = StringBuffer();
  String _output = '';
  String _metricsText = '';
  String _statusMessage = '';

  final _promptController = TextEditingController(
    text: 'What is the capital of Poland? Describe it in 2 sentences.',
  );
  final _scrollController = ScrollController();

  static const _modelUrl =
      'https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf';
  // 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf';

  @override
  void initState() {
    super.initState();
    _checkModelExists();
  }

  @override
  void dispose() {
    _streamSubscription?.cancel();
    // Widget teardown is synchronous; let the model release itself.
    unawaited(_ggufPlugin?.dispose() ?? Future<void>.value());
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  Future<void> _checkModelExists() async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/model.gguf');
    if (file.existsSync()) {
      setState(() {
        _modelPath = file.path;
        _isModelReady = true;
        _statusMessage = 'Model ready: Llama-3.2-1B-Instruct Q4_K_M';
      });
    }
  }

  Future<void> _downloadModel() async {
    setState(() {
      _isDownloading = true;
      _downloadProgress = 0;
      _statusMessage = 'Downloading model...';
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      final filePath = '${dir.path}/model.gguf';
      final httpClient = HttpClient();
      final request = await httpClient.getUrl(Uri.parse(_modelUrl));
      final response = await request.close();

      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        final file = File(filePath);
        final sink = file.openWrite();
        int downloadedBytes = 0;

        await for (final chunk in response) {
          sink.add(chunk);
          downloadedBytes += chunk.length;
          if (contentLength > 0) {
            setState(() {
              _downloadProgress = downloadedBytes / contentLength;
              _statusMessage =
                  'Downloading: ${(_downloadProgress * 100).toStringAsFixed(1)}%';
            });
          }
        }

        await sink.close();
        setState(() {
          _modelPath = filePath;
          _isModelReady = true;
          _isDownloading = false;
          _statusMessage = 'Model downloaded successfully';
        });
      } else {
        throw HttpException('HTTP ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        _isDownloading = false;
        _statusMessage = '';
      });
      _showError('Download error: $e');
    }
  }

  Future<bool> _initializeGguf() async {
    final previous = _ggufPlugin;
    _ggufPlugin = null;
    await previous?.dispose();

    try {
      _ggufPlugin = LocalModel(
        backend: ModelBackend.inProcess,
        config: LlmConfig(
          // topP: 0.1,
          // temp: 0.3,
          temp: 0,
          topK: 1,
          topP: 1,
          penaltyRepeat: 1.0,
          //
          nPredict: 2048,
          nCtx: 2048,
          gpuBackend: GpuBackend.cpu,
          nGpuLayers: 0,
          nBatch: 512,
          // nBatch: 128,
          nThreads: 6,
          // nCtx: 1024,
        ),
      );
      await _ggufPlugin!.loadModel(_modelPath!);
      return true;
    } catch (e) {
      _showError('Initialization error: $e');
      _ggufPlugin = null;
      return false;
    }
  }

  Future<void> _sendPrompt() async {
    if (_isGenerating) return;
    setState(() {
      _isGenerating = true;
      _output = '';
      _metricsText = '';
      _statusMessage = 'Initializing...';
      _outputBuffer.clear();
    });

    await _sendGgufPrompt();
  }

  Future<void> _sendGgufPrompt() async {
    if (_ggufPlugin == null || !_ggufPlugin!.isInitialized) {
      final success = await _initializeGguf();
      if (!success) {
        setState(() => _isGenerating = false);
        return;
      }
    }

    setState(() => _statusMessage = 'Generating...');

    _streamSubscription = _ggufPlugin!
        .sendPromptStream(_promptController.text)
        .listen(
          (chunk) {
            setState(() {
              if (chunk.text.isNotEmpty) {
                _outputBuffer.write(chunk.text);
                _output = _outputBuffer.toString();
              }
              if (chunk.metrics != null) {
                final m = chunk.metrics!;
                _metricsText =
                    'Tokens: ${m.tokensGenerated} │ ${m.tokensPerSecond.toStringAsFixed(1)} t/s │ ${m.msPerToken.toStringAsFixed(0)} ms/token';
              }
              if (chunk.isFinal) {
                _isGenerating = false;
                _statusMessage = 'Done';
              }
            });
          },
          onError: (Object error) {
            setState(() {
              _output = 'Error: $error';
              _isGenerating = false;
              _statusMessage = 'Error';
            });
          },
          onDone: () {
            if (mounted) {
              setState(() {
                _isGenerating = false;
                if (_statusMessage == 'Generating...') _statusMessage = 'Done';
              });
            }
          },
        );
  }

  void _showError(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), backgroundColor: Colors.red.shade700),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // ── Provider selector ─────────────────────────────────────────────────
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
          child: SegmentedButton<ProviderType>(
            segments: const [
              ButtonSegment(
                value: ProviderType.localGGUF,
                label: Text('Local GGUF'),
                icon: Icon(Icons.computer),
              ),
              ButtonSegment(
                value: ProviderType.restApi,
                label: Text('Rest API'),
                icon: Icon(Icons.cloud_outlined),
              ),
            ],
            selected: {_selectedProvider},
            onSelectionChanged: (selected) {
              final previous = _ggufPlugin;
              setState(() {
                _selectedProvider = selected.first;
                _ggufPlugin = null;
              });
              unawaited(previous?.dispose() ?? Future<void>.value());
            },
          ),
        ),

        // ── Rest API tab — fully self-contained ───────────────────────────────
        if (_selectedProvider == ProviderType.restApi)
          const Expanded(child: RestApiTab())
        // ── Local GGUF tab ────────────────────────────────────────────────────
        else ...[
          _buildLocalGGUFSection(),
          const Divider(height: 1),
          if (_isModelReady)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
              child: TextField(
                controller: _promptController,
                maxLines: 3,
                minLines: 1,
                decoration: const InputDecoration(
                  labelText: 'Prompt',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
          if (_metricsText.isNotEmpty)
            Container(
              color: Colors.blue.shade50,
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
              child: Text(
                _metricsText,
                style: TextStyle(
                  fontSize: 11,
                  color: Colors.blue.shade700,
                  fontFamily: 'monospace',
                ),
              ),
            ),
          Expanded(
            child: SingleChildScrollView(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (_output.isNotEmpty)
                    SelectableText(
                      _output,
                      style: Theme.of(context).textTheme.bodyLarge,
                    )
                  else if (!_isGenerating && _isModelReady)
                    Text(
                      'Press ▶ to generate...',
                      style: TextStyle(color: Colors.grey.shade400),
                    ),
                  if (_isGenerating)
                    const Padding(
                      padding: EdgeInsets.only(top: 12),
                      child: SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      ),
                    ),
                ],
              ),
            ),
          ),
          if (_statusMessage.isNotEmpty)
            Container(
              color: Colors.grey.shade100,
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Text(
                _statusMessage,
                style: const TextStyle(fontSize: 11, color: Colors.black54),
              ),
            ),
        ],
      ],
    );
  }

  Widget _buildLocalGGUFSection() {
    if (_isModelReady) {
      return Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
        child: Row(
          children: [
            Icon(Icons.check_circle, color: Colors.green.shade600, size: 18),
            const SizedBox(width: 8),
            const Expanded(
              child: Text(
                'Llama-3.2-1B-Instruct Q4_K_M',
                style: TextStyle(fontSize: 13),
              ),
            ),
            if (_isGenerating)
              IconButton(
                icon: const Icon(Icons.stop_circle_outlined),
                onPressed: () async {
                  await _streamSubscription?.cancel();
                  setState(() => _isGenerating = false);
                },
              )
            else
              IconButton(icon: const Icon(Icons.send), onPressed: _sendPrompt),
          ],
        ),
      );
    }
    if (_isDownloading) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Downloading: ${(_downloadProgress * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 13),
            ),
            const SizedBox(height: 6),
            LinearProgressIndicator(value: _downloadProgress),
          ],
        ),
      );
    }
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
      child: ElevatedButton.icon(
        onPressed: _downloadModel,
        icon: const Icon(Icons.download),
        label: const Text('Download GGUF model (~800MB)'),
      ),
    );
  }
}
