// example/lib/vision_page.dart
//
// Demonstrates LocalModel vision support:
//
//  1. Download vision model + mmproj (Gemma 3 4B)
//  2. Pick an image from the device
//  3. Enter a prompt with <image> placeholder
//  4. Stream the model's response

import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:path_provider/path_provider.dart';

// ── Model download specs ───────────────────────────────────────────────────

class _ModelSpec {
  final String name;
  final String url;
  final String filename;
  final String description;

  const _ModelSpec({
    required this.name,
    required this.url,
    required this.filename,
    required this.description,
  });
}

const _visionModel = _ModelSpec(
  name: 'Gemma 3 4B IT Q4_K_M',
  url:
      'https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf',
  filename: 'vision_model.gguf',
  description: 'Vision model (~2.5 GB)',
);

const _mmprojSpec = _ModelSpec(
  name: 'Gemma 3 4B mmproj F16',
  url:
      'https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf',
  filename: 'vision_mmproj_f16.gguf',
  description: 'Multimodal projector (~600 MB)',
);

// ── Vision page ────────────────────────────────────────────────────────────

class VisionPage extends StatefulWidget {
  const VisionPage({super.key});

  @override
  State<VisionPage> createState() => _VisionPageState();
}

class _VisionPageState extends State<VisionPage> {
  // ── State: models ──────────────────────────────────────────────────────
  String? _modelPath;
  String? _mmprojPath;
  bool _modelReady = false;
  bool _mmprojReady = false;
  bool _isDownloadingModel = false;
  bool _isDownloadingMmproj = false;
  double _downloadProgressModel = 0;
  double _downloadProgressMmproj = 0;

  bool get _modelsReady => _modelReady && _mmprojReady;

  // ── State: image ───────────────────────────────────────────────────────
  File? _selectedImage;

  // ── State: generation ─────────────────────────────────────────────────
  LocalModel? _plugin;
  StreamSubscription<StreamingChunk>? _subscription;
  bool _isGenerating = false;
  String _output = '';
  String _metricsText = '';

  final _promptController = TextEditingController(
    text: 'Describe what you see in this image in detail. <image>',
  );
  final _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _checkExistingModels();
  }

  @override
  void dispose() {
    _subscription?.cancel();
    // Widget teardown is synchronous; let the model release itself.
    unawaited(_plugin?.dispose() ?? Future<void>.value());
    _promptController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ── Check for existing models ──────────────────────────────────────────

  Future<void> _checkExistingModels() async {
    final dir = await getApplicationDocumentsDirectory();
    final modelFile = File('${dir.path}/${_visionModel.filename}');
    final mmprojFile = File('${dir.path}/${_mmprojSpec.filename}');

    setState(() {
      if (modelFile.existsSync()) {
        _modelPath = modelFile.path;
        _modelReady = true;
      }
      if (mmprojFile.existsSync()) {
        _mmprojPath = mmprojFile.path;
        _mmprojReady = true;
      }
    });
  }

  // ── Model downloads ────────────────────────────────────────────────────

  Future<void> _downloadModel(_ModelSpec spec) async {
    final isMainModel = spec == _visionModel;
    setState(() {
      if (isMainModel) {
        _isDownloadingModel = true;
        _downloadProgressModel = 0;
      } else {
        _isDownloadingMmproj = true;
        _downloadProgressMmproj = 0;
      }
    });

    try {
      final dir = await getApplicationDocumentsDirectory();
      final filePath = '${dir.path}/${spec.filename}';
      final httpClient = HttpClient();
      final request = await httpClient.getUrl(Uri.parse(spec.url));
      final response = await request.close();

      if (response.statusCode == 200) {
        final contentLength = response.contentLength;
        final sink = File(filePath).openWrite();
        int downloaded = 0;

        await for (final chunk in response) {
          sink.add(chunk);
          downloaded += chunk.length;
          if (contentLength > 0) {
            final progress = downloaded / contentLength;
            setState(() {
              if (isMainModel) {
                _downloadProgressModel = progress;
              } else {
                _downloadProgressMmproj = progress;
              }
            });
          }
        }
        await sink.close();

        // Invalidate the loaded plugin when a model changes — awaited so the
        // old model's native handles are freed before the next load.
        final previous = _plugin;
        _plugin = null;
        await previous?.dispose();

        setState(() {
          if (isMainModel) {
            _modelPath = filePath;
            _modelReady = true;
            _isDownloadingModel = false;
          } else {
            _mmprojPath = filePath;
            _mmprojReady = true;
            _isDownloadingMmproj = false;
          }
        });
      } else {
        throw HttpException('HTTP ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        if (isMainModel) {
          _isDownloadingModel = false;
        } else {
          _isDownloadingMmproj = false;
        }
      });
      _showError('Download error for ${spec.name}: $e');
    }
  }

  // ── Image selection ────────────────────────────────────────────────────

  Future<void> _pickImage() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
    );
    if (result == null || result.files.isEmpty) return;
    final path = result.files.first.path;
    if (path == null) return;
    setState(() => _selectedImage = File(path));
  }

  // ── Plugin initialization ──────────────────────────────────────────────

  Future<bool> _ensurePlugin() async {
    if (_plugin != null && _plugin!.isInitialized) return true;
    final previous = _plugin;
    _plugin = null;
    await previous?.dispose();
    try {
      _plugin = LocalModel(
        config: LlmConfig(
          mmprojPath: _mmprojPath,
          nGpuLayers: 4,
          nCtx: 4096,
          nPredict: 512,
          temp: 0.3,
          gpuBackend: GpuBackend.opencl,
        ),
      );
      await _plugin!.loadModel(_modelPath!);
      return true;
    } catch (e) {
      _plugin = null;
      _showError('Failed to load model: $e');
      return false;
    }
  }

  // ── Generation ─────────────────────────────────────────────────────────

  Future<void> _generate() async {
    if (_isGenerating) return;
    if (_selectedImage == null) {
      _showError('Please select an image first.');
      return;
    }
    final prompt = _promptController.text.trim();
    if (!prompt.contains('<image>')) {
      _showError('Prompt must contain the <image> placeholder.');
      return;
    }

    setState(() {
      _isGenerating = true;
      _output = '';
      _metricsText = '';
    });

    final ready = await _ensurePlugin();
    if (!ready) {
      setState(() => _isGenerating = false);
      return;
    }

    final image = LlamaImageContent(path: _selectedImage!.path);

    _subscription = _plugin!
        .sendPromptStream(prompt, images: [image])
        .listen(
          (chunk) {
            setState(() {
              if (chunk.text.isNotEmpty) _output += chunk.text;
              if (chunk.metrics != null) {
                final m = chunk.metrics!;
                _metricsText =
                    'Tokens: ${m.tokensGenerated} │ '
                    '${m.tokensPerSecond.toStringAsFixed(1)} t/s │ '
                    '${m.msPerToken.toStringAsFixed(0)} ms/token';
              }
              if (chunk.isFinal) _isGenerating = false;
            });
            WidgetsBinding.instance.addPostFrameCallback((_) {
              if (_scrollController.hasClients) {
                _scrollController.animateTo(
                  _scrollController.position.maxScrollExtent,
                  duration: const Duration(milliseconds: 80),
                  curve: Curves.easeOut,
                );
              }
            });
          },
          onError: (Object e) {
            setState(() {
              _output = 'Error: $e';
              _isGenerating = false;
            });
          },
          onDone: () {
            if (mounted) setState(() => _isGenerating = false);
          },
        );
  }

  Future<void> _stop() async {
    await _subscription?.cancel();
    setState(() => _isGenerating = false);
  }

  // ── UI helpers ─────────────────────────────────────────────────────────

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: Colors.red.shade700),
    );
  }

  // ── Build ──────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      controller: _scrollController,
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── Section 1: Models ────────────────────────────────────────
          _SectionHeader(title: '1. Models', icon: Icons.memory),
          _buildModelRow(
            _visionModel,
            _modelReady,
            _isDownloadingModel,
            _downloadProgressModel,
          ),
          const SizedBox(height: 8),
          _buildModelRow(
            _mmprojSpec,
            _mmprojReady,
            _isDownloadingMmproj,
            _downloadProgressMmproj,
          ),
          if (_modelsReady) ...[
            const SizedBox(height: 6),
            const _StatusChip(label: 'Models ready', color: Colors.green),
          ],
          const SizedBox(height: 20),

          // ── Section 2: Image ─────────────────────────────────────────
          _SectionHeader(title: '2. Image', icon: Icons.image_outlined),
          if (_selectedImage != null)
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.file(
                _selectedImage!,
                height: 200,
                fit: BoxFit.cover,
                width: double.infinity,
              ),
            )
          else
            Container(
              height: 120,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: const Center(
                child: Text(
                  'No image selected',
                  style: TextStyle(color: Colors.grey),
                ),
              ),
            ),
          const SizedBox(height: 8),
          OutlinedButton.icon(
            onPressed: _pickImage,
            icon: const Icon(Icons.photo_library_outlined),
            label: Text(_selectedImage == null ? 'Pick image' : 'Change image'),
          ),
          const SizedBox(height: 20),

          // ── Section 3: Prompt ────────────────────────────────────────
          _SectionHeader(title: '3. Prompt', icon: Icons.chat_outlined),
          const Text(
            'Use <image> as a placeholder where the image should be embedded.',
            style: TextStyle(fontSize: 12, color: Colors.grey),
          ),
          const SizedBox(height: 8),
          TextField(
            controller: _promptController,
            maxLines: 3,
            minLines: 1,
            decoration: const InputDecoration(
              labelText: 'Prompt',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              if (_isGenerating)
                FilledButton.icon(
                  onPressed: _stop,
                  icon: const Icon(Icons.stop_circle_outlined),
                  label: const Text('Stop'),
                  style: FilledButton.styleFrom(
                    backgroundColor: Colors.red.shade600,
                  ),
                )
              else
                FilledButton.icon(
                  onPressed: _modelsReady && _selectedImage != null
                      ? _generate
                      : null,
                  icon: const Icon(Icons.send),
                  label: const Text('Analyze image'),
                ),
              if (_isGenerating) ...[
                const SizedBox(width: 12),
                const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2),
                ),
              ],
            ],
          ),

          // ── Metrics ──────────────────────────────────────────────────
          if (_metricsText.isNotEmpty) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(6),
              ),
              child: Row(
                children: [
                  Icon(Icons.speed, size: 14, color: Colors.blue.shade700),
                  const SizedBox(width: 6),
                  Text(
                    _metricsText,
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.blue.shade700,
                      fontFamily: 'monospace',
                    ),
                  ),
                ],
              ),
            ),
          ],

          // ── Output ───────────────────────────────────────────────────
          if (_output.isNotEmpty) ...[
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade200),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.smart_toy, size: 16),
                      const SizedBox(width: 6),
                      const Text(
                        'Response',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      if (_isGenerating) ...[
                        const SizedBox(width: 8),
                        const SizedBox(
                          width: 12,
                          height: 12,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 8),
                  SelectableText(
                    _output,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildModelRow(
    _ModelSpec spec,
    bool isReady,
    bool isDownloading,
    double progress,
  ) {
    if (isReady) {
      return Row(
        children: [
          Icon(Icons.check_circle, color: Colors.green.shade600, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(spec.name, style: const TextStyle(fontSize: 13)),
          ),
          Text(
            spec.description,
            style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
          ),
        ],
      );
    }
    if (isDownloading) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Downloading ${spec.name}: ${(progress * 100).toStringAsFixed(1)}%',
            style: const TextStyle(fontSize: 13),
          ),
          const SizedBox(height: 4),
          LinearProgressIndicator(value: progress),
        ],
      );
    }
    return TextButton.icon(
      onPressed: () => _downloadModel(spec),
      icon: const Icon(Icons.download, size: 18),
      label: Text('Download ${spec.name}'),
    );
  }
}

// ── Helper widgets ──────────────────────────────────────────────────────────

class _SectionHeader extends StatelessWidget {
  final String title;
  final IconData icon;
  const _SectionHeader({required this.title, required this.icon});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Icon(icon, size: 20, color: Theme.of(context).colorScheme.primary),
          const SizedBox(width: 8),
          Text(
            title,
            style: Theme.of(
              context,
            ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }
}

class _StatusChip extends StatelessWidget {
  final String label;
  final Color color;
  const _StatusChip({required this.label, required this.color});

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(Icons.check, size: 16, color: color),
      label: Text(label, style: TextStyle(color: color, fontSize: 12)),
      backgroundColor: color.withValues(alpha: 0.1),
      side: BorderSide(color: color.withValues(alpha: 0.3)),
      padding: EdgeInsets.zero,
    );
  }
}
