// example/lib/rag_page.dart
//
// Full RAG pipeline demonstration:
//
//  ┌─────────────────────────────────────────────────────────────────────┐
//  │  1. Models: generation model + embedding model (separate GGUF)      │
//  │  2. Documents: PDF/TXT → text extraction → TextChunker → chunks     │
//  │  3. Ingestion: chunks → LlamaEmbeddingProvider → InMemoryVectorStore │
//  │  4. Query: query → embed → search → augment prompt → stream         │
//  └─────────────────────────────────────────────────────────────────────┘

import 'dart:async';
import 'dart:io';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:mt_llmkit/mt_llmkit.dart';
import 'package:path_provider/path_provider.dart';
import 'package:read_pdf_text/read_pdf_text.dart';

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

const _generationModel = _ModelSpec(
  name: 'Llama-3.2-3B-Instruct Q4_K_M',
  url:
      'https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
  filename: 'rag.gguf',
  description: 'Generation model (~2GB)',
);

const _embeddingModel = _ModelSpec(
  name: 'nomic-embed-text-v1.5 Q4_K_M',
  url:
      'https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf',
  filename: 'embed_model.gguf',
  description: 'Embedding model (~270MB)',
);

// ── Document list item ────────────────────────────────────────────────────

class _IndexedDoc {
  final Document document;
  final int chunkCount;
  _IndexedDoc(this.document, this.chunkCount);
}

// ── RAG page ──────────────────────────────────────────────────────────────

class RagPage extends StatefulWidget {
  const RagPage({super.key});

  @override
  State<RagPage> createState() => _RagPageState();
}

class _RagPageState extends State<RagPage> {
  // ── State: models ──────────────────────────────────────────────────────
  String? _generationModelPath;
  String? _embeddingModelPath;
  bool _generationModelReady = false;
  bool _embeddingModelReady = false;
  bool _isDownloadingGen = false;
  bool _isDownloadingEmbed = false;
  double _downloadProgressGen = 0;
  double _downloadProgressEmbed = 0;

  // ── State: pipeline ────────────────────────────────────────────────────
  RagEngine? _rag;
  bool get _pipelineReady => _rag?.isReady ?? false;
  bool _hasIndexFile = false;

  // ── State: documents ───────────────────────────────────────────────────
  final List<_IndexedDoc> _indexedDocs = [];
  bool _isIngesting = false;
  RagIngestionProgress? _ingestionProgress;

  // ── State: query ───────────────────────────────────────────────────────
  bool _isQuerying = false;
  String _answer = '';
  List<VectorSearchResult> _sources = [];
  String _metricsText = '';
  StreamSubscription<StreamingChunk>? _querySubscription;

  // ── Controllers ────────────────────────────────────────────────────────
  final _queryController = TextEditingController();
  final _scrollController = ScrollController();

  @override
  void initState() {
    super.initState();
    _checkExistingModels();
  }

  @override
  void dispose() {
    _querySubscription?.cancel();
    // Widget teardown is synchronous; let the engine release itself.
    unawaited(_rag?.dispose() ?? Future<void>.value());
    _queryController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  // ── Check for existing models ──────────────────────────────────────────

  Future<void> _checkExistingModels() async {
    final dir = await getApplicationDocumentsDirectory();
    final genFile = File('${dir.path}/${_generationModel.filename}');
    final embedFile = File('${dir.path}/${_embeddingModel.filename}');
    final indexFile = File('${dir.path}/rag_index.json');

    setState(() {
      if (genFile.existsSync()) {
        _generationModelPath = genFile.path;
        _generationModelReady = true;
      }
      if (embedFile.existsSync()) {
        _embeddingModelPath = embedFile.path;
        _embeddingModelReady = true;
      }
      _hasIndexFile = indexFile.existsSync();
    });
  }

  Future<void> _deleteIndexFile() async {
    final dir = await getApplicationDocumentsDirectory();
    final indexFile = File('${dir.path}/rag_index.json');
    if (indexFile.existsSync()) {
      await indexFile.delete();
    }
    setState(() {
      _hasIndexFile = false;
      _indexedDocs.clear();
    });
    _showSnack('Index file deleted');
  }

  // ── Model downloads ────────────────────────────────────────────────────

  Future<void> _downloadModel(_ModelSpec spec) async {
    final isGen = spec == _generationModel;
    setState(() {
      if (isGen) {
        _isDownloadingGen = true;
        _downloadProgressGen = 0;
      } else {
        _isDownloadingEmbed = true;
        _downloadProgressEmbed = 0;
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
              if (isGen) {
                _downloadProgressGen = progress;
              } else {
                _downloadProgressEmbed = progress;
              }
            });
          }
        }
        await sink.close();

        setState(() {
          if (isGen) {
            _generationModelPath = filePath;
            _generationModelReady = true;
            _isDownloadingGen = false;
          } else {
            _embeddingModelPath = filePath;
            _embeddingModelReady = true;
            _isDownloadingEmbed = false;
          }
        });
      } else {
        throw HttpException('HTTP ${response.statusCode}');
      }
    } catch (e) {
      setState(() {
        if (isGen) {
          _isDownloadingGen = false;
        } else {
          _isDownloadingEmbed = false;
        }
      });
      _showError('Download error for ${spec.name}: $e');
    }
  }

  // ── Pipeline initialization ────────────────────────────────────────────

  Future<void> _initializePipeline() async {
    if (!_generationModelReady || !_embeddingModelReady) return;

    final dir = await getApplicationDocumentsDirectory();
    final rag = RagEngine(
      genModelPath: _generationModelPath!,
      embedModelPath: _embeddingModelPath!,
      indexPath: '${dir.path}/rag_index.json',
      genConfig: const LlmConfig(
        temp: 0.3,
        topP: 0.85,
        topK: 30,
        // nBatch: 1024,
        nBatch: 32,
        penaltyRepeat: 1.15,
        nCtx: 4096,
        nGpuLayers: 6,
        nThreads: 4,
        nPredict: 512,
        gpuBackend: GpuBackend.auto,
      ),
      chunker: const TextChunker(chunkSize: 600, chunkOverlap: 100),
    );

    try {
      await rag.initialize();
      setState(() => _rag = rag);
      if (rag.indexedSize > 0) {
        _showSnack('Index loaded: ${rag.indexedSize} chunks');
      }
    } catch (e) {
      await rag.dispose();
      _showError('Pipeline initialization error: $e');
    }
  }

  // ── Document ingestion ─────────────────────────────────────────────────

  Future<void> _pickAndIngestFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'txt'],
      allowMultiple: false,
    );

    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    final path = file.path;
    if (path == null) return;

    final extension = file.extension?.toLowerCase();
    late final String content;

    try {
      content = extension == 'pdf'
          ? await _extractPdfText(path)
          : await File(path).readAsString();
    } catch (e) {
      _showError('File read error: $e');
      return;
    }

    if (content.trim().isEmpty) {
      _showError('File is empty or text cannot be extracted.');
      return;
    }

    final document = extension == 'pdf'
        ? Document.fromPdf(content, source: path)
        : Document.fromText(content, source: path);

    await _ingestDocument(document);
  }

  /// PDF text extraction using read_pdf_text.
  Future<String> _extractPdfText(String pdfPath) async {
    final text = await ReadPdfText.getPDFtextPaginated(pdfPath);

    final buffer = StringBuffer();
    for (int i = 0; i < text.length; i++) {
      if (text[i].isNotEmpty) {
        buffer.writeln(text[i]);
      }
    }

    return buffer.toString();
  }

  /// Document ingestion: chunk → embed → store.
  ///
  /// The [RagEngine.ingestDocument] stream reports progress after each
  /// embedded chunk — we update the UI in real time.
  Future<void> _ingestDocument(Document document) async {
    if (_rag == null) return;

    setState(() {
      _isIngesting = true;
      _ingestionProgress = null;
    });

    int chunkCount = 0;
    try {
      await for (final progress in _rag!.ingestDocument(document)) {
        setState(() {
          _ingestionProgress = progress;
          chunkCount = progress.embeddedChunks;
        });
      }

      setState(() {
        _indexedDocs.add(_IndexedDoc(document, chunkCount));
        _isIngesting = false;
        _ingestionProgress = null;
      });

      _showSnack('Indexed "${document.title}" ($chunkCount chunks)');
    } catch (e) {
      setState(() => _isIngesting = false);
      _showError('Ingestion error: $e');
    }
  }

  Future<void> _removeDocument(_IndexedDoc doc) async {
    await _rag!.removeDocument(doc.document.id);
    setState(() => _indexedDocs.remove(doc));
    _showSnack('Removed "${doc.document.title}" from the index');
  }

  Future<void> _clearIndex() async {
    if (_rag == null) return;
    await _rag!.clearIndex();
    setState(() => _indexedDocs.clear());
    _showSnack('Index cleared');
  }

  // ── RAG query ──────────────────────────────────────────────────────────

  /// Queries the RAG pipeline and displays the streaming response with sources.
  ///
  /// Flow:
  /// 1. embed(query) → queryEmbedding
  /// 2. vectorStore.search(queryEmbedding) → relevantChunks
  /// 3. buildPrompt(context, question) → augmentedPrompt
  /// 4. generationPlugin.sendPromptStream(augmentedPrompt) → `Stream<StreamingChunk>`
  Future<void> _sendQuery() async {
    final question = _queryController.text.trim();
    if (question.isEmpty || _rag == null || _isQuerying) return;

    await _querySubscription?.cancel();

    setState(() {
      _isQuerying = true;
      _answer = '';
      _sources = [];
      _metricsText = '';
    });

    try {
      // Retrieve relevant chunks (for display as sources)
      _sources = await _rag!.findRelevant(
        question,
        topK: 4,
        minSimilarity: 0.25,
      );
      setState(() {});

      // Streaming RAG response
      final answerBuffer = StringBuffer();
      _querySubscription = _rag!
          .query(question, topK: 4, minSimilarity: 0.25)
          .listen(
            (chunk) {
              setState(() {
                if (chunk.text.isNotEmpty) {
                  answerBuffer.write(chunk.text);
                  _answer = answerBuffer.toString();
                }
                if (chunk.metrics != null) {
                  final m = chunk.metrics!;
                  _metricsText =
                      '${m.tokensGenerated} tokens │ '
                      '${m.tokensPerSecond.toStringAsFixed(1)} t/s │ '
                      '${(m.durationMs / 1000).toStringAsFixed(1)}s';
                }
                if (chunk.isFinal) {
                  _isQuerying = false;
                }
              });

              WidgetsBinding.instance.addPostFrameCallback((_) {
                // if (_scrollController.hasClients) {
                //   _scrollController.animateTo(
                //     _scrollController.position.maxScrollExtent,
                //     duration: const Duration(milliseconds: 100),
                //     curve: Curves.easeOut,
                //   );
                // }
              });
            },
            onError: (Object error) {
              setState(() {
                _answer = 'Error: $error';
                _isQuerying = false;
              });
            },
            onDone: () {
              if (mounted) setState(() => _isQuerying = false);
            },
          );
    } catch (e) {
      setState(() {
        _answer = 'Error: $e';
        _isQuerying = false;
      });
    }
  }

  Future<void> _stopQuery() async {
    await _querySubscription?.cancel();
    setState(() => _isQuerying = false);
  }

  // ── UI helpers ─────────────────────────────────────────────────────────

  void _showError(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), backgroundColor: Colors.red.shade700),
    );
  }

  void _showSnack(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
  }

  // ── Build ──────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SingleChildScrollView(
        controller: _scrollController,
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── Section: Models ─────────────────────────────────────────
            _SectionHeader(title: '1. Models', icon: Icons.memory),
            _buildModelRow(
              _generationModel,
              _generationModelReady,
              _isDownloadingGen,
              _downloadProgressGen,
            ),
            const SizedBox(height: 8),
            _buildModelRow(
              _embeddingModel,
              _embeddingModelReady,
              _isDownloadingEmbed,
              _downloadProgressEmbed,
            ),
            const SizedBox(height: 8),

            if (_generationModelReady &&
                _embeddingModelReady &&
                !_pipelineReady)
              Column(
                children: [
                  ElevatedButton.icon(
                    onPressed: _initializePipeline,
                    icon: const Icon(Icons.play_arrow),
                    label: const Text('Initialize RAG Pipeline'),
                  ),
                  if (_hasIndexFile) ...[
                    const SizedBox(width: 8),
                    OutlinedButton.icon(
                      onPressed: _deleteIndexFile,
                      icon: const Icon(Icons.delete_forever, color: Colors.red),
                      label: const Text(
                        'Delete index',
                        style: TextStyle(color: Colors.red),
                      ),
                    ),
                  ],
                ],
              ),

            if (_pipelineReady)
              const _StatusChip(label: 'Pipeline ready', color: Colors.green),

            const SizedBox(height: 20),

            // ── Section: Documents ──────────────────────────────────────
            _SectionHeader(title: '2. Knowledge Base', icon: Icons.folder_open),

            if (!_pipelineReady)
              const Text(
                'Initialize the pipeline first.',
                style: TextStyle(color: Colors.grey),
              ),

            if (_pipelineReady) ...[
              Row(
                children: [
                  ElevatedButton.icon(
                    onPressed: _isIngesting ? null : _pickAndIngestFile,
                    icon: const Icon(Icons.attach_file),
                    label: const Text('Add file (PDF / TXT)'),
                  ),
                  const SizedBox(width: 8),
                  if (_indexedDocs.isNotEmpty)
                    OutlinedButton.icon(
                      onPressed: _isIngesting ? null : _clearIndex,
                      icon: const Icon(Icons.delete_sweep, color: Colors.red),
                      label: const Text(
                        'Clear index',
                        style: TextStyle(color: Colors.red),
                      ),
                    ),
                ],
              ),

              // Ingestion progress bar
              if (_isIngesting && _ingestionProgress != null) ...[
                const SizedBox(height: 12),
                _IngestionProgressBar(progress: _ingestionProgress!),
              ],

              // List of indexed documents
              if (_indexedDocs.isNotEmpty) ...[
                const SizedBox(height: 12),
                ...List.generate(_indexedDocs.length, (i) {
                  final doc = _indexedDocs[i];
                  return _DocumentTile(
                    doc: doc,
                    onRemove: () => _removeDocument(doc),
                  );
                }),
              ],
            ],

            const SizedBox(height: 20),

            // ── Section: RAG Query ───────────────────────────────────────
            _SectionHeader(title: '3. Ask', icon: Icons.question_answer),

            if (_pipelineReady && _rag!.indexedSize > 0) ...[
              Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _queryController,
                      maxLines: 2,
                      minLines: 1,
                      decoration: const InputDecoration(
                        hintText: 'Type a question about the documents...',
                        border: OutlineInputBorder(),
                      ),
                      onSubmitted: (_) => _sendQuery(),
                    ),
                  ),
                  const SizedBox(width: 8),
                  if (_isQuerying)
                    IconButton(
                      icon: const Icon(Icons.stop_circle),
                      onPressed: _stopQuery,
                      tooltip: 'Stop',
                    )
                  else
                    IconButton.filled(
                      icon: const Icon(Icons.send),
                      onPressed: _sendQuery,
                      tooltip: 'Send query',
                    ),
                ],
              ),
              const SizedBox(height: 12),

              // Metrics
              if (_metricsText.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
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

              // Answer
              if (_answer.isNotEmpty) ...[
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
                            'Answer',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          if (_isQuerying) ...[
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
                        _answer,
                        style: Theme.of(context).textTheme.bodyMedium,
                      ),
                    ],
                  ),
                ),
              ],

              // Sources (relevant chunks)
              if (_sources.isNotEmpty) ...[
                const SizedBox(height: 12),
                const Text(
                  'Sources (relevant excerpts):',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13),
                ),
                const SizedBox(height: 6),
                ..._sources.map((r) => _SourceChip(result: r)),
              ],
            ] else if (_pipelineReady) ...[
              const Text(
                'Add documents to the knowledge base to be able to ask questions.',
                style: TextStyle(color: Colors.grey),
              ),
            ],
          ],
        ),
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

// ── Helper widgets ─────────────────────────────────────────────────────────

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

class _DocumentTile extends StatelessWidget {
  final _IndexedDoc doc;
  final VoidCallback onRemove;
  const _DocumentTile({required this.doc, required this.onRemove});

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      child: ListTile(
        dense: true,
        leading: Icon(
          doc.document.type == 'pdf'
              ? Icons.picture_as_pdf
              : Icons.text_snippet,
          color: doc.document.type == 'pdf' ? Colors.red : Colors.blue,
        ),
        title: Text(doc.document.title, style: const TextStyle(fontSize: 13)),
        subtitle: Text(
          '${doc.chunkCount} chunks • ${doc.document.contentLength} chars',
          style: const TextStyle(fontSize: 11),
        ),
        trailing: IconButton(
          icon: const Icon(Icons.delete_outline, size: 20),
          onPressed: onRemove,
          tooltip: 'Remove from index',
        ),
      ),
    );
  }
}

class _IngestionProgressBar extends StatelessWidget {
  final RagIngestionProgress progress;
  const _IngestionProgressBar({required this.progress});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Embedding ${progress.embeddedChunks}/${progress.totalChunks} chunks...',
          style: const TextStyle(fontSize: 12),
        ),
        const SizedBox(height: 4),
        LinearProgressIndicator(value: progress.fraction),
        if (progress.currentPreview.isNotEmpty)
          Padding(
            padding: const EdgeInsets.only(top: 4),
            child: Text(
              '"${progress.currentPreview}..."',
              style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
            ),
          ),
      ],
    );
  }
}

class _SourceChip extends StatelessWidget {
  final VectorSearchResult result;
  const _SourceChip({required this.result});

  @override
  Widget build(BuildContext context) {
    final pct = (result.similarity * 100).toStringAsFixed(0);
    final title =
        result.chunk.metadata['documentTitle'] as String? ??
        result.chunk.documentId;

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.green.shade50,
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: Colors.green.shade200),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.format_quote, size: 14, color: Colors.green.shade700),
              const SizedBox(width: 4),
              Flexible(
                child: Text(
                  '#${result.rank} • $title • $pct% match',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Colors.green.shade700,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            result.chunk.text.length > 300
                ? '${result.chunk.text.substring(0, 300)}…'
                : result.chunk.text,
            style: const TextStyle(fontSize: 11),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}
