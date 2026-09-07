// lib/src/rag/rag_engine.dart

import '../core/llm_config.dart';
import '../core/streaming_result.dart';
import 'chunking/text_chunker.dart';
import 'document/document.dart';
import 'llama_rag_coordinator.dart';
import 'rag_pipeline.dart';
import 'vector_store/in_memory_vector_store.dart';
import 'vector_store/vector_store.dart';

/// Facade over [LlamaRagCoordinator] + [InMemoryVectorStore] + [RagPipeline].
///
/// Lifecycle:
///   1. Construct with parameters
///   2. Call [initialize()] — async, loads both models in a single isolate
///   3. Use: [ingestDocument], [query], [findRelevant], [removeDocument], [clearIndex]
///   4. Call [dispose()] when done — idempotent, safe to call twice
///
/// ## Critical architecture note
///
/// Internally delegates to [LlamaRagCoordinator], which loads both the
/// embedding model and the generation model inside ONE worker isolate.
/// This avoids the fatal "Cannot invoke native callback from a different
/// isolate" crash caused by the global `llama_log_set` pointer in llama.cpp.
///
/// ## Persistence
///
/// When [indexPath] is provided, [InMemoryVectorStore] auto-saves the JSON
/// index after every [ingestDocument] / [removeDocument] / [clearIndex] call,
/// and loads the existing index during [initialize()].
/// When [indexPath] is null, the store is in-memory only (no persistence).
///
/// ## Example
///
/// ```dart
/// final rag = RagEngine(
///   genModelPath: '/path/to/llama.gguf',
///   embedModelPath: '/path/to/nomic-embed.gguf',
///   indexPath: '${dir.path}/rag_index.json',
///   genConfig: const LlmConfig(temp: 0.3, nCtx: 4096, nGpuLayers: 4),
/// );
/// await rag.initialize();
///
/// await for (final p in rag.ingestDocument(doc)) {
///   print('${p.embeddedChunks}/${p.totalChunks}');
/// }
///
/// await for (final chunk in rag.query('What is RAG?')) {
///   print(chunk.text);
/// }
///
/// await rag.dispose();
/// ```
class RagEngine {
  // ── Construction parameters ──────────────────────────────────────────────

  final String genModelPath;
  final String embedModelPath;

  /// Path for automatic JSON persistence of the vector index.
  /// When null the store is in-memory only.
  final String? indexPath;

  final LlmConfig genConfig;
  final int embedNCtx;
  final TextChunker chunker;
  final String? promptTemplate;

  /// Overrides the RAG system message. See [RagPipeline.defaultSystemPrompt].
  final String? systemPrompt;

  // ── Internal components (null until initialize() completes) ──────────────

  LlamaRagCoordinator? _coordinator;
  InMemoryVectorStore? _vectorStore;
  RagPipeline? _pipeline;
  bool _isReady = false;

  // ── Constructor ──────────────────────────────────────────────────────────

  RagEngine({
    required this.genModelPath,
    required this.embedModelPath,
    this.indexPath,
    this.genConfig = const LlmConfig(),
    this.embedNCtx = 512,
    this.chunker = const TextChunker(),
    this.promptTemplate,
    this.systemPrompt,
  });

  // ── Lifecycle ────────────────────────────────────────────────────────────

  /// Loads both models in a single worker isolate and restores the persisted
  /// index (if [indexPath] was provided and the file exists).
  ///
  /// Subsequent calls are no-ops (idempotent).
  Future<void> initialize() async {
    if (_isReady) return;

    final coordinator = await LlamaRagCoordinator.create(
      embedModelPath: embedModelPath,
      genModelPath: genModelPath,
      genConfig: genConfig,
      embedNCtx: embedNCtx,
    );

    final vectorStore = InMemoryVectorStore(autoSavePath: indexPath);
    if (indexPath != null) await vectorStore.load(indexPath!);

    final pipeline = RagPipeline(
      embeddingProvider: coordinator.embeddingProvider,
      vectorStore: vectorStore,
      generationPlugin: coordinator.generationPlugin,
      chunker: chunker,
      promptTemplate: promptTemplate,
      systemPrompt: systemPrompt,
    );

    _coordinator = coordinator;
    _vectorStore = vectorStore;
    _pipeline = pipeline;
    _isReady = true;
  }

  /// Releases the worker isolate and both loaded models.
  ///
  /// Awaiting this matters: the worker frees llama.cpp handles for both the
  /// embedding and the generation model during teardown.
  /// Safe to call multiple times (idempotent).
  Future<void> dispose() async {
    if (!_isReady) return;
    final coordinator = _coordinator;
    _coordinator = null;
    await coordinator?.dispose();
    _vectorStore = null;
    _pipeline = null;
    _isReady = false;
  }

  // ── State inspection ─────────────────────────────────────────────────────

  /// Whether [initialize()] has completed successfully.
  bool get isReady => _isReady;

  /// Number of chunks with a generated embedding (available for search).
  /// Returns 0 before [initialize()] is called.
  int get indexedSize => _vectorStore?.indexedSize ?? 0;

  /// Unique document identifiers present in the vector store.
  /// Returns an empty list before [initialize()] is called.
  List<String> get documentIds => _vectorStore?.documentIds ?? const [];

  /// Direct access to the underlying [InMemoryVectorStore].
  /// Returns null before [initialize()] is called.
  InMemoryVectorStore? get vectorStore => _vectorStore;

  // ── Ingestion ────────────────────────────────────────────────────────────

  /// Splits [document] into chunks, embeds each chunk, and saves to the store.
  ///
  /// Streams [RagIngestionProgress] events for UI progress updates.
  /// Throws [StateError] if [isReady] is false.
  Stream<RagIngestionProgress> ingestDocument(Document document) {
    _checkReady();
    return _pipeline!.ingestDocument(document);
  }

  /// Removes all chunks belonging to [documentId] from the store.
  /// Throws [StateError] if [isReady] is false.
  Future<void> removeDocument(String documentId) {
    _checkReady();
    return _vectorStore!.removeDocument(documentId);
  }

  /// Clears all chunks from the store (and deletes the index file if present).
  /// Throws [StateError] if [isReady] is false.
  Future<void> clearIndex() {
    _checkReady();
    return _vectorStore!.clear();
  }

  // ── Query ────────────────────────────────────────────────────────────────

  /// Retrieves relevant chunks and streams a generated answer.
  ///
  /// [topK] — number of context chunks (default 5)
  /// [minSimilarity] — minimum cosine similarity 0.0–1.0 (default 0.25)
  ///
  /// Throws [StateError] if [isReady] is false.
  Stream<StreamingChunk> query(
    String question, {
    int topK = 5,
    double minSimilarity = 0.25,
  }) {
    _checkReady();
    return _pipeline!.query(question, topK: topK, minSimilarity: minSimilarity);
  }

  /// Returns the top-K most relevant chunks without generating an answer.
  /// Throws [StateError] if [isReady] is false.
  Future<List<VectorSearchResult>> findRelevant(
    String question, {
    int topK = 5,
    double minSimilarity = 0.0,
  }) {
    _checkReady();
    return _pipeline!.findRelevant(
      question,
      topK: topK,
      minSimilarity: minSimilarity,
    );
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  void _checkReady() {
    if (!_isReady) {
      throw StateError(
        'RagEngine is not initialized. Call initialize() first.',
      );
    }
  }
}
