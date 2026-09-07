// lib/src/rag/rag_pipeline.dart

import '../core/llm_interface.dart';
import '../core/streaming_result.dart';
import 'chunking/text_chunker.dart';
import 'document/document.dart';
import 'embeddings/embedding_provider.dart';
import 'vector_store/vector_store.dart';

// ── Progress models ───────────────────────────────────────────────────────────

/// Progress of document ingestion (chunking + embedding).
class RagIngestionProgress {
  /// Total number of chunks to be embedded
  final int totalChunks;

  /// Number of chunks already embedded
  final int embeddedChunks;

  /// Preview of the text of the chunk currently being processed (max 60 chars)
  final String currentPreview;

  /// Whether ingestion has completed
  final bool isComplete;

  const RagIngestionProgress({
    required this.totalChunks,
    required this.embeddedChunks,
    required this.currentPreview,
    this.isComplete = false,
  });

  /// Completion fraction (0.0–1.0)
  double get fraction => totalChunks > 0 ? embeddedChunks / totalChunks : 0.0;

  @override
  String toString() =>
      'RagIngestionProgress($embeddedChunks/$totalChunks, '
      '${(fraction * 100).toStringAsFixed(0)}%)';
}

// ── RagPipeline ───────────────────────────────────────────────────────────────

/// Orchestrator for the RAG (Retrieval-Augmented Generation) pipeline.
///
/// ## DI pattern
///
/// Accepts via constructor:
/// - [EmbeddingProvider] — vector generation for chunks and queries
/// - [VectorStore] — vector storage and search
/// - [LlmPlugin] — response generation based on context
/// - [TextChunker] — document splitting into chunks (optional)
///
/// ## Ingestion pipeline
///
/// ```
/// Document → TextChunker → chunks → EmbeddingProvider.embed() → VectorStore.addChunks()
/// ```
///
/// ## Query pipeline
///
/// ```
/// question → embed() → VectorStore.search() → build prompt → LlmPlugin.sendPromptStream()
/// ```
///
/// ## Usage example
///
/// ```dart
/// final pipeline = RagPipeline(
///   embeddingProvider: LlamaEmbeddingProvider(),
///   vectorStore: InMemoryVectorStore(autoSavePath: '/path/to/index.json'),
///   generationPlugin: LlmPlugin.localGGUF(modelPath: '/path/to/llama.gguf'),
/// );
///
/// // Initialization
/// await pipeline.embeddingProvider.initialize({'modelPath': '/path/to/embed.gguf'});
/// await pipeline.generationPlugin.initialize();
///
/// // Document ingestion
/// final doc = Document.fromText(text, source: 'file.txt');
/// await for (final progress in pipeline.ingestDocument(doc)) {
///   print('${progress.embeddedChunks}/${progress.totalChunks}');
/// }
///
/// // RAG query
/// await for (final chunk in pipeline.query('What is RAG?')) {
///   print(chunk.text);
/// }
/// ```
class RagPipeline {
  final EmbeddingProvider embeddingProvider;
  final VectorStore vectorStore;
  final LlmInterface generationPlugin;
  final TextChunker chunker;
  final String promptTemplate;

  /// Instructions sent as a real `system` message, so the model's chat
  /// template places them where it expects instructions rather than treating
  /// them as part of the user's question.
  final String systemPrompt;

  /// Default RAG instructions, sent as the system message.
  static const String defaultSystemPrompt =
      'Answer the question using only the context the user provides. '
      'If the answer cannot be derived from that context, say so explicitly — '
      'do not make up information.';

  /// Default RAG prompt template with `{context}` and `{question}` placeholders.
  ///
  /// Context is built from the `topK` best-matching chunks,
  /// separated by `---`.
  static const String defaultPromptTemplate =
      'CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:';

  RagPipeline({
    required this.embeddingProvider,
    required this.vectorStore,
    required this.generationPlugin,
    TextChunker? chunker,
    String? promptTemplate,
    String? systemPrompt,
  }) : chunker = chunker ?? const TextChunker(),
       promptTemplate = promptTemplate ?? defaultPromptTemplate,
       systemPrompt = systemPrompt ?? defaultSystemPrompt;

  // ── Ingestion ─────────────────────────────────────────────────────────────

  /// Splits [document] into chunks, generates embeddings, and saves to [vectorStore].
  ///
  /// Returns a stream of [RagIngestionProgress] — progress can be tracked in the UI:
  ///
  /// ```dart
  /// await for (final p in pipeline.ingestDocument(doc)) {
  ///   setState(() => progress = p.fraction);
  /// }
  /// ```
  ///
  /// After completion the last event has [RagIngestionProgress.isComplete] = true.
  Stream<RagIngestionProgress> ingestDocument(Document document) async* {
    if (!embeddingProvider.isInitialized) {
      throw StateError(
        'EmbeddingProvider is not initialized. '
        'Call embeddingProvider.initialize() before ingestion.',
      );
    }

    // Step 1: split into chunks
    final chunks = chunker.chunk(document);
    if (chunks.isEmpty) return;

    yield RagIngestionProgress(
      totalChunks: chunks.length,
      embeddedChunks: 0,
      currentPreview: 'Splitting into ${chunks.length} chunks...',
    );

    // Step 2: embed each chunk
    for (int i = 0; i < chunks.length; i++) {
      final chunk = chunks[i];
      chunk.embedding = await embeddingProvider.embed(chunk.text);

      yield RagIngestionProgress(
        totalChunks: chunks.length,
        embeddedChunks: i + 1,
        currentPreview: chunk.text.length > 60
            ? chunk.text.substring(0, 60)
            : chunk.text,
      );
    }

    // Step 3: save to VectorStore (with automatic persist if configured)
    await vectorStore.addChunks(chunks);

    yield RagIngestionProgress(
      totalChunks: chunks.length,
      embeddedChunks: chunks.length,
      currentPreview: 'Saved ${chunks.length} chunks to the index.',
      isComplete: true,
    );
  }

  // ── Query ─────────────────────────────────────────────────────────────────

  /// Retrieves relevant chunks and generates a response via [generationPlugin].
  ///
  /// Returns [Stream<StreamingChunk>] — response tokens appear as they are generated.
  ///
  /// [topK] — number of context chunks (default 5)
  /// [minSimilarity] — minimum similarity to include in context (0.0–1.0)
  ///
  /// Throws [StateError] if no documents are indexed.
  Stream<StreamingChunk> query(
    String question, {
    int topK = 5,
    double minSimilarity = 0.25,
  }) async* {
    if (!embeddingProvider.isInitialized) {
      throw StateError('EmbeddingProvider is not initialized.');
    }
    if (!generationPlugin.isInitialized) {
      throw StateError('GenerationPlugin is not initialized.');
    }
    if (vectorStore.indexedSize == 0) {
      throw StateError(
        'Vector store is empty. Add documents via ingestDocument().',
      );
    }

    // Step 1: embed the query
    final queryEmbedding = await embeddingProvider.embed(question);

    // Step 2: find relevant chunks
    final results = await vectorStore.search(
      queryEmbedding,
      topK: topK,
      minSimilarity: minSimilarity,
    );

    if (results.isEmpty) {
      // Fallback: respond without context, inform the user
      yield* generationPlugin.sendPromptStream(
        'No relevant information was found in the knowledge base for the question: "$question". '
        'Inform the user that the knowledge base does not contain relevant data.',
        systemPrompt: systemPrompt,
      );
      return;
    }

    // Step 3: build prompt with context
    final contextParts = results
        .map((r) {
          final source =
              r.chunk.metadata['documentTitle'] ?? r.chunk.documentId;
          return '[$source, similarity: ${(r.similarity * 100).toStringAsFixed(0)}%]\n${r.chunk.text}';
        })
        .join('\n\n---\n\n');

    final augmentedPrompt = promptTemplate
        .replaceAll('{context}', contextParts)
        .replaceAll('{question}', question);

    // Step 4: generate streaming response
    yield* generationPlugin.sendPromptStream(
      augmentedPrompt,
      systemPrompt: systemPrompt,
    );
  }

  /// Retrieves relevant chunks for [question] without generating a response.
  ///
  /// Useful for inspection — checking what the RAG found in the store.
  Future<List<VectorSearchResult>> findRelevant(
    String question, {
    int topK = 5,
    double minSimilarity = 0.0,
  }) async {
    if (!embeddingProvider.isInitialized) {
      throw StateError('EmbeddingProvider is not initialized.');
    }
    final queryEmbedding = await embeddingProvider.embed(question);
    return vectorStore.search(
      queryEmbedding,
      topK: topK,
      minSimilarity: minSimilarity,
    );
  }

  // ── Persistence ───────────────────────────────────────────────────────────

  /// Loads the vector index from a JSON file and sets the auto-save path.
  Future<void> loadIndex(String path) => vectorStore.load(path);

  // ── Lifecycle ─────────────────────────────────────────────────────────────

  /// Releases the embedding provider and generation model resources.
  ///
  /// Does not release VectorStore — in-memory data is preserved until end of session
  /// (and persisted to disk if autoSavePath was set).
  Future<void> dispose() async {
    await embeddingProvider.dispose();
    await generationPlugin.dispose();
  }
}
