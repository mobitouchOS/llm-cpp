# Migracja MediVault (`~/medvalt`) z `llmcpp` na `mt_llmkit`

Dokument dla aplikacji `mt_starter` / MediVault. Opisuje przejście z git-dependency
`llmcpp` (→ llamadart 0.6.10) na `mt_llmkit` (→ llamadart 0.8.22).

Stan na dziś: drzewo robocze medvalt jest **czyste**, branch `feature/background-service`.
Niezacommitowane zmiany, o których była mowa, były w repo `mt_llmkit` — są już na branchu
`feature/phase-2`.

---

## 1. Co się realnie zmienia

Warstwa Dart `llmcpp` i **wyjściowego** `mt_llmkit` była identyczna (poza nazwą pakietu).
Po zmianach z `feature/phase-2` **już nie jest** — jest kilka zmian łamiących API, wszystkie
mechaniczne, plus jeden realny bloker infrastrukturalny (SwiftPM).

| Obszar | Było (`llmcpp`) | Jest (`mt_llmkit` @ `feature/phase-2`) |
|---|---|---|
| Pakiet | `package:llmcpp/llmcpp.dart` | `package:mt_llmkit/mt_llmkit.dart` |
| `dispose()` | `void` | `Future<void>`, **musi być awaitowane** |
| `RagEngine.dispose()` | `void` | `Future<void>` |
| Załączniki w `sendPrompt*` | `images: List<LlamaImageContent>` | `attachments: List<LlamaContentPart>` |
| System prompt | brak | `sendPrompt*(…, systemPrompt: …)` |
| Domyślne `nGpuLayers`/`nBatch`/`nThreads` | 64 / 4096 / 6 | auto llamadart |
| iOS | CocoaPods, native assets | **SwiftPM wymagany** + `llamadart_llama_cpp_flutter` |

Zmiana domyślnych **nie dotyczy medvalt**: aplikacja podaje `nCtx`, `nBatch`, `nGpuLayers`,
`nThreads`, `numberOfThreadsBatch` jawnie w obu miejscach
(`lib/data/llm_local/llm_local_datasource.dart:387`, `lib/domain/chat/services/rag_service.dart:56`).

Załączniki też nie dotyczą medvalt — aplikacja nie używa `images:` w żadnym wywołaniu
(ścieżka jest czysto tekstowa; `ModelInfo.mmprojDownloadUrl` jest zeskafoldowane, ale
`isVision => false`).

## 2. Faza 0 — odblokowanie zależności (po stronie mt_llmkit)

Branch `feature/phase-2` zawiera migrację SwiftPM (w tym wcześniej untracked
`ios/mt_llmkit/Package.swift`) oraz fazy A1–A3. Przed migracją medvalt:

1. Zmerguj `feature/phase-2` i otaguj (np. `v0.0.1-beta.2`) — medvalt ma pinować się do
   niezmiennego refa, nie do brancha.
2. Zweryfikuj, że przykład buduje się poprawnie:
   ```bash
   cd mt_llmkit && flutter analyze && flutter test
   cd example && flutter clean && flutter pub get && flutter build ios --release --no-codesign
   ls build/ios/iphoneos/Runner.app/Frameworks   # brak llamadart.framework
   ```
   Jeśli przykład nie potrafi zbudować się bez `llamadart.framework`, medvalt też nie będzie.

Do czasu tagu jedyną działającą opcją jest lokalny override:

```yaml
dependency_overrides:
  mt_llmkit:
    path: ../LLM/mt_llmkit
```

— lokalnie, **nie do merge'a**. Katalog checkoutu musi się nazywać dokładnie `mt_llmkit`
(tożsamość lokalnego pakietu SwiftPM = nazwa katalogu).

## 3. Faza 1 — pubspec i kod Dart (jeden atomowy commit)

### 3.1 `pubspec.yaml`

- Usuń blok `llmcpp:` **w całości**. Nie zostawiaj obu pluginów: oba rejestrują iOS-ową klasę
  o tej samej nazwie `LlmcppPlugin`, więc wygenerowany registrant się nie skompiluje.
- Dodaj:
  ```yaml
  mt_llmkit:
    git:
      url: https://github.com/mobitouchOS/mt_llmkit.git
      ref: v0.0.1-beta.2
  llamadart_llama_cpp_flutter: ^0.0.17
  ```
- Przestaw `flutter: config: enable-swift-package-manager` z `false` na `true`.

### 3.2 Importy — trzy pliki

`package:llmcpp/llmcpp.dart` → `package:mt_llmkit/mt_llmkit.dart` w:

- `lib/data/llm_local/llm_local_datasource.dart:6`
- `lib/data/llm_cloud/llm_cloud_datasource.dart:3`
- `lib/domain/chat/services/rag_service.dart:2`

### 3.3 `dispose()` — teraz asynchroniczne

Cztery miejsca:

- `lib/data/llm_local/llm_local_datasource.dart:222` — w `loadModel`, przed załadowaniem
  nowego modelu: `await _model.dispose();`
- `lib/data/llm_local/llm_local_datasource.dart:273-275` — `void dispose()` →
  `Future<void> dispose() async { … await _model.dispose(); }`
- `lib/domain/chat/services/rag_service.dart:220` — `void dispose() => _ragEngine?.dispose();`
  → `Future<void> dispose() async => _ragEngine?.dispose();`
- Wywołujący te `dispose()` (m.in. `model_lifecycle_service.dart`) — dodać `await`.

**Bonus:** `_isolateTeardownGrace` (150 ms `Future.delayed` po `_model.dispose()`,
`llm_local_datasource.dart:223`) można usunąć. Obchodził dokładnie ten wyścig, który
`mt_llmkit` zamyka teraz handshakiem z workerem — plugin czeka na potwierdzenie, że llama.cpp
zwolnił natywne uchwyty, zanim zabije isolate. Usuń dopiero po smoke-teście dwukrotnej zmiany
modelu pod rząd (patrz checklista, pkt 7).

### 3.4 Regeneracja i weryfikacja

```bash
dart run build_runner build --delete-conflicting-outputs
flutter analyze && flutter test
```

`rag_service.g.dart`, `llm_local_datasource.g.dart`, `llm_cloud_datasource.g.dart` to providery
Riverpod — wymagają regeneracji po zmianie sygnatur.

## 4. Faza 2 — iOS / przełączenie na SwiftPM

To jest **obowiązkowe**, nie opcjonalne. `llamadart_llama_cpp_flutter` **nie ma podspeca** —
ma wyłącznie `Package.swift` (iOS 16.4, binaryTarget `leehack/llamadart-native@v0.3.0`,
`-reexport_framework llama`). `llamadart/hook/build.dart` czyta pubspec konsumenta i dopiero
wtedy emituje asset jako `LookupInProcess()`. Bez obu tych rzeczy wracamy na ścieżkę
native-assets → `llamadart.framework` z `MinimumOSVersion 13.0` przy binarce 16.4 →
odrzucenie w App Store jako **ITMS-90208**.

Flip SwiftPM **nie usuwa CocoaPods** — pody Firebase / ML Kit działają dalej obok. Deployment
target medvalt (16.6, jednolicie w `Podfile` i `project.pbxproj`) z zapasem spełnia próg 16.4.

```bash
flutter clean
rm -rf ios/Pods ios/Podfile.lock ios/.symlinks build
flutter pub get
cd ios && pod deintegrate && pod install --repo-update
cd .. && flutter build ios --release --flavor prod --no-codesign
```

Pierwszy build ściąga XCFramework z GitHub Releases (potrzebna sieć, trwa). Zacommituj:

- `ios/Runner.xcodeproj/project.pbxproj` (Flutter wstrzyknie referencje SwiftPM — dziś jest
  ich zero)
- `ios/Runner.xcworkspace/xcshareddata/swiftpm/Package.resolved`

`ios/Podfile` zostaw bez zmian — `post_install` podnoszący target do 16.6 dotyczy wyłącznie
podów i nie sięga pakietów SwiftPM (te deklarują 16.4 same).

Android: **bez zmian**. `mt_llmkit/android/build.gradle` ma te same `abiFilters`
(`arm64-v8a`, `x86_64`) co `llmcpp`, a `minSdk` medvalt (26) jest wyższy niż wymagany 24.

## 5. Faza 3 — weryfikacja zachowania i co warto włączyć

Kompilacyjnie migracja jest bezpieczna. Ryzyko siedzi w **szablonach czatu i reasoningu**
llamadart 0.8.22.

### 5.1 Podwójne owijanie promptu

medvalt ręcznie składa prompty ChatML / Llama-3 / Gemma (`PromptFormat` w
`ai_model_notifier.dart`) i podaje je do `sendPrompt`, które i tak owija je w wiadomość
`user` i przepuszcza przez wbudowany szablon Jinja modelu. To istnieje **już dziś** — nie jest
regresją migracji — ale 0.8.x dołożyło nowy renderer szablonów, parser PEG i
`thinkingForcedOpen`, więc to miejsce numer jeden do smoke-testu.

Docelowo warto to uprościć: `mt_llmkit` ma teraz `systemPrompt`, więc instrukcje systemowe
mogą iść osobnym kanałem zamiast być wklejane w ręcznie budowany prompt. To osobne zadanie,
nie część migracji.

### 5.2 Reasoning

Modele, których blok `<think>` wcześniej wyciekał do `delta.content`, w 0.8.x trafiają do
`delta.thinking`. `mt_llmkit` **już go nie gubi** — wystawia jako `StreamingChunk.thinking`.
Dwie możliwości do wykorzystania:

- `chunk.thinking` można podpiąć pod istniejący label „myślenia” w UI
  (commit `96a6950`) zamiast heurystyki na czasie.
- `LlmConfig.enableThinking: false` (albo `thinkingBudget`) zastępuje sufiks `/no_think`
  z `chat_notifier.dart:635` — działa niezależnie od tego, czy model rozumie `/no_think`.

### 5.3 Ucięte odpowiedzi

`chunk.isTruncated` / `chunk.finishReason` mówi wprost, czy generacja skończyła się czysto,
czy wyczerpała budżet tokenów. `chat_generation_service.dart` ma na to własną heurystykę
`_endedPrematurely()` — teraz może pytać wprost.

### 5.4 GPU

Katalog modeli ustawia `gpuLayers: 0` dla prawie wszystkich pozycji
(`ai_model_notifier.dart:226/247/258`), a Android wymusza `GpuBackend.cpu` — GPU jest dziś
faktycznie wyłączone. `LocalModel.diagnostics()` pokazuje teraz, który backend realnie wygrał
i ile warstw naprawdę trafiło na GPU, więc można to zrewidować na twardych danych zamiast
zgadywać po tokenach/s.

Uwaga na `nGpuLayers: -1` w `rag_service.dart:60` — llamadart oczekuje `0` (CPU) albo dodatniej
liczby warstw (`ModelParams.maxGpuLayers` = 999 dla pełnego offloadu). `-1` to prawdopodobnie
zaszłość; warto zamienić na `0` lub `999` zależnie od intencji.

### 5.5 Pamięć

`cacheTypeK` / `cacheTypeV` = `KvCacheType.q8_0` (z `flashAttention` w `auto`) połowi pamięć
KV cache. Przy `nCtx: 8192` na telefonie to największa pojedyncza oszczędność, jaka jest
dostępna. Warto przetestować na Bieliku 4.5B.

## 6. Checklista weryfikacyjna

**Dart**
- `grep llmcpp pubspec.lock` — pusto; w lockfile `llamadart 0.8.22` i
  `llamadart_llama_cpp_flutter 0.0.17`.
- `flutter analyze` — 0 issues. `flutter test` — zielone.

**iOS (po `flutter clean`, build release)**
- `ls build/ios/iphoneos/Runner.app/Frameworks` zawiera `llama.framework` i
  `llamadart-llama-cpp-flutter.framework`, a **nie zawiera** `llamadart.framework`.
  Twarda bramka: `test ! -d build/ios/iphoneos/Runner.app/Frameworks/llamadart.framework`
- `/usr/libexec/PlistBuddy -c "Print :MinimumOSVersion" .../llama.framework/Info.plist` → `16.4`
- `vtool -show-build .../llama.framework/llama | grep minos` → `16.4`
- `flutter build ipa --flavor prod` + upload do TestFlight bez ITMS-90208.

**Android**
- `flutter build apk --flavor prod --release`
- `unzip -l build/app/outputs/flutter-apk/*.apk | grep '\.so$'` → wyłącznie `lib/arm64-v8a/`
  i `lib/x86_64/`, z obecnym `libllama*.so`.

**Runtime — po jednym przejściu na model z katalogu** (Qwen ChatML, Bielik llama3, Gemma)
1. Pobranie i załadowanie modelu; `isModelLoaded()` = true.
2. Streaming produkuje tekst bez wycieku `<|im_end|>` / `<|eot_id|>` / `<end_of_turn>`.
3. Model z `supportsNoThink`: brak bloku `<think>` w widocznej odpowiedzi i brak długiego
   „zawieszenia” strumienia.
4. Label „myślenia” / hint czasu nadal działa.
5. **Prefill/wznowienie**: start długiej odpowiedzi → zbackgroundowanie → powrót; odpowiedź
   kontynuuje się spójnie, nie restartuje i nie powtarza. Potem force-kill w trakcie i
   ponowne otwarcie → `_loadPendingJob()` wznawia z zapisanego fragmentu.
6. RAG: zapis wyniku badania (indeksacja) → pytanie → `RagContext.retrievedChunks` niepuste →
   odpowiedź cytuje dane. Potem `rebuildIndex`.
7. Dwukrotna zmiana modelu pod rząd — walidacja, że handshake przy `dispose()` wystarcza i
   `_isolateTeardownGrace` można usunąć (brak crasha „Cannot invoke native callback from a
   different isolate”).
8. Providery chmurowe (OpenAI / Gemini / Claude / Mistral) + `validateApiKey`.

## 7. Ryzyka, wg wagi

1. **Flip SwiftPM.** Przepisuje `project.pbxproj`, wymaga cyklu `pod deintegrate` /
   `pod install` i pobrania binarnego XCFramework. Rollback = revert `pubspec.yaml`,
   `project.pbxproj` i `Package.resolved`, potem `flutter clean` + `pod install`.
2. **Szablony i reasoning pod 0.8.22** na ręcznie budowanych, podwójnie owijanych promptach
   medvalt oraz na prefillu tury asystenta. Bezpieczne kompilacyjnie, testowalne wyłącznie na
   urządzeniu z realnymi GGUF-ami.
3. **Tag `mt_llmkit`.** Pinuj do tagu, nie do brancha — repo jest w fazie `0.0.1-beta`.
