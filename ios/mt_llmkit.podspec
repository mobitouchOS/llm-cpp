#
# To learn more about a Podspec see http://guides.cocoapods.org/syntax/podspec.html.
# Run `pod lib lint mt_llmkit.podspec` to validate before publishing.
#
Pod::Spec.new do |s|
  s.name             = 'mt_llmkit'
  s.version          = '0.0.1'
  s.summary          = 'Run Large Language Models locally on iOS with Flutter.'
  s.description      = <<-DESC
mt_llmkit enables running Large Language Models locally on iOS using llama.cpp. This package provides real-time streaming inference, performance metrics, cloud AI chat providers, and a fully local RAG pipeline — all from Flutter.
                       DESC
  s.homepage         = 'https://github.com/mobitouchOS/mt_llmkit'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Mobitouch' => 'mobitouch.net@gmail.com' }
  s.source           = { :path => '.' }
  # Shared with the Swift Package Manager layout in ios/mt_llmkit/.
  s.source_files = 'mt_llmkit/Sources/mt_llmkit/**/*.swift'
  s.dependency 'Flutter'
  # Deliberate guard: the prebuilt `libllamadart.dylib` that llamadart bundles as a
  # native asset is built with `minos 16.4`, so an app with a lower deployment target
  # would ship a framework it cannot load. CocoaPods rejects the integration instead.
  s.platform = :ios, '16.4'

  # Flutter.framework does not contain a i386 slice.
  s.pod_target_xcconfig = { 'DEFINES_MODULE' => 'YES', 'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386' }
  s.swift_version = '5.0'

  # Privacy manifest — the Swift Package Manager equivalent is `resources:` in
  # ios/mt_llmkit/Package.swift. See
  # https://developer.apple.com/documentation/bundleresources/privacy_manifest_files
  s.resource_bundles = {
    'mt_llmkit_privacy' => ['mt_llmkit/Sources/mt_llmkit/Resources/PrivacyInfo.xcprivacy']
  }
end
