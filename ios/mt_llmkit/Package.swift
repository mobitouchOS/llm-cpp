// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "mt_llmkit",
    platforms: [
        // llamadart's prebuilt llama.cpp runtime for Apple requires iOS 16.4.
        .iOS("16.4")
    ],
    products: [
        .library(name: "mt-llmkit", targets: ["mt_llmkit"])
    ],
    dependencies: [
        .package(name: "FlutterFramework", path: "../FlutterFramework")
    ],
    targets: [
        .target(
            name: "mt_llmkit",
            dependencies: [
                .product(name: "FlutterFramework", package: "FlutterFramework")
            ],
            resources: [
                .process("Resources")
            ]
        )
    ]
)
