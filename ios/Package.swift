// swift-tools-version: 5.9
import PackageDescription

// Exercise the production networking/recovery code on macOS without a camera or workstation.
let package = Package(
    name: "ScanCore",
    platforms: [.macOS(.v13)],
    products: [.library(name: "ScanCore", targets: ["ScanCore"])],
    targets: [
        .target(name: "ScanCore", path: "Scan",
                exclude: ["AirGrabController.swift", "AirGrabGesture.swift", "CaptureController.swift",
                          "CaptureView.swift", "L10n.swift", "ScanApp.swift", "SessionsView.swift",
                          "Info.plist", "Contents.json", "Assets.xcassets"],
                sources: ["WorkstationSettings.swift", "LocalRecorder.swift", "SessionDetailStore.swift", "ScanLocalization.swift"]),
        .testTarget(name: "ScanCoreTests", dependencies: ["ScanCore"], path: "Tests")
    ]
)
