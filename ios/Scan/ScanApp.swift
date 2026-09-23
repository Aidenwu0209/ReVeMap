import SwiftUI

enum AppTab: Hashable {
    case capture
    case records
}

@main
struct ScanApp: App {
    @StateObject private var capture = CaptureController()
    @StateObject private var airGrab = AirGrabController()
    @Environment(\.scenePhase) private var scenePhase
    @AppStorage("receiverHost") private var receiverHost = ""
    @AppStorage("receiverPort") private var receiverPort = 7001
    @State private var handledLaunchArguments = false
    @State private var tab: AppTab = .capture
    @AppStorage("appLang") private var lang = "zh"

    var body: some Scene {
        WindowGroup {
            TabView(selection: $tab) {
                NavigationStack {
                    CaptureView(capture: capture)
                }
                .tabItem { Label(L10n.t("扫描", "Scan"), systemImage: "record.circle") }
                .tag(AppTab.capture)

                SessionsView()
                    .tabItem { Label(L10n.t("记录", "Records"), systemImage: "clock.arrow.circlepath") }
                    .tag(AppTab.records)
            }
            .environmentObject(capture)
            .environmentObject(airGrab)
            .environment(\.scanRecordsVisible, tab == .records)
            .onChange(of: tab) { _, newTab in if newTab != .records { airGrab.stop() } }
            .onChange(of: scenePhase) { _, phase in if phase != .active { airGrab.stop() } }
            .environment(\.locale, Locale(identifier: lang == "en" ? "en" : "zh-Hans"))
            .preferredColorScheme(.dark)
            .task {
                guard !handledLaunchArguments else { return }
                handledLaunchArguments = true
                let arguments = ProcessInfo.processInfo.arguments
                if arguments.contains("--open-sessions") {
                    tab = .records
                }
                guard arguments.contains("--autostart") else { return }
                let host = receiverHost.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !host.isEmpty, let port = UInt16(exactly: receiverPort) else {
                    return
                }
                capture.start(host: host, port: port)
            }
        }
    }
}
