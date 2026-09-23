import SwiftUI
import WebKit
import Combine

// MARK: - Models

struct SessionRecord: Codable, Identifiable, Hashable {
    let id: String
    let mode: String
    let status: String
    let frames: Int
    let points: Int
    var display_name: String? = nil

    var displayDate: String {
        if let display_name, !display_name.isEmpty { return display_name }
        // scan_YYYYMMDD_HHMMSS_hex
        let parts = id.split(separator: "_")
        guard parts.count >= 3 else { return id }
        return "\(parts[1]) \(parts[2])"
    }

    var statusLabel: String {
        switch status {
        case "active": return L10n.t("进行中", "Active")
        case "completed": return L10n.t("已完成", "Done")
        case "failed": return L10n.t("失败", "Failed")
        case "captured": return L10n.t("已采集", "Captured")
        case "incomplete": return L10n.t("未完成", "Incomplete")
        case "cancelled": return L10n.t("已取消", "Cancelled")
        default: return status
        }
    }

    var statusColor: Color {
        switch status {
        case "active": return .blue
        case "completed": return .green
        case "failed": return .red
        case "captured", "incomplete": return .orange
        default: return .gray
        }
    }

    var modeLabel: String {
        switch mode {
        case "ipad": return "iPad"
        case "replay": return L10n.t("回放", "Replay")
        case "demo": return L10n.t("数据集演示", "Dataset demo")
        default: return L10n.t("相机", "Camera")
        }
    }
}

struct SessionsResponse: Codable {
    let current: String?
    let sessions: [SessionRecord]
}

// MARK: - Store

@MainActor
final class SessionsStore: ObservableObject {
    @Published var sessions: [SessionRecord] = []
    @Published var current: String?
    @Published var errorText: String?
    @Published var deleting: SessionRecord?
    @Published var deletionError: String?
    @Published var isDeleting = false
    @Published var hasLoaded = false
    var host: String = ""

    private var token: String?
    private var tokenHost: String = ""
    private var listRevision = 0
    private let network: URLSession

    init(network: URLSession = .shared) { self.network = network }

    func load() async {
        guard !host.isEmpty else {
            errorText = L10n.t("未配置工作站地址：请先在「扫描」页设置接收端", "No workstation set — configure receiver in Scan tab")
            return
        }
        let requestHost = host
        let revision = listRevision
        guard let url = URL(string: "http://\(requestHost):8765/api/sessions") else { return }
        do {
            let (data, response) = try await network.data(for: URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            let decoded = try JSONDecoder().decode(SessionsResponse.self, from: data)
            guard host == requestHost, revision == listRevision else { return }
            sessions = decoded.sessions
            current = decoded.current
            errorText = nil
            hasLoaded = true
        } catch {
            guard host == requestHost, revision == listRevision else { return }
            errorText = L10n.t("无法连接工作站：", "Cannot reach workstation: ") + error.localizedDescription
        }
    }

    // Use the immutable record captured by the confirmation action, not dialog state.
    func confirmDelete(_ record: SessionRecord) async {
        guard !isDeleting else { return }
        isDeleting = true
        deletionError = nil
        defer { isDeleting = false }
        let requestHost = host
        guard !requestHost.isEmpty,
              let url = URL(string: "http://\(requestHost):8765/api/session/delete") else {
            deletionError = L10n.t("工作站地址无效", "Invalid workstation address")
            return
        }
        do {
            for attempt in 0..<2 {
                guard let token = await fetchToken(host: requestHost, refresh: attempt > 0) else {
                    deletionError = L10n.t("无法获取会话令牌，请检查工作站连接后重试", "Could not obtain session token. Check the workstation connection and retry.")
                    return
                }
                var request = URLRequest(url: url)
                request.httpMethod = "POST"
                request.setValue(token, forHTTPHeaderField: "X-Scan-Token")
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                request.httpBody = try JSONEncoder().encode(["id": record.id])
                let (data, response) = try await network.data(for: request)
                let code = (response as? HTTPURLResponse)?.statusCode ?? 0
                if code == 403, attempt == 0 { continue }
                guard code == 200 else {
                    deletionError = code == 409
                        ? L10n.t("工作站仍将这条记录标记为当前会话，暂时不能删除。", "The workstation still marks this record as its current session; it cannot be deleted yet.")
                        : L10n.t("删除失败（HTTP \(code)），请稍后重试", "Delete failed (HTTP \(code)). Please retry.")
                    return
                }
                let receipt = try JSONDecoder().decode([String: String].self, from: data)
                guard receipt["deleted"] == record.id else {
                    deletionError = L10n.t("删除回执与所选记录不一致，请刷新列表核对", "The deletion receipt does not match the selected record. Refresh to check.")
                    return
                }
                guard host == requestHost else { return }
                listRevision += 1
                sessions.removeAll { $0.id == record.id }
                if current == record.id { current = nil }
                hasLoaded = true
                await load()
                return
            }
        } catch {
            deletionError = L10n.t("未能确认删除结果，请刷新列表核对：", "Could not confirm deletion. Refresh to check: ") + error.localizedDescription
        }
    }

    private func fetchToken(host requestHost: String, refresh: Bool) async -> String? {
        if !refresh, let token, tokenHost == requestHost { return token }
        if refresh { token = nil }
        guard let url = URL(string: "http://\(requestHost):8765/") else { return nil }
        guard let (data, response) = try? await network.data(for: URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData)),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let html = String(data: data, encoding: .utf8),
              let range = html.range(of: "token='[0-9a-f]{48}'", options: .regularExpression)
        else { return nil }
        let freshToken = String(html[range].dropFirst(7).dropLast())
        token = freshToken
        tokenHost = requestHost
        return freshToken
    }
}

// MARK: - Web view

struct WebPage: UIViewRepresentable {
    let url: URL
    var language: String = "zh"
    var isFullscreen = false
    var selection: CloudObject? = nil
    var isolated = false
    var isolationRevision = 0
    var viewMode = "semantic_id"
    var airGrab: AirGrabController? = nil
    var onAirGrab: (Int) -> Void = { _ in }
    var onObjectsReady: (Bool) -> Void = { _ in }
    var onToggleFullscreen: () -> Void = {}

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(context.coordinator, name: "scanViewer")
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.isOpaque = false
        view.backgroundColor = .black
        view.scrollView.backgroundColor = .black
        view.scrollView.isScrollEnabled = false
        view.scrollView.contentInsetAdjustmentBehavior = .never
        view.scrollView.minimumZoomScale = 1
        view.scrollView.maximumZoomScale = 1
        view.scrollView.bounces = false
        view.allowsLinkPreview = false
        view.navigationDelegate = context.coordinator
        context.coordinator.update(from: self, view: view)
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {
        context.coordinator.update(from: self, view: view)
    }

    static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) {
        view.configuration.userContentController.removeScriptMessageHandler(forName: "scanViewer")
        view.navigationDelegate = nil
        coordinator.detachHandTracking()
    }

    @MainActor final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        private var loadedPage: String?
        private var language = "zh"
        private var fullscreen = false
        private var selection: CloudObject?
        private var isolated = false
        private var isolationRevision = 0
        private var viewMode = "semantic_id"
        private weak var airGrab: AirGrabController?
        private var airSubscription: AnyCancellable?
        private var lastSettings: Data?
        private var onAirGrab: (Int) -> Void = { _ in }
        private var onObjectsReady: (Bool) -> Void = { _ in }
        private var onToggleFullscreen: () -> Void = {}

        func update(from page: WebPage, view: WKWebView) {
            language = page.language
            fullscreen = page.isFullscreen
            selection = page.selection
            isolated = page.isolated
            isolationRevision = page.isolationRevision
            viewMode = page.viewMode
            onToggleFullscreen = page.onToggleFullscreen
            onAirGrab = page.onAirGrab
            onObjectsReady = page.onObjectsReady
            if airGrab !== page.airGrab {
                airSubscription?.cancel()
                airGrab = page.airGrab
                airSubscription = page.airGrab?.packets.sink { [weak view] packet in
                    guard let view, let data = try? JSONEncoder().encode(packet),
                          let json = String(data: data, encoding: .utf8) else { return }
                    view.evaluateJavaScript("window.scanViewer?.hand(\(json))", completionHandler: nil)
                }
            }
            var identity = URLComponents(url: page.url, resolvingAgainstBaseURL: false)
            identity?.queryItems?.removeAll { $0.name == "lang" }
            if loadedPage != identity?.string {
                loadedPage = identity?.string
                lastSettings = nil
                view.load(URLRequest(url: page.url))
            } else {
                synchronize(view)
            }
        }

        private func synchronize(_ view: WKWebView) {
            var settings: [String: Any] = ["language": language, "fullscreen": fullscreen,
                                           "isolated": isolated, "viewMode": viewMode,
                                           "isolationRevision": isolationRevision,
                                           "airEnabled": airGrab?.enabled ?? false,
                                           "airSession": airGrab?.sessionID ?? "",
                                           "airMode": airGrab?.mode.rawValue ?? "move",
                                           "reduceMotion": UIAccessibility.isReduceMotionEnabled]
            if let selection {
                settings["selection"] = ["kind": selection.kind, "id": selection.label_id, "title": selection.title]
            } else {
                settings["selection"] = NSNull()
            }
            guard let data = try? JSONSerialization.data(withJSONObject: settings, options: [.sortedKeys]),
                  let json = String(data: data, encoding: .utf8) else { return }
            guard data != lastSettings else { return }
            lastSettings = data
            view.evaluateJavaScript("window.scanViewer?.configure(\(json))", completionHandler: nil)
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
            lastSettings = nil
            synchronize(webView)
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            airGrab?.stop()
            onObjectsReady(false)
            lastSettings = nil
            webView.reload()
        }

        func detachHandTracking() { airSubscription?.cancel(); airSubscription = nil }

        func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
            guard message.frameInfo.isMainFrame,
                  let body = message.body as? [String: String] else { return }
            switch body["action"] {
            case "toggleFullscreen": onToggleFullscreen()
            case "airGrabbed":
                guard airGrab?.enabled == true, body["session"] == airGrab?.sessionID else { return }
                guard (body["mode"] ?? "move") == airGrab?.mode.rawValue else { return }
                guard let revision = Int(body["isolationRevision"] ?? ""), revision == isolationRevision else { return }
                airGrab?.viewerDidGrab()
                onAirGrab(revision)
            case "airTouch": airGrab?.rearm()
            case "objectsReady": onObjectsReady(true)
            default: break
            }
        }
    }
}

// MARK: - Views

struct SessionsView: View {
    @AppStorage("appLang") private var lang = "zh"
    @AppStorage("receiverHost") private var receiverHost = ""
    @StateObject private var store = SessionsStore()

    var body: some View {
        NavigationStack {
            ZStack {
                Color(uiColor: .systemGroupedBackground).ignoresSafeArea()
                Group {
                    if store.sessions.isEmpty {
                        if let error = store.errorText {
                            VStack(spacing: 12) {
                                Image(systemName: "wifi.exclamationmark")
                                    .font(.system(size: 40))
                                    .foregroundStyle(.white.opacity(0.55))
                                Text(error)
                                    .font(.footnote)
                                    .multilineTextAlignment(.center)
                                Button(L10n.t("重试", "Retry")) { Task { await store.load() } }
                            }
                            .padding()
                        } else if store.hasLoaded {
                            ContentUnavailableView(L10n.t("暂无扫描记录", "No scans yet"), systemImage: "clock.arrow.circlepath")
                        } else {
                            ProgressView(L10n.t("载入扫描记录…", "Loading scans…"))
                        }
                    } else {
                        VStack(spacing: 0) {
                            HStack {
                                LanguageToggle()
                                Spacer()
                                Button {
                                    Task { await store.load() }
                                } label: {
                                    Label(L10n.t("刷新", "Refresh"), systemImage: "arrow.clockwise")
                                        .font(.footnote)
                                        .foregroundStyle(.primary)
                                        .frame(width: 96, height: 32)
                                        .glassEffect(.regular.interactive(), in: .capsule)
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(.horizontal, 18)
                            .padding(.vertical, 8)
                            List {
                                ForEach(store.sessions) { record in
                                NavigationLink {
                                    SessionDetailView(host: store.host, record: record)
                                } label: {
                                    sessionRow(record)
                                }
                                
                                .listRowBackground(Color(uiColor: .secondarySystemGroupedBackground))
                                .listRowSeparatorTint(Color.white.opacity(0.08))
                                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                    Button {
                                        store.deleting = record
                                    } label: {
                                        Label(L10n.t("删除", "Delete"), systemImage: "trash")
                                    }
                                    .tint(.red)
                                    .disabled(store.isDeleting)
                                }
                            }
                            }
                            .listStyle(.plain)
                            .scrollContentBackground(.hidden)
                            .background(Color(uiColor: .secondarySystemGroupedBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                            .overlay {
                                RoundedRectangle(cornerRadius: 20, style: .continuous)
                                    .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
                                    .allowsHitTesting(false)
                            }
                            .padding(.horizontal, 18)
                        }
                    }
                }
            }
            .navigationTitle("")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar(.hidden, for: .navigationBar)
            .confirmationDialog(
                L10n.t("删除这次扫描？", "Delete this scan?"),
                isPresented: Binding(
                    get: { store.deleting != nil },
                    set: { if !$0 { store.deleting = nil } }
                ),
                titleVisibility: .visible,
                presenting: store.deleting
            ) { record in
                Button(L10n.t("删除「\(record.displayDate)」", "Delete \(record.displayDate)"), role: .destructive) {
                    Task { await store.confirmDelete(record) }
                }
                Button(L10n.t("取消", "Cancel"), role: .cancel) {}
            }
            .alert(L10n.t("删除未完成", "Deletion not completed"), isPresented: Binding(
                get: { store.deletionError != nil },
                set: { if !$0 { store.deletionError = nil } }
            )) {
                Button(L10n.t("知道了", "OK"), role: .cancel) { store.deletionError = nil }
            } message: {
                Text(store.deletionError ?? "")
            }
        }
        .preferredColorScheme(.dark)
        .task {
            store.host = receiverHost.trimmingCharacters(in: .whitespacesAndNewlines)
            await store.load()
        }
        .refreshable { await store.load() }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 3_000_000_000)
                guard !Task.isCancelled else { break }
                await store.load()
            }
        }
    }

    private func sessionRow(_ record: SessionRecord) -> some View {
        HStack(spacing: 12) {
            RoundedRectangle(cornerRadius: 6)
                .fill(record.statusColor)
                .frame(width: 12, height: 36)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(record.displayDate)
                        .font(.body.weight(.medium))
                        .foregroundStyle(.primary)
                    Text(record.modeLabel)
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.white.opacity(0.08), in: Capsule())
                        .foregroundStyle(.secondary)
                }
                Text(L10n.t("\(record.statusLabel) · \(record.frames) 帧 · \(record.points) 点", "\(record.statusLabel) · \(record.frames) fr · \(record.points) pts"))
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.55))
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }
}

// MARK: - Session detail (native)

struct StageTiming: Codable, Identifiable {
    let stage: String
    let seconds: Double
    var id: String { stage }
}

struct DetailCapture: Codable {
    let frames: Int?
    let elapsed_s: Double?
    let valid_depth_fraction: Double?
    let fps: Double?
    let rejected_pairs: Int?
}

struct DetailMapping: Codable {
    let stages: [StageTiming]?
    let status: String?
}

struct DetailCloud: Codable {
    let points: Int?
}

enum FuzzyValue: Codable {
    case string(String)
    case bool(Bool)
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else {
            self = .string(try container.decode(String.self))
        }
    }
    var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }
}

struct TimelineEntry: Codable, Identifiable {
    let stage: String
    let seconds: Double
    var id: String { stage }
}

struct DetailStatus: Codable {
    let status: String
    let mode: String?
    let options: [String: FuzzyValue]?
    let capture: DetailCapture?
    let mapping: DetailMapping?
    let cloud: DetailCloud?
    let view: String?
    let timeline: [TimelineEntry]?
}

struct InstanceRow: Codable, Identifiable {
    let instance_id: Int
    let semantic_name: String
    let vlm_name: String?
    let point_count: Int
    var id: Int { instance_id }
}

struct CloudObject: Codable, Identifiable, Equatable {
    let kind: String
    let label_id: Int
    let semantic_id: Int
    let semantic_name: String
    let vlm_name: String?
    let point_count: Int
    let size_m: [Double]
    var id: String { "\(kind)-\(label_id)" }
    var title: String {
        let name = (vlm_name?.isEmpty == false && vlm_name != "unknown") ? vlm_name! : semantic_name
        let localized = L10n.semanticName(name)
        return kind == "instance" ? "\(localized) · #\(label_id)" : localized
    }
    var dimensions: String { size_m.map { String(format: "%.2f", $0) }.joined(separator: " × ") + " m" }
}

struct ObjectCatalog: Codable {
    let revision: String
    let points: Int
    let rgb_available: Bool
    let classes: [CloudObject]
    let instances: [CloudObject]
}

private struct ObjectShareFile: Identifiable {
    let url: URL
    var id: URL { url }
}

private struct ObjectShareSheet: UIViewControllerRepresentable {
    let url: URL
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

@MainActor
final class SessionDetailStore: ObservableObject {
    @Published var status: DetailStatus?
    @Published var instances: [InstanceRow] = []
    @Published var objects: ObjectCatalog?
    @Published var objectError: String?
    @Published var viewMode = "semantic_id"
    @Published var busy = false
    @Published var errorText: String?
    var host = ""
    var recordID = ""
    private var token: String?

    func load() async {
        guard let url = URL(string: "http://\(host):8765/s/\(recordID)/api/status") else { return }
        guard let (data, _) = try? await URLSession.shared.data(from: url),
              let decoded = try? JSONDecoder().decode(DetailStatus.self, from: data) else {
            errorText = L10n.t("无法加载会话数据", "Failed to load session")
            return
        }
        status = decoded
        errorText = nil
        viewMode = decoded.view ?? "semantic_id"
        if let iurl = URL(string: "http://\(host):8765/s/\(recordID)/api/instances"),
           let (idata, _) = try? await URLSession.shared.data(from: iurl),
           let list = try? JSONDecoder().decode([InstanceRow].self, from: idata) {
            instances = list
        }
        await loadObjects()
    }

    func loadObjects() async {
        guard let url = URL(string: "http://\(host):8765/s/\(recordID)/api/objects") else { return }
        do {
            let (data, response) = try await URLSession.shared.data(for: URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            objects = try JSONDecoder().decode(ObjectCatalog.self, from: data)
            objectError = nil
        } catch {
            objectError = L10n.t("暂时无法加载可选取物体", "Selectable objects are unavailable")
        }
    }

    func setView(_ mode: String) async {
        if objects != nil { viewMode = mode; return }
        busy = true
        defer { busy = false }
        guard let token = await fetchToken() else { return }
        guard let url = URL(string: "http://\(host):8765/s/\(recordID)/api/view") else { return }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(token, forHTTPHeaderField: "X-Scan-Token")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(["mode": mode])
        _ = try? await URLSession.shared.data(for: request)
        viewMode = mode
    }

    private func fetchToken() async -> String? {
        if let token { return token }
        guard let url = URL(string: "http://\(host):8765/") else { return nil }
        guard let (data, _) = try? await URLSession.shared.data(from: url),
              let html = String(data: data, encoding: .utf8),
              let range = html.range(of: "token='[0-9a-f]{48}'", options: .regularExpression)
        else { return nil }
        token = String(html[range].dropFirst(7).dropLast())
        return token
    }
}

struct SessionDetailView: View {
    let host: String
    let record: SessionRecord
    @StateObject private var store = SessionDetailStore()
    @EnvironmentObject private var capture: CaptureController
    @EnvironmentObject private var airGrab: AirGrabController
    @Environment(\.scanRecordsVisible) private var recordsVisible
    @AppStorage("appLang") private var lang = "zh"
    @Environment(\.dismiss) private var dismiss
    @Environment(\.horizontalSizeClass) private var hSize
    @State private var isViewerFullscreen = false
    @State private var selectedObject: CloudObject?
    @State private var isolateObject = false
    @State private var isolationRevision = 0
    @State private var objectGrouping = "class"
    @State private var objectClassFilter: Int?
    @State private var viewerObjectsReady = false
    @State private var exportingObject = false
    @State private var shareFile: ObjectShareFile?
    @State private var exportError: String?

    var body: some View {
        Group {
            if hSize == .compact {
                VStack(spacing: 0) {
                    viewer.frame(maxWidth: .infinity, maxHeight: .infinity)
                    if !isViewerFullscreen {
                        Divider()
                        detailPanel.frame(maxWidth: .infinity)
                    }
                }
            } else {
                HStack(spacing: 0) {
                    viewer.frame(maxWidth: .infinity, maxHeight: .infinity)
                    if !isViewerFullscreen {
                        Divider()
                        detailPanel.frame(width: 380)
                    }
                }
            }
        }
        .background(Color.black)
        .ignoresSafeArea(isViewerFullscreen ? .container : [], edges: .all)
        .statusBarHidden(isViewerFullscreen)
        .toolbar(isViewerFullscreen ? .hidden : .visible, for: .navigationBar, .tabBar)
        .navigationBarBackButtonHidden(true)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .navigationBarLeading) {
                Button {
                    dismiss()
                } label: {
                    Image(systemName: "chevron.left")
                        .font(.body.weight(.semibold))
                }
                .accessibilityLabel(L10n.t("返回", "Back"))
            }
            ToolbarItem(placement: .navigationBarTrailing) {
                Picker("", selection: Binding(
                    get: { store.viewMode },
                    set: { newValue in Task { await store.setView(newValue) } }
                )) {
                    Text(L10n.t("语义", "Semantic")).tag("semantic_id")
                    Text(L10n.t("实例", "Instance")).tag("instance_id")
                    Text(L10n.t("原色", "RGB")).tag("rgb")
                        .disabled(store.objects?.rgb_available == false)
                }
                .pickerStyle(.segmented)
                .frame(width: 230)
                .disabled(store.busy)
            }
            // The segmented picker already draws its own surface.
            .sharedBackgroundVisibility(.hidden)
        }
        .onDisappear { airGrab.stop() }
        .onChange(of: recordsVisible) { _, visible in if !visible { airGrab.stop() } }
        .onChange(of: selectedObject?.id) { _, id in
            if id == nil { airGrab.stop() } else { airGrab.rearm() }
        }
        .alert(L10n.t("隔空操控", "Air Grab"), isPresented: Binding(
            get: { airGrab.error != nil }, set: { if !$0 { airGrab.error = nil } }
        )) {
            Button(L10n.t("知道了", "OK"), role: .cancel) { airGrab.error = nil }
        } message: { Text(airGrab.error ?? "") }
        .sheet(item: $shareFile) { file in
            ObjectShareSheet(url: file.url)
        }
        .alert(L10n.t("导出未完成", "Export not completed"), isPresented: Binding(
            get: { exportError != nil }, set: { if !$0 { exportError = nil } }
        )) {
            Button(L10n.t("知道了", "OK"), role: .cancel) { exportError = nil }
        } message: { Text(exportError ?? "") }
        .task {
            store.host = host
            store.recordID = record.id
            await store.load()
        }
    }

    @ViewBuilder
    private var viewer: some View {
        if let pageURL = URL(string: "http://\(host):8765/?session=\(record.id)&embed=1&lang=\(lang)") {
            WebPage(url: pageURL, language: lang, isFullscreen: isViewerFullscreen,
                    selection: selectedObject, isolated: isolateObject, isolationRevision: isolationRevision, viewMode: store.viewMode,
                    airGrab: airGrab, onAirGrab: { revision in
                        guard revision == isolationRevision else { return }
                        isolateObject = true
                        UIImpactFeedbackGenerator(style: .soft).impactOccurred()
                    }, onObjectsReady: { viewerObjectsReady = $0 }) {
                isViewerFullscreen.toggle()
            }
            .background(AirGrabOrientationReader { airGrab.updateOrientation($0) })
            .safeAreaInset(edge: .top, spacing: 0) {
                if airGrab.enabled { airGrabBanner }
            }
            .safeAreaInset(edge: .bottom, spacing: 0) {
                if let object = selectedObject { selectionBar(object) }
            }
        } else {
            Text(L10n.t("会话地址无效", "Invalid session URL"))
                .foregroundStyle(.white.opacity(0.55))
        }
    }

    private var detailPanel: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text(record.displayDate)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.55))
                if let status = store.status {
                    if record.mode == "demo" {
                        Label(L10n.t("ScanNet 数据集 · SAM3 预测结果", "ScanNet dataset · SAM3 predictions"), systemImage: "testtube.2")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if store.objects != nil { objectSection }
                    overviewSection(status)
                    if let timeline = status.timeline, timeline.count >= 2 {
                        fullTimelineSection(timeline)
                    } else if let stages = status.mapping?.stages, !stages.isEmpty {
                        timingSection(stages)
                    }
                    if store.objects == nil {
                        if let error = store.objectError, status.status == "completed" {
                            Button { Task { await store.loadObjects() } } label: {
                                Label(error + L10n.t("，点此重试", ". Tap to retry"), systemImage: "arrow.clockwise")
                                    .font(.caption)
                            }
                        }
                        instanceSection
                    }
                } else if let error = store.errorText {
                    Text(error).font(.footnote).foregroundStyle(.red)
                } else {
                    ProgressView().frame(maxWidth: .infinity)
                }
            }
            .padding(18)
        }
        .background(Color(white: 0.07))
    }

    private func overviewSection(_ status: DetailStatus) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            // ---- results ----
            VStack(alignment: .leading, spacing: 10) {
                Label(L10n.t("扫描概况", "Overview"), systemImage: "chart.bar")
                    .font(.subheadline.weight(.semibold))
                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    metricCell(L10n.t("点数", "Points"),
                               status.cloud?.points.map { String($0) } ?? "—")
                    metricCell(L10n.t("实例", "Instances"),
                               String(store.objects?.instances.count ?? store.instances.count))
                    metricCell(L10n.t("已命名", "Named"),
                               "\(namedCount)/\(store.instances.count)")
                    if let valid = status.capture?.valid_depth_fraction {
                        metricCell(L10n.t("深度有效", "Valid depth"),
                                   String(format: "%.0f%%", valid * 100))
                    }
                    metricCell(L10n.t("帧数", "Frames"),
                               status.capture?.frames.map(String.init) ?? "—")
                    metricCell(L10n.t("采集时长", "Duration"),
                               formatSeconds(status.capture?.elapsed_s))
                }
            }
            .padding(14)
            .background(Color(uiColor: .secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
            }

            // ---- capture & config parameters ----
            VStack(alignment: .leading, spacing: 10) {
                Label(L10n.t("采集与配置", "Capture & config"), systemImage: "slider.horizontal.3")
                    .font(.subheadline.weight(.semibold))
                paramRow(L10n.t("平均帧率", "Avg frame rate"),
                         status.capture?.fps.map { String(format: "%.1f fps", $0) } ?? "—")
                paramRow(L10n.t("数据模式", "Source"),
                         modeName(status.mode))
                paramRow(L10n.t("命名模型", "Model"),
                         status.options?["vlm"]?.stringValue ?? "—")
                paramRow(L10n.t("调度方式", "Schedule"),
                         status.options?["schedule"]?.stringValue == "parallel"
                            ? L10n.t("阶段并行", "Parallel") : L10n.t("分开运行(串行)", "Serial"))
                paramRow(L10n.t("多视角补全", "Refinement"),
                         refineLabel(status.options?["refine"]))
                if let rejected = status.capture?.rejected_pairs {
                    paramRow(L10n.t("拒收帧对", "Rejected pairs"), String(rejected))
                }
            }
            .padding(14)
            .background(Color(uiColor: .secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
            }
        }
    }

    private var namedCount: Int {
        store.instances.filter { row in
            if let vlm = row.vlm_name { return vlm != "unknown" && !vlm.isEmpty }
            return false
        }.count
    }

    private func refineLabel(_ value: FuzzyValue?) -> String {
        switch value {
        case .bool(true): return L10n.t("开启", "On")
        case .bool(false): return L10n.t("关闭", "Off")
        default: return "—"
        }
    }

    private func modeName(_ raw: String?) -> String {
        switch raw {
        case "ipad": return "iPad LiDAR"
        case "replay": return L10n.t("回放", "Replay")
        case "camera": return L10n.t("相机", "Camera")
        default: return raw ?? "—"
        }
    }

    private func paramRow(_ name: String, _ value: String) -> some View {
        HStack {
            Text(name)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.55))
            Spacer()
            Text(value)
                .font(.caption.monospacedDigit().weight(.medium))
                .foregroundStyle(.white)
                .lineLimit(1)
        }
    }

    private func metricCell(_ name: String, _ value: String, small: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(name)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.55))
            Text(value)
                .font(small ? .caption.monospacedDigit() : .body.bold().monospacedDigit())
                .foregroundStyle(.white)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func fullTimelineSection(_ timeline: [TimelineEntry]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(L10n.t("全流程耗时", "Full pipeline time"), systemImage: "clock")
                .font(.subheadline.weight(.semibold))
            let total = max(1, timeline.reduce(0) { $0 + $1.seconds })
            ForEach(timeline) { entry in
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(fullStageName(entry.stage))
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.7))
                        Spacer()
                        Text(formatSeconds(entry.seconds))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.7))
                    }
                    GeometryReader { geo in
                        Capsule().fill(barColor(entry.stage).opacity(0.7))
                            .frame(width: geo.size.width * CGFloat(entry.seconds / total))
                    }
                    .frame(height: 5)
                }
            }
            Divider().overlay(Color.white.opacity(0.1))
            HStack {
                Text(L10n.t("总计", "Total"))
                    .font(.caption.weight(.semibold))
                Spacer()
                Text(formatSeconds(total))
                    .font(.caption.monospacedDigit().weight(.semibold))
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
        }
    }

    private func fullStageName(_ raw: String) -> String {
        switch raw {
        case "capture": return L10n.t("采集", "Capture")
        case "mapping": return L10n.t("几何建图", "Geometry")
        case "sam3": return "SAM3 " + L10n.t("分割", "segment")
        case "vlm": return "VLM " + L10n.t("命名", "naming")
        case "backfill": return L10n.t("融合回填", "Fuse+backfill")
        case "done": return L10n.t("完成导出", "Export")
        default: return raw
        }
    }

    private func barColor(_ stage: String) -> Color {
        switch stage {
        case "capture": return .blue
        case "mapping": return .teal
        case "sam3": return .orange
        case "vlm": return .purple
        default: return .green
        }
    }

    private func timingSection(_ stages: [StageTiming]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Label(L10n.t("处理耗时", "Processing time"), systemImage: "clock")
                .font(.subheadline.weight(.semibold))
            let total = max(1, stages.reduce(0) { $0 + $1.seconds })
            ForEach(stages) { stage in
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text(stageName(stage.stage))
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.55))
                        Spacer()
                        Text(formatSeconds(stage.seconds))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.55))
                    }
                    GeometryReader { geo in
                        Capsule().fill(Color.green.opacity(0.6))
                            .frame(width: geo.size.width * CGFloat(stage.seconds / total))
                    }
                    .frame(height: 5)
                }
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
        }
    }

    private var objectSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(L10n.t("探索物体", "Explore objects"), systemImage: "cube.transparent")
                .font(.subheadline.weight(.semibold))
            Picker(L10n.t("选择方式", "Group by"), selection: $objectGrouping) {
                Text(L10n.t("类别", "Classes")).tag("class")
                Text(L10n.t("实例", "Instances")).tag("instance")
            }
            .pickerStyle(.segmented)
            .onChange(of: objectGrouping) { _, grouping in
                objectClassFilter = grouping == "instance" && selectedObject?.kind == "class" ? selectedObject?.semantic_id : nil
            }
            if objectGrouping == "instance", let classID = objectClassFilter,
               let category = store.objects?.classes.first(where: { $0.semantic_id == classID }) {
                HStack {
                    Text(category.title).font(.caption.weight(.medium)).foregroundStyle(.green)
                    Spacer()
                    Button(L10n.t("查看全部", "Show all")) { objectClassFilter = nil }
                        .font(.caption)
                }
            }
            Text(L10n.t("点选高亮，再单独查看三维点云", "Select to highlight, then open the object in 3D"))
                .font(.caption2).foregroundStyle(.secondary)
            let rows = objectGrouping == "class" ? (store.objects?.classes ?? []) : (store.objects?.instances ?? []).filter {
                objectClassFilter == nil || $0.semantic_id == objectClassFilter
            }
            if rows.isEmpty {
                Text(L10n.t("没有已标注的物体", "No labeled objects")).font(.footnote).foregroundStyle(.secondary)
            } else {
                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(rows) { object in
                            Button {
                                selectedObject = object
                            } label: {
                                HStack(spacing: 10) {
                                    RoundedRectangle(cornerRadius: 3)
                                        .fill(instanceColor(objectGrouping == "class" ? object.semantic_id : object.label_id))
                                        .frame(width: 8, height: 28)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(object.title).font(.footnote.weight(.medium)).foregroundStyle(.primary)
                                        Text(L10n.t("\(object.point_count.formatted()) 点", "\(object.point_count.formatted()) points"))
                                            .font(.caption2).foregroundStyle(.secondary)
                                    }
                                    Spacer(minLength: 4)
                                    Image(systemName: selectedObject?.id == object.id ? "checkmark.circle.fill" : "viewfinder")
                                        .foregroundStyle(selectedObject?.id == object.id ? Color.green : Color.secondary)
                                }
                                .padding(.horizontal, 8).padding(.vertical, 8)
                                .background(selectedObject?.id == object.id ? Color.green.opacity(0.12) : .clear,
                                            in: RoundedRectangle(cornerRadius: 10))
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            .accessibilityIdentifier("object-\(object.id)")
                            .accessibilityAddTraits(selectedObject?.id == object.id ? .isSelected : [])
                        }
                    }
                }
                .frame(maxHeight: 238)
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 20, style: .continuous).strokeBorder(.white.opacity(0.06), lineWidth: 0.5) }
    }

    private var airGrabBanner: some View {
        VStack(alignment: .leading, spacing: 10) {
          HStack(spacing: 10) {
            Image(systemName: "hand.draw.fill").foregroundStyle(.green)
            VStack(alignment: .leading, spacing: 3) {
                Text(airGrab.instruction).font(.caption.weight(.medium)).lineLimit(2, reservesSpace: true)
                Text(L10n.t("前置相机 · 仅本机识别", "Front camera · On-device processing"))
                    .font(.caption2).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            if hSize != .compact { airGrabModePicker }
            Button { airGrab.stop() } label: { Image(systemName: "xmark.circle.fill").font(.title3) }
                .buttonStyle(.plain).foregroundStyle(.secondary)
                .accessibilityLabel(L10n.t("关闭隔空操控", "Turn off Air Grab"))
          }
          if hSize == .compact { airGrabModePicker }
        }
        .padding(.horizontal, 20).padding(.vertical, 10)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
    }

    private var airGrabModePicker: some View {
        Picker(L10n.t("手势操作", "Hand control"), selection: Binding(
            get: { airGrab.mode }, set: { airGrab.setMode($0) }
        )) {
            Text(L10n.t("移动", "Move")).tag(AirGrabController.Mode.move)
            Text(L10n.t("旋转", "Rotate")).tag(AirGrabController.Mode.rotate)
        }
        .pickerStyle(.segmented)
        .frame(width: 172)
        .accessibilityIdentifier("air-grab-mode")
    }

    private func selectionBar(_ object: CloudObject) -> some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 16) { selectionTitle(object); Spacer(minLength: 8); selectionActions(object) }
            VStack(alignment: .leading, spacing: 12) { selectionTitle(object); selectionActions(object) }
        }
        .padding(.horizontal, 20).padding(.vertical, 14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
    }

    private func selectionTitle(_ object: CloudObject) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(object.title).font(.subheadline.weight(.semibold)).lineLimit(1)
            Text(object.dimensions).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                .accessibilityLabel(L10n.t("包围盒尺寸 \(object.dimensions)", "Bounding box size \(object.dimensions)"))
        }
    }

    private func selectionActions(_ object: CloudObject) -> some View {
        HStack(spacing: 12) {
            Button {
                airGrab.rearm()
                isolationRevision += 1
                isolateObject.toggle()
            } label: {
                Label(isolateObject ? L10n.t("放回场景", "Put back") : L10n.t("单独查看", "Isolate"),
                      systemImage: isolateObject ? "arrow.uturn.backward" : "cube")
            }
            .buttonStyle(.borderedProminent).tint(.green)
            .accessibilityIdentifier("isolate-object")
            .accessibilityHint(isolateObject ? L10n.t("放回原位置并恢复抓取前的场景视角", "Return to the original position and scene view") : "")
            Button {
                if airGrab.enabled { airGrab.stop() }
                else { airGrab.start(capture: capture) }
            } label: {
                Image(systemName: airGrab.enabled ? "hand.draw.fill" : "hand.draw")
            }
            .buttonStyle(.bordered).tint(airGrab.enabled ? .green : .white)
            .accessibilityLabel(L10n.t("隔空操控", "Air Grab"))
            .accessibilityValue(airGrab.enabled ? L10n.t("已开启", "On") : L10n.t("已关闭", "Off"))
            .accessibilityIdentifier("air-grab-toggle")
            .disabled(!viewerObjectsReady)
            Button {
                Task { await exportObject(object) }
            } label: {
                if exportingObject { ProgressView() } else { Image(systemName: "square.and.arrow.up") }
            }
            .buttonStyle(.bordered).disabled(exportingObject)
            .accessibilityLabel(L10n.t("导出物体 PLY", "Export object PLY"))
            Button {
                selectedObject = nil
                isolateObject = false
                isolationRevision += 1
            } label: { Image(systemName: "xmark") }
            .buttonStyle(.bordered)
            .accessibilityLabel(L10n.t("取消选择", "Clear selection"))
        }
        .font(.footnote.weight(.semibold))
        .controlSize(.regular)
    }

    private func exportObject(_ object: CloudObject) async {
        guard !exportingObject else { return }
        airGrab.stop()
        exportingObject = true
        defer { exportingObject = false }
        guard let url = URL(string: "http://\(host):8765/s/\(record.id)/object.ply?kind=\(object.kind)&id=\(object.label_id)") else { return }
        do {
            let (data, response) = try await URLSession.shared.data(for: URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData))
            guard (response as? HTTPURLResponse)?.statusCode == 200,
                  data.starts(with: Data("ply\n".utf8)) else { throw URLError(.badServerResponse) }
            let folder = FileManager.default.temporaryDirectory.appendingPathComponent("ScanObjectExports", isDirectory: true)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let file = folder.appendingPathComponent("\(record.id)_\(object.kind)_\(object.label_id).ply")
            try data.write(to: file, options: .atomic)
            shareFile = ObjectShareFile(url: file)
        } catch {
            exportError = L10n.t("请检查工作站连接后重试：", "Check the workstation connection and retry: ") + error.localizedDescription
        }
    }

    private var instanceSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(L10n.t("识别结果", "Instances"), systemImage: "cube")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(L10n.t("\(store.instances.count) 个", "\(store.instances.count) total"))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.55))
            }
            if store.instances.isEmpty {
                Text(L10n.t("暂无实例数据", "No instance data"))
                    .font(.footnote)
                    .foregroundStyle(.white.opacity(0.55))
            } else {
                ForEach(store.instances) { item in
                    HStack(spacing: 10) {
                        RoundedRectangle(cornerRadius: 2)
                            .fill(instanceColor(item.instance_id))
                            .frame(width: 10, height: 26)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(displayName(item))
                                .font(.footnote.weight(.medium))
                                .foregroundStyle(.white)
                            Text(L10n.t("\(item.semantic_name) · \(item.point_count) 点",
                                        "\(item.semantic_name) · \(item.point_count) pts"))
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.55))
                        }
                        Spacer()
                        if let vlm = item.vlm_name, vlm != "unknown", !vlm.isEmpty {
                            Text(vlm)
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Capsule().fill(Color.green.opacity(0.25)))
                                .foregroundStyle(.green)
                        } else {
                            Circle().fill(Color.gray.opacity(0.5)).frame(width: 6, height: 6)
                        }
                    }
                }
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
        }
    }

    private func displayName(_ item: InstanceRow) -> String {
        "#\(item.instance_id)"
    }

    private func instanceColor(_ id: Int) -> Color {
        if id <= 0 { return Color(white: 0.45) }
        let r = Double(50 + (id * 73) % 206) / 255.0
        let g = Double(50 + (id * 151) % 206) / 255.0
        let b = Double(50 + (id * 199) % 206) / 255.0
        return Color(red: r, green: g, blue: b)
    }

    private func stageName(_ raw: String) -> String {
        switch raw {
        case "dense": return L10n.t("轨迹估计", "Trajectory")
        case "graph": return L10n.t("图优化", "Pose graph")
        case "refill": return L10n.t("全帧回填", "Refill")
        case "fusion": return L10n.t("融合", "Fusion")
        default: return raw
        }
    }

    private func formatSeconds(_ value: Double?) -> String {
        guard let value else { return "—" }
        let s = Int(value.rounded())
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}
