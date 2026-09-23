import SwiftUI
import ARKit
import SceneKit

struct ModelsResponse: Codable {
    let models: [String]
}

/// Full-bleed ARKit camera preview sharing the controller's capture session.
struct CameraPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = true
        view.scene = SCNScene()
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

private enum LinkState {
    case idle, starting, waiting, connected, reconnecting, stopped, failed

    static func from(_ raw: String) -> LinkState {
        switch raw {
        case "Connected": return .connected
        case "Starting ARKit": return .starting
        case "Waiting for workstation": return .waiting
        case "Reconnecting": return .reconnecting
        case "Stopped": return .stopped
        case "Failed": return .failed
        default: return .idle
        }
    }

    var label: String {
        switch self {
        case .idle: return "未连接"
        case .starting: return "启动 ARKit"
        case .waiting: return "等待接收端"
        case .connected: return "已连接 · 串流中"
        case .reconnecting: return "重连中"
        case .stopped: return "已停止"
        case .failed: return "失败"
        }
    }

    var color: Color {
        switch self {
        case .idle, .stopped: return .gray
        case .starting: return .blue
        case .waiting: return .yellow
        case .connected: return .green
        case .reconnecting, .failed: return .red
        }
    }
}

private func elapsedText(from date: Date?, now: Date) -> String {
    guard let date else { return "00:00" }
    let seconds = max(0, Int(now.timeIntervalSince(date).rounded(.down)))
    return String(format: "%02d:%02d", seconds / 60, seconds % 60)
}

struct CaptureView: View {
    @AppStorage("appLang") private var lang = "zh"
    @ObservedObject var capture: CaptureController
    @AppStorage("receiverHost") private var receiverHost = ""
    @AppStorage("receiverPort") private var receiverPort = 7001
    @AppStorage("vlmModel") private var vlmModel = "qwen3vl_2b_nf4"
    @State private var availableModels: [String] = []
    @State private var modelPushFailed = false
    @State private var confirmDiscard = false
    @FocusState private var hostFieldFocused: Bool
    @Environment(\.horizontalSizeClass) private var hSize
    private var compact: Bool { hSize == .compact }

    private static let defaultHost = "100.72.138.33"

    private var trimmedHost: String {
        receiverHost.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var validConfiguration: Bool {
        !trimmedHost.isEmpty && (1...65535).contains(receiverPort)
    }

    var body: some View {
        ZStack {
            if capture.isStreaming, let session = capture.activeSession {
                streamingScreen(session: session)
            } else if capture.isPreviewing, let session = capture.activeSession {
                previewScreen(session: session)
            } else {
                setupScreen
            }
        }
        .confirmationDialog(L10n.t("删除这份本地录制？", "Delete this local recording?"), isPresented: $confirmDiscard, titleVisibility: .visible) {
            Button(L10n.t("删除录制", "Delete recording"), role: .destructive) { capture.discardRecording() }
            Button(L10n.t("取消", "Cancel"), role: .cancel) { }
        } message: {
            Text(L10n.t("尚未上传的数据将无法恢复。", "Data that has not been uploaded cannot be recovered."))
        }
        .animation(.easeInOut(duration: 0.25), value: capture.isStreaming)
        .animation(.easeInOut(duration: 0.25), value: capture.isPreviewing)
        .toolbar(capture.isStreaming || capture.isPreviewing ? .hidden : .visible, for: .tabBar)
    }

    // MARK: - Setup

    private var setupScreen: some View {
        ScrollView {
            VStack(spacing: 24) {
                Spacer(minLength: 36)

                ZStack {
                    VStack(spacing: 8) {
                        Image(systemName: "cube.transparent")
                            .font(.system(size: 40, weight: .light))
                            .foregroundStyle(.green)
                            .accessibilityHidden(true)
                        Text(L10n.t("LiDAR RGB-D 串流", "LiDAR RGB-D Streaming"))
                            .foregroundStyle(.white)
                    }
                    HStack {
                        Spacer()
                        LanguageToggle()
                    }
                }

                configCard

                modelCard

                checkButton

                modePicker

                if capture.pendingRecordings > 0 || capture.isSavingRecording {
                    uploadCard
                }

                startButton
                if let message = capture.uploadMessage {
                    Text(message).font(.footnote).foregroundStyle(.green)
                }

                if capture.receiverFinished {
                    Text(L10n.t("本次采集已完成，工作站建图中 · 到「记录」页查看结果", "Capture done · mapping in progress — see Records"))
                        .font(.footnote.weight(.medium))
                        .foregroundStyle(.green)
                        .frame(maxWidth: .infinity, alignment: .leading)
                } else if let error = capture.lastError {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                Text(L10n.t("状态：\(capture.status) · \(capture.receiverEndpoint)", "Status: \(capture.status) · \(capture.receiverEndpoint)"))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
                    .frame(maxWidth: .infinity, alignment: .leading)

                Spacer(minLength: 32)
            }
            .padding(.horizontal, 44)
            .frame(maxWidth: 620)
            .frame(maxWidth: .infinity)
        }
        .background(Color(uiColor: .systemGroupedBackground).ignoresSafeArea())
        .navigationTitle("Scan")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: "\(trimmedHost):\(receiverPort)") {
            guard !trimmedHost.isEmpty else { return }
            await capture.checkReceiver(host: trimmedHost, port: UInt16(receiverPort))
            await loadModels()
        }
    }

    private var configCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: "server.rack")
                    .foregroundStyle(.green)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(L10n.t("接收端地址", "Receiver address"))
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.5))
                    TextField("100.72.138.33", text: $receiverHost)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .focused($hostFieldFocused)
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.white)
                        .accessibilityIdentifier("receiverHost")
                }
            }
            Divider().overlay(Color.white.opacity(0.12))
            HStack {
                Text(L10n.t("TCP 端口", "TCP port"))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                TextField("7001", value: $receiverPort, format: .number)
                    .keyboardType(.numberPad)
                    .multilineTextAlignment(.trailing)
                    .frame(width: 110)
                    .font(.body.monospacedDigit())
            }
            Button {
                receiverHost = Self.defaultHost
                receiverPort = 7001
            } label: {
                Text(L10n.t("使用默认工作站 \(Self.defaultHost):7001", "Use default workstation \(Self.defaultHost):7001"))
                    .font(.footnote.weight(.medium))
                    .foregroundStyle(.green)
            }
        }
        .padding(18)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
                .allowsHitTesting(false)
        }
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                Image(systemName: "brain")
                    .foregroundStyle(.green)
                    .accessibilityHidden(true)
                Text(L10n.t("命名模型", "Naming model"))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
                Spacer()
                Menu {
                    ForEach(availableModels.isEmpty ? [vlmModel] : availableModels, id: \.self) { model in
                        Button {
                            vlmModel = model
                            Task { await pushModelChoice() }
                        } label: {
                            if model == vlmModel {
                                Label(model, systemImage: "checkmark")
                            } else {
                                Text(model)
                            }
                        }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Text(vlmModel)
                            .font(.system(.footnote, design: .monospaced))
                            .foregroundStyle(.white)
                            .lineLimit(1)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.5))
                    }
                }
                .accessibilityIdentifier("vlmModelMenu")
            }
            if modelPushFailed {
                Text(L10n.t("模型选择未同步到工作站（将在开始扫描时重试）", "Model choice not synced (will retry on start)"))
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
            if !trimmedHost.isEmpty && trimmedHost != Self.defaultHost {
                Text(L10n.t("非默认工作站：iOS 26 可能拦截其 http 访问，记录页若加载失败请改回 \(Self.defaultHost)", "Non-default host: iOS 26 may block its http; revert to \(Self.defaultHost) if Records fails"))
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
        .padding(18)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
                .allowsHitTesting(false)
        }
    }

    private func loadModels() async {
        guard let url = URL(string: "http://\(trimmedHost):8765/api/models") else { return }
        guard let (data, _) = try? await URLSession.shared.data(from: url),
              let decoded = try? JSONDecoder().decode(ModelsResponse.self, from: data) else { return }
        availableModels = decoded.models
    }

    private func pushModelChoice() async {
        modelPushFailed = !(await pushOptions())
    }

    private func pushOptions() async -> Bool {
        let host = trimmedHost
        guard !host.isEmpty else { return false }
        guard let token = await fetchWorkstationToken(host: host) else { return false }
        guard let url = URL(string: "http://\(host):8765/api/options") else { return false }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue(token, forHTTPHeaderField: "X-Scan-Token")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try? JSONEncoder().encode(["vlm": vlmModel])
        guard let (_, response) = try? await URLSession.shared.data(for: request) else { return false }
        return (response as? HTTPURLResponse)?.statusCode == 200
    }

    private func fetchWorkstationToken(host: String) async -> String? {
        guard let url = URL(string: "http://\(host):8765/") else { return nil }
        guard let (data, _) = try? await URLSession.shared.data(from: url),
              let html = String(data: data, encoding: .utf8),
              let range = html.range(of: "token='[0-9a-f]{48}'", options: .regularExpression)
        else { return nil }
        return String(html[range].dropFirst(7).dropLast())
    }

    private var modePicker: some View {
        HStack(spacing: 12) {
            modeButton(local: false, label: L10n.t("实时串流", "Live stream"), icon: "antenna.radiowaves.left.and.right")
            modeButton(local: true, label: L10n.t("本地录制", "Local record"), icon: "internaldrive")
        }
    }

    private func modeButton(local: Bool, label: String, icon: String) -> some View {
        Button {
            capture.setMode(local: local)
        } label: {
            VStack(spacing: 5) {
                Image(systemName: icon).font(.body)
                Text(label).font(.caption.weight(.medium))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 9)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill((capture.isLocalMode == local) ? Color.green.opacity(0.16) : Color(uiColor: .secondarySystemGroupedBackground))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .stroke((capture.isLocalMode == local) ? Color.green.opacity(0.65) : Color.white.opacity(0.06), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }

    private var uploadCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(L10n.t("待上传录制", "Pending upload"), systemImage: "arrow.up.doc")
                    .font(.footnote.weight(.semibold))
                Spacer()
                Text("\(capture.recordedPackets) \(L10n.t("帧", "fr")) · \(String(format: "%.0f", capture.recordedMB)) MB")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.white.opacity(0.6))
            }
            if capture.isSavingRecording {
                ProgressView(L10n.t("正在保存录制，请稍候…", "Saving recording…"))
            } else if let progress = capture.uploadProgress {
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: Double(progress.sent), total: Double(max(progress.total, 1)))
                        .tint(.green)
                    Text(L10n.t("上传中 \(progress.sent)/\(progress.total)", "Uploading \(progress.sent)/\(progress.total)"))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Button(L10n.t("暂停上传", "Pause upload")) { capture.cancelUpload() }
                    .font(.footnote)
            } else {
                HStack(spacing: 10) {
                    Button {
                        Task { await capture.uploadRecording(host: trimmedHost) }
                    } label: {
                        Label(L10n.t("上传到工作站", "Upload"), systemImage: "icloud.and.arrow.up")
                            .font(.footnote.weight(.semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 8)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(trimmedHost.isEmpty)
                    Button(role: .destructive) {
                        confirmDiscard = true
                    } label: {
                        Image(systemName: "trash")
                    }
                }
            }
            if capture.pendingRecordings > 1 {
                Text(L10n.t("还有 \(capture.pendingRecordings) 份录制待处理", "\(capture.pendingRecordings) recordings pending"))
                    .font(.caption2).foregroundStyle(.secondary)
            }
            if let error = capture.uploadError {
                Text(error).font(.caption2).foregroundStyle(.red)
            }
        }
        .padding(14)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .strokeBorder(Color.white.opacity(0.06), lineWidth: 0.5)
                .allowsHitTesting(false)
        }
    }

    private var checkButton: some View {
        Button {
            hostFieldFocused = false
            Task { await capture.checkReceiver(host: trimmedHost, port: UInt16(receiverPort)) }
        } label: {
            HStack {
                if capture.checking {
                    ProgressView().controlSize(.small)
                    Text(L10n.t("检查中…", "Checking…"))
                } else {
                    Image(systemName: capture.checkPassed == true ? "checkmark.seal.fill" :
                            capture.checkPassed == false ? "exclamationmark.triangle.fill" : "stethoscope")
                    Text(checkLabel)
                }
                Spacer()
            }
            .font(.footnote.weight(.medium))
            .foregroundStyle(.white.opacity(0.85))
            .padding(.horizontal, 14)
            .padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(checkTint.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .strokeBorder(checkTint.opacity(0.3), lineWidth: 0.5)
            )
        }
        .disabled(capture.checking || trimmedHost.isEmpty)
        .accessibilityIdentifier("checkButton")
    }

    private var checkLabel: String {
        if capture.checkPassed == true { return L10n.t("接收端就绪 · 点开始即自动建会话", "Receiver ready · auto-starts on connect") }
        if capture.checkPassed == false { return L10n.t("无法连接工作站（检查网络或服务是否运行）", "Workstation unreachable (check network/service)") }
        return L10n.t("检查工作站连接", "Check workstation")
    }

    private var checkTint: Color {
        if capture.checkPassed == true { return .green }
        if capture.checkPassed == false { return .orange }
        return .blue
    }

    private var startButton: some View {
        Button {
            hostFieldFocused = false
            capture.startPreview()
        } label: {
            Text(capture.isLocalMode ? L10n.t("开始本地录制", "Start local recording") : L10n.t("开始扫描串流", "Start streaming"))
                .font(.body.bold())
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.glassProminent)
        .controlSize(.large)
        .tint(.green)
        .disabled(!validConfiguration || !capture.canStartCapture)
        .accessibilityIdentifier("streamButton")
    }

    private func paramRow(_ name: String, _ value: String) -> some View {
        HStack(spacing: 6) {
            Text(name)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.6))
            Spacer()
            Text(value)
                .font(.caption2.monospacedDigit().weight(.medium))
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Camera preview (pre-start)

    private func previewScreen(session: ARSession) -> some View {
        ZStack {
            CameraPreview(session: session)
                .ignoresSafeArea()

            VStack {
                HStack(spacing: 10) {
                    statusChip
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("\(trimmedHost):\(receiverPort)")
                            .font(.caption.monospaced())
                            .foregroundStyle(.white.opacity(0.9))
                        Text(vlmModel)
                            .font(.caption2.monospaced())
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
                .padding(12)
                .background(Color.black.opacity(0.55), in: Capsule())
                .padding(.horizontal, 16)

                Spacer()

                HStack(alignment: .bottom) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(L10n.t("参数", "Parameters"))
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.white.opacity(0.6))
                        paramRow(L10n.t("接收端", "Receiver"), "\(trimmedHost):\(receiverPort)")
                        paramRow(L10n.t("命名模型", "Model"), vlmModel)
                        paramRow(L10n.t("链路", "Link"),
                                 capture.checkPassed == true ? L10n.t("就绪", "Ready") :
                                 (capture.checking ? L10n.t("检测中…", "Checking…") : L10n.t("未就绪", "Not ready")))
                        HStack(spacing: 6) {
                            Circle().fill(capture.depthStats.tracking == "normal" ? Color.green : Color.orange)
                                .frame(width: 6, height: 6)
                            Text(L10n.t("跟踪", "Track"))
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.6))
                            Spacer()
                            Text(capture.depthStats.tracking == "normal" ? L10n.t("正常", "OK") : L10n.t("受限", "Limited"))
                                .font(.caption2.monospacedDigit())
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        HStack(spacing: 6) {
                            Circle().fill(capture.depthStats.validFraction >= 0.6 ? Color.green : Color.orange)
                                .frame(width: 6, height: 6)
                            Text(L10n.t("深度有效", "Depth"))
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.6))
                            Spacer()
                            Text(String(format: "%.0f%%", capture.depthStats.validFraction * 100))
                                .font(.caption2.monospacedDigit())
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        HStack(spacing: 6) {
                            Circle().fill(capture.depthStats.medianDepthM >= 0.4 && capture.depthStats.medianDepthM <= 4.0 ? Color.green : Color.orange)
                                .frame(width: 6, height: 6)
                            Text(L10n.t("中位距离", "Median"))
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.6))
                            Spacer()
                            Text(String(format: "%.1f m", capture.depthStats.medianDepthM))
                                .font(.caption2.monospacedDigit())
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        if let depth = capture.depthImage {
                            Image(uiImage: depth)
                                .resizable()
                                .interpolation(.medium)
                                .frame(width: 150, height: 112)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.25), lineWidth: 1))
                        }
                    }
                    .padding(10)
                    .background(Color.black.opacity(0.55))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal, 16)
                    Spacer()
                }

                Text(L10n.t("预览模式 · 未推流", "Preview · not streaming"))
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white.opacity(0.9))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color.black.opacity(0.55), in: Capsule())

                HStack(spacing: 14) {
                    Button {
                        capture.stopPreview()
                    } label: {
                        Label(L10n.t("返回", "Back"), systemImage: "chevron.left")
                            .font(.body.weight(.medium))
                            .padding(.horizontal, 20)
                            .padding(.vertical, 12)
                            .background(Color.black.opacity(0.55), in: Capsule())
                    }

                    Button {
                        Task {
                            _ = await pushOptions()
                            capture.start(host: trimmedHost, port: UInt16(receiverPort), local: capture.isLocalMode)
                        }
                    } label: {
                        Label(L10n.t("开始扫描", "Start scan"), systemImage: "record.circle")
                            .font(.body.bold())
                            .foregroundStyle(.black)
                            .padding(.horizontal, 26)
                            .padding(.vertical, 12)
                            .background(Capsule().fill(Color.green))
                    }
                    .accessibilityIdentifier("confirmStart")
                }
                .padding(.bottom, 26)
            }
        }
        .task(id: "\(trimmedHost):\(receiverPort)") {
            guard !trimmedHost.isEmpty else { return }
            await capture.checkReceiver(host: trimmedHost, port: UInt16(receiverPort))
        }
    }

    // MARK: - Streaming

    private func streamingScreen(session: ARSession) -> some View {
        ZStack {
            CameraPreview(session: session)
                .ignoresSafeArea()

            VStack {
                HStack(alignment: .top, spacing: 12) {
                    statusChip
                    Spacer()
                    VStack(alignment: .trailing, spacing: 6) {
                        Text(capture.receiverEndpoint)
                            .font(.caption.monospaced())
                            .foregroundStyle(.white.opacity(0.9))
                        TimelineView(.periodic(from: .now, by: 1)) { timeline in
                            Text(elapsedText(from: capture.streamStartedAt, now: timeline.date))
                                .font(.title3.bold().monospacedDigit())
                                .foregroundStyle(.white)
                        }
                        Text(capture.isLocalMode
     ? L10n.t("已录 \(capture.recordedPackets) 帧 · \(String(format: "%.0f", capture.recordedMB)) MB",
              "REC \(capture.recordedPackets) · \(String(format: "%.0f", capture.recordedMB)) MB")
     : L10n.t("已发帧 \(capture.encodedFrames) · 跳过 \(capture.skippedFrames)",
              "Sent \(capture.encodedFrames) · Skip \(capture.skippedFrames)"))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.white.opacity(0.85))
                    }
                }
                .padding(14)
                .background(Color.black.opacity(0.55))
                .clipShape(RoundedRectangle(cornerRadius: 14))
                .padding(.horizontal, 16)

                Spacer()

                HStack(alignment: .bottom) {
                    ScrollView(.vertical, showsIndicators: false) {
                    VStack(alignment: .leading, spacing: 4) {
                        if let image = capture.depthImage {
                            Image(uiImage: image)
                                .resizable()
                                .interpolation(.medium)
                                .frame(maxWidth: .infinity)
                                .aspectRatio(4 / 3, contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(Color.white.opacity(0.25), lineWidth: 1)
                                )
                        } else {
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color.black.opacity(0.4))
                                .frame(maxWidth: .infinity)
                                .aspectRatio(4 / 3, contentMode: .fit)
                                .overlay(Text(L10n.t("等待深度…", "Waiting for depth…")).font(.caption2).foregroundStyle(.white.opacity(0.6)))
                        }
                        Text(L10n.t("深度预览 0.4–4 m（近绿远红）", "Depth 0.4–4 m (near=green)"))
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.85))
                        depthStatsLines
                        TrajectoryMiniMap(trajectory: capture.trajectory)
                            .frame(maxWidth: .infinity)
                                .aspectRatio(2 / 1, contentMode: .fit)
                            .background(Color.black.opacity(0.5))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.25), lineWidth: 1))
                        Text(L10n.t("轨迹小地图 · 白点=起点 绿点=当前位置", "Path map · white=start green=now"))
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.85))
                        if let coverage = capture.coverageImage {
                            Image(uiImage: coverage)
                                .resizable()
                                .interpolation(.medium)
                                .frame(maxWidth: .infinity)
                                .aspectRatio(2 / 1, contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.white.opacity(0.25), lineWidth: 1)
                                )
                            Text(L10n.t("已扫覆盖 \(Int(capture.depthStats.coveragePercent * 100))% · 白线=当前朝向", "Coverage \(Int(capture.depthStats.coveragePercent * 100))% · white=facing"))
                                .font(.caption2)
                                .foregroundStyle(.white.opacity(0.85))
                        }
                    }
                    .padding(10)
                    }
                    .background(Color.black.opacity(0.55))
                    .clipShape(RoundedRectangle(cornerRadius: 14))
                    .padding(.horizontal, 16)
                    .frame(maxWidth: compact ? .infinity : 260, maxHeight: compact ? 300 : .infinity, alignment: .bottom)
                    Spacer()
                }

                Spacer(minLength: 8)

                VStack(spacing: 14) {
                    if let error = capture.lastError {
                        Text(error)
                            .font(.footnote)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 8)
                            .background(Color.red.opacity(0.85), in: Capsule())
                    }
                    capture.isLocalMode
                        ? Text(L10n.t("本地录制中 · 停止后上传", "Recording locally · upload after stop"))
                            .font(.caption.weight(.medium))
                            .foregroundStyle(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Color.green.opacity(0.7), in: Capsule())
                        : scanGuidance
                        .font(.caption.weight(.medium))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 7)
                        .background(guidanceColor.opacity(0.8), in: Capsule())

                    Button {
                        capture.stop()
                    } label: {
                        ZStack {
                            Circle()
                                .fill(Color.red.gradient)
                                .frame(width: 78, height: 78)
                                .shadow(color: .black.opacity(0.35), radius: 10, y: 4)
                            Image(systemName: "stop.fill")
                                .font(.system(size: 30))
                                .foregroundStyle(.white)
                        }
                    }
                    .accessibilityLabel(L10n.t("停止串流", "Stop streaming"))
                    .accessibilityIdentifier("streamButton")
                }
                .padding(.bottom, 28)
            }
        }
    }

    private var depthStatsLines: some View {
        let s = capture.depthStats
        return VStack(alignment: .leading, spacing: 3) {
            statLine(L10n.t("有效深度", "Valid depth"), String(format: "%.0f%%", s.validFraction * 100),
                     ok: s.validFraction >= 0.6)
            statLine(L10n.t("中位距离", "Median dist"), String(format: "%.1f m", s.medianDepthM),
                     ok: s.medianDepthM >= 0.4 && s.medianDepthM <= 4.0)
            statLine(L10n.t("跟踪状态", "Tracking"), s.tracking == "normal" ? L10n.t("正常", "OK") : L10n.t("受限", "Limited"), ok: s.tracking == "normal")
            statLine(L10n.t("已扫覆盖", "Coverage"), String(format: "%.0f%%", s.coveragePercent * 100),
                     ok: s.coveragePercent >= 0.55)
        }
        .font(.caption2.monospacedDigit())
    }

    private func statLine(_ name: String, _ value: String, ok: Bool) -> some View {
        HStack(spacing: 6) {
            Circle().fill(ok ? Color.green : Color.orange).frame(width: 6, height: 6)
            Text(name).foregroundStyle(.white.opacity(0.5))
            Spacer(minLength: 4)
            Text(value).foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var guidanceColor: Color {
        let s = capture.depthStats
        if s.tracking != "normal" { return .red }
        if s.medianDepthM > 4.0 || (s.medianDepthM > 0 && s.medianDepthM < 0.4) { return .orange }
        if s.validFraction < 0.5 { return .orange }
        return .green.opacity(0.75)
    }

    private var scanGuidance: Text {
        let s = capture.depthStats
        if s.tracking == "limited" {
            return Text(L10n.t("跟踪受限：放慢速度，回到纹理丰富的区域", "Tracking limited: slow down, return to textured area"))
        }
        if s.tracking == "not_available" {
            return Text(L10n.t("跟踪不可用：暂停移动，等待恢复", "Tracking unavailable: hold still until recovered"))
        }
        if s.medianDepthM > 4.0 {
            return Text(L10n.t("太远了：靠近目标到 4 米以内", "Too far: move within 4 m"))
        }
        if s.medianDepthM > 0 && s.medianDepthM < 0.4 {
            return Text(L10n.t("太近了：退后到 0.4 米以外", "Too close: step back beyond 0.4 m"))
        }
        if s.validFraction < 0.5 {
            return Text(L10n.t("深度覆盖低：正对表面，避免斜掠角度", "Low depth coverage: face surfaces directly"))
        }
        if s.coveragePercent < 0.55 {
            return Text(L10n.t("还有空白方向：转动身体扫向上图暗区", "Gaps remain: sweep toward dark areas above"))
        }
        return Text(L10n.t("状态良好：缓慢移动，多角度覆盖", "Good: move slowly, cover angles"))
    }

    private var statusChip: some View {
        let state = LinkState.from(capture.status)
        return HStack(spacing: 8) {
            Circle()
                .fill(state.color)
                .frame(width: 10, height: 10)
            Text(state.label)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.55), in: Capsule())
    }
}

#Preview {
    NavigationStack {
        CaptureView(capture: CaptureController())
    }
}

// MARK: - Trajectory mini-map (top-down, ARKit positions)

struct TrajectoryMiniMap: View {
    let trajectory: [CGPoint]

    var body: some View {
        Canvas { context, size in
            guard trajectory.count >= 2 else { return }
            let xs = trajectory.map { $0.x }, ys = trajectory.map { $0.y }
            let minX = xs.min()!, maxX = xs.max()!, minY = ys.min()!, maxY = ys.max()!
            let spanX = max(0.3, maxX - minX), spanY = max(0.3, maxY - minY)
            let scale = min((size.width - 16) / spanX, (size.height - 16) / spanY)
            let map = { (p: CGPoint) -> CGPoint in
                CGPoint(x: (p.x - minX) * scale + 8 + (size.width - 16 - spanX * scale) / 2,
                        y: (p.y - minY) * scale + 8 + (size.height - 16 - spanY * scale) / 2)
            }
            var path = Path()
            path.move(to: map(trajectory[0]))
            for point in trajectory.dropFirst() { path.addLine(to: map(point)) }
            context.stroke(path, with: .color(.green.opacity(0.45)), lineWidth: 2)
            let start = map(trajectory[0]), end = map(trajectory.last!)
            context.fill(Path(ellipseIn: CGRect(x: start.x-4, y: start.y-4, width: 8, height: 8)),
                         with: .color(.white))
            context.fill(Path(ellipseIn: CGRect(x: end.x-5, y: end.y-5, width: 10, height: 10)),
                         with: .color(.green))
        }
    }
}
