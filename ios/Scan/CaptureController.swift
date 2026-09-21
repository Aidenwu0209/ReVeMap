import ARKit
import Combine
import CoreImage
import CoreMotion
import Foundation
import Network
import UIKit

struct DepthPreviewStats {
    var validFraction: Double = 0
    var medianDepthM: Double = 0
    var tracking = "normal"
    var coveragePercent: Double = 0
}

@MainActor
final class CaptureController: ObservableObject {
    @Published private(set) var isStreaming = false
    @Published private(set) var status = "Idle"
    @Published private(set) var receiverEndpoint = "Not configured"
    @Published private(set) var encodedFrames: UInt64 = 0
    @Published private(set) var skippedFrames: UInt64 = 0
    @Published private(set) var lastError: String?
    @Published private(set) var streamStartedAt: Date?
    @Published private(set) var checking = false
    @Published private(set) var checkPassed: Bool?
    @Published private(set) var receiverFinished = false
    @Published private(set) var isPreviewing = false
    @Published private(set) var isLocalMode = UserDefaults.standard.bool(forKey: "scanLocalMode")
    @Published private(set) var recordedPackets = 0
    @Published private(set) var recordedMB = 0.0
    @Published private(set) var uploadProgress: (sent: Int, total: Int)? = nil
    @Published private(set) var uploadError: String? = nil
    @Published private(set) var isSavingRecording = false
    @Published private(set) var isUploading = false
    @Published private(set) var pendingRecordings = 0
    @Published private(set) var uploadMessage: String?
    private var uploadTask: Task<UploadReceipt, Error>?
    private var localStore: LocalPacketStore?

    var canStartCapture: Bool { !isSavingRecording && !isUploading && pendingRecordings == 0 }

    init() { recoverPendingRecording() }

    private func recoverPendingRecording() {
        do {
            let directories = try LocalPacketStore.pendingDirectories().filter {
                try FileManager.default.contentsOfDirectory(atPath: $0.path).contains { $0.hasSuffix(".bin") }
            }
            pendingRecordings = directories.count
            localStore = try directories.first.map { try LocalPacketStore(recovering: $0) }
            recordedPackets = localStore?.packetCount ?? 0
            recordedMB = Double(localStore?.totalBytes ?? 0) / 1048576.0
        } catch {
            pendingRecordings = max(1, pendingRecordings)
            uploadError = L10n.t("无法恢复录制，原始文件已保留：", "Cannot recover recording; files retained: ") + error.localizedDescription
        }
    }

    @Published private(set) var depthImage: UIImage?
    @Published private(set) var coverageImage: UIImage?
    @Published private(set) var depthStats = DepthPreviewStats()
    @Published private(set) var trajectory: [CGPoint] = []

    private var streamer: ARKitFrameStreamer?
    private var watchdogTask: Task<Void, Never>?
    private let localPreviewSession = ARSession()
    private var previewSampler: PreviewStatsSampler?

    /// The live ARKit session while streaming or previewing, for rendering.
    var activeSession: ARSession? {
        streamer?.session ?? (isPreviewing ? localPreviewSession : nil)
    }

    /// Camera preview WITHOUT streaming: see what the camera sees first.
    func startPreview() {
        guard !isStreaming, canStartCapture else { return }
        let configuration = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            configuration.frameSemantics.insert(.sceneDepth)
        }
        configuration.worldAlignment = .gravity
        let sampler = PreviewStatsSampler { [weak self] image, coverage, stats, position in
            self?.depthImage = image
            self?.coverageImage = coverage
            self?.depthStats = stats
            if let last = self?.trajectory.last {
                if abs(last.x - position.x) > 0.02 || abs(last.y - position.y) > 0.02 {
                    self?.trajectory.append(position)
                    if (self?.trajectory.count ?? 0) > 2400 { self?.trajectory.removeFirst(1200) }
                }
            } else {
                self?.trajectory.append(position)
            }
        }
        localPreviewSession.delegate = sampler
        self.previewSampler = sampler
        localPreviewSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        trajectory = []
        isPreviewing = true
        UIApplication.shared.isIdleTimerDisabled = true
    }

    func setMode(local: Bool) {
        guard !isStreaming else { return }
        isLocalMode = local
        UserDefaults.standard.set(local, forKey: "scanLocalMode")
    }

    func stopPreview() {
        localPreviewSession.delegate = nil
        previewSampler = nil
        localPreviewSession.pause()
        isPreviewing = false
        depthImage = nil
        depthStats = DepthPreviewStats()
        UIApplication.shared.isIdleTimerDisabled = false
    }

    /// Pre-flight check: can we reach the workstation's stream port right now?
    /// The GUI only listens on the stream port after "开始扫描" is pressed on
    /// the web page, so failure usually means the receiver is not armed yet.
    func checkReceiver(host: String, port: UInt16) async {
        guard !host.isEmpty, !checking else { return }
        checking = true
        checkPassed = nil
        defer { checking = false }

        // The stream port only accepts a connection while the GUI is armed;
        // a refused/refused-timeout connection tells us the receiver state.
        var streamReachable = false
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            let connection = NWConnection(
                host: NWEndpoint.Host(host),
                port: NWEndpoint.Port(rawValue: port) ?? 7001,
                using: .tcp
            )
            let box = TimeoutBox()
            connection.stateUpdateHandler = { state in
                switch state {
                case .ready:
                    guard box.hit() else { return }
                    streamReachable = true
                    connection.cancel()
                    continuation.resume()
                case .failed, .cancelled:
                    guard box.hit() else { return }
                    continuation.resume()
                default:
                    break
                }
            }
            connection.start(queue: DispatchQueue(label: "scan.check"))
            DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
                guard box.hit() else { return }
                connection.cancel()
                continuation.resume()
            }
        }
        checkPassed = streamReachable
    }

    func start(host: String, port: UInt16, local: Bool = false) {
        guard !isStreaming, canStartCapture else { return }
        if isPreviewing { stopPreview() }
        isLocalMode = local
        recordedPackets = 0
        recordedMB = 0
        uploadProgress = nil
        uploadError = nil
        uploadMessage = nil
        if local {
            do { localStore = try LocalPacketStore() }
            catch {
                lastError = L10n.t("无法创建本地录制目录", "Cannot create local recording dir")
                status = "Failed"
                return
            }
        } else {
            localStore = nil
        }
        lastError = nil
        receiverEndpoint = "\(host):\(port)"
        encodedFrames = 0
        skippedFrames = 0
        depthImage = nil
        coverageImage = nil
        depthStats = DepthPreviewStats()
        trajectory = []
        let callbacks = StreamCallbacks(
            status: { [weak self] value in self?.status = value },
            encoded: { [weak self] value in self?.encodedFrames = value },
            skipped: { [weak self] value in self?.skippedFrames = value },
            error: { [weak self] value in
                guard let self else { return }
                if self.isLocalMode, let value {
                    self.lastError = value
                    self.stop()
                } else if !self.isLocalMode {
                    self.lastError = value
                }
            },
            preview: { [weak self] image, coverage, stats, position in
                self?.depthImage = image
                self?.coverageImage = coverage
                self?.depthStats = stats
                if let store = self?.localStore {
                    self?.recordedPackets = store.packetCount
                    self?.recordedMB = Double(store.totalBytes) / 1048576.0
                }
                if let last = self?.trajectory.last {
                    if abs(last.x - position.x) > 0.02 || abs(last.y - position.y) > 0.02 {
                        self?.trajectory.append(position)
                        if (self?.trajectory.count ?? 0) > 2400 { self?.trajectory.removeFirst(1200) }
                    }
                } else {
                    self?.trajectory.append(position)
                }
            }
        )
        let candidate = ARKitFrameStreamer(
            serverHost: host,
            serverPort: port,
            callbacks: callbacks,
            localStore: localStore
        )
        do {
            try candidate.start()
            streamer = candidate
            isStreaming = true
            streamStartedAt = Date()
            receiverFinished = false
            if !local { startWatchdog() }
            UIApplication.shared.isIdleTimerDisabled = true
        } catch {
            lastError = error.localizedDescription
            status = "Failed"
        }
    }

    /// Auto-finish: once frames have flowed, a receiver that stays
    /// unreachable for 12 s has sealed and moved on to mapping — stop
    /// streaming and hand the user back to the setup screen.
    private func startWatchdog() {
        watchdogTask?.cancel()
        watchdogTask = Task { [weak self] in
            var lastCount: UInt64 = 0
            var lastChange = Date()
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                guard let self, self.isStreaming else { break }
                if self.encodedFrames != lastCount {
                    lastCount = self.encodedFrames
                    lastChange = Date()
                    continue
                }
                let stalled = self.status == "Waiting for workstation" || self.status == "Reconnecting"
                if stalled && self.encodedFrames > 0 && Date().timeIntervalSince(lastChange) > 12 {
                    self.receiverFinished = true
                    self.stop()
                    break
                }
            }
        }
    }

    func stop() {
        guard isStreaming, !isSavingRecording else { return }
        watchdogTask?.cancel()
        watchdogTask = nil
        isStreaming = false
        streamStartedAt = nil
        isSavingRecording = true
        status = L10n.t("正在保存…", "Saving…")
        streamer?.stop { [weak self] in
            guard let self else { return }
            self.streamer = nil
            self.isSavingRecording = false
            self.status = "Stopped"
            self.recoverPendingRecording()
            UIApplication.shared.isIdleTimerDisabled = false
        }
    }

    // MARK: - Durable local recording upload

    func uploadRecording(host: String) async {
        guard let store = localStore, !isUploading, !isSavingRecording, !isStreaming else { return }
        isUploading = true
        uploadError = nil
        uploadMessage = nil
        uploadProgress = (0, store.packetCount)
        UIApplication.shared.isIdleTimerDisabled = true
        let reportProgress: @MainActor (Int, Int) -> Void = { [weak self] sent, total in
            self?.uploadProgress = (sent, total)
        }
        let task = Task {
            try await RecordingUploader.upload(store: store, host: host, progress: reportProgress)
        }
        uploadTask = task
        // This is a finite grace period, not a claim of unlimited background transfer.
        let background = UIApplication.shared.beginBackgroundTask(withName: "Scan recording upload") {
            task.cancel()
        }
        defer {
            uploadTask = nil
            isUploading = false
            uploadProgress = nil
            UIApplication.shared.isIdleTimerDisabled = false
            if background != .invalid { UIApplication.shared.endBackgroundTask(background) }
        }
        do {
            let receipt = try await task.value
            // Only a matching durable receipt permits local cleanup.
            try LocalPacketStore.deleteRecording(directory: store.directory)
            localStore = nil
            recoverPendingRecording()
            uploadMessage = L10n.t("工作站已完整接收，记录页可查看处理进度", "Workstation received all frames; see Records for processing") + " · " + receipt.session
        } catch {
            uploadError = (task.isCancelled
                ? L10n.t("上传已暂停，点击上传可续传", "Upload paused; tap Upload to resume")
                : error.localizedDescription)
        }
    }

    func cancelUpload() { uploadTask?.cancel() }

    func discardRecording() {
        guard !isUploading, !isSavingRecording, !isStreaming, let store = localStore else { return }
        do {
            try LocalPacketStore.deleteRecording(directory: store.directory)
            localStore = nil
            uploadError = nil
            recoverPendingRecording()
        } catch { uploadError = error.localizedDescription }
    }

}

private struct StreamCallbacks {
    let status: @MainActor (String) -> Void
    let encoded: @MainActor (UInt64) -> Void
    let skipped: @MainActor (UInt64) -> Void
    let error: @MainActor (String?) -> Void
    let preview: @MainActor (UIImage?, UIImage?, DepthPreviewStats, CGPoint) -> Void
}

/// One-shot latch so concurrent connection events fire the continuation once.
private final class TimeoutBox: @unchecked Sendable {
    private let lock = NSLock()
    private var used = false
    func hit() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if used { return false }
        used = true
        return true
    }
}

// Retained as protocol v1 compatibility, even though the app is backend-neutral.
private let wireMagic = Data("SGFIPD01".utf8)

private struct CRCHeader: Codable {
    let color: UInt32
    let depth: UInt32
    let confidence: UInt32
}

private struct IMUSampleHeader: Codable {
    let timestamp_ns: UInt64
    let kind: String
    let x: Double
    let y: Double
    let z: Double
}

private struct FrameHeader: Codable {
    let schema_version: Int
    let session_id: String
    let frame_id: UInt64
    let timestamp_ns: UInt64
    let tracking_state: String
    let image_orientation: String
    let color_width: Int
    let color_height: Int
    let depth_width: Int
    let depth_height: Int
    let intrinsics_reference_width: Int
    let intrinsics_reference_height: Int
    let camera_intrinsics: [Float]
    let arkit_camera_to_world_m: [Float]
    let imu_samples: [IMUSampleHeader]
    let color_encoding: String
    let depth_encoding: String
    let confidence_encoding: String
    let color_bytes: Int
    let depth_bytes: Int
    let confidence_bytes: Int
    let crc32: CRCHeader
}

private final class MotionSampler {
    private let manager = CMMotionManager()
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "scan.coremotion"
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInteractive
        return queue
    }()
    private let lock = NSLock()
    private var samples: [IMUSampleHeader] = []

    func start() throws {
        guard manager.isDeviceMotionAvailable else {
            throw NSError(
                domain: "Scan.iPad",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Core Motion device motion is unavailable"]
            )
        }
        manager.deviceMotionUpdateInterval = 1.0 / 100.0
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) {
            [weak self] motion, _ in
            guard let self, let motion, motion.timestamp >= 0 else { return }
            let timestampNS = UInt64((motion.timestamp * 1_000_000_000.0).rounded())

            // Core Motion uses device-fixed axes. ARKit's camera axes are
            // fixed to landscape-left sensor orientation: camera X = -device
            // Y, camera Y = device X, and camera Z = device Z.
            func cameraAxes(x: Double, y: Double, z: Double) -> (Double, Double, Double) {
                (-y, x, z)
            }

            let gravity = cameraAxes(
                x: motion.gravity.x * 9.80665,
                y: motion.gravity.y * 9.80665,
                z: motion.gravity.z * 9.80665
            )
            let gyro = cameraAxes(
                x: motion.rotationRate.x,
                y: motion.rotationRate.y,
                z: motion.rotationRate.z
            )
            let current = [
                IMUSampleHeader(
                    timestamp_ns: timestampNS,
                    kind: "accelerometer",
                    x: gravity.0,
                    y: gravity.1,
                    z: gravity.2
                ),
                IMUSampleHeader(
                    timestamp_ns: timestampNS,
                    kind: "gyroscope",
                    x: gyro.0,
                    y: gyro.1,
                    z: gyro.2
                ),
            ]
            self.lock.lock()
            self.samples.append(contentsOf: current)
            if self.samples.count > 2_048 {
                self.samples.removeFirst(self.samples.count - 2_048)
            }
            self.lock.unlock()
        }
    }

    func snapshot(upTo timestampNS: UInt64) -> [IMUSampleHeader] {
        lock.lock()
        defer { lock.unlock() }
        return Array(samples.filter { $0.timestamp_ns <= timestampNS }.suffix(512))
    }

    func commit(upTo timestampNS: UInt64) {
        lock.lock()
        samples.removeAll { $0.timestamp_ns <= timestampNS }
        lock.unlock()
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
        queue.cancelAllOperations()
        lock.lock()
        samples.removeAll(keepingCapacity: false)
        lock.unlock()
    }
}

private extension Data {
    mutating func appendBigEndian(_ value: UInt32) {
        var encoded = value.bigEndian
        Swift.withUnsafeBytes(of: &encoded) { append(contentsOf: $0) }
    }

    mutating func appendLittleEndian(_ value: UInt32) {
        var encoded = value.littleEndian
        Swift.withUnsafeBytes(of: &encoded) { append(contentsOf: $0) }
    }
}

private let crcTable: [UInt32] = {
    (0...255).map { i -> UInt32 in
        var c = UInt32(i)
        for _ in 0..<8 { c = (c >> 1) ^ (0xedb8_8320 & (0 &- (c & 1))) }
        return c
    }
}()

private func crc32(_ data: Data) -> UInt32 {
    var crc: UInt32 = 0xffff_ffff
    for byte in data {
        crc = crcTable[Int((crc ^ UInt32(byte)) & 0xff)] ^ (crc >> 8)
    }
    return ~crc
}

private func rowMajor(_ matrix: simd_float4x4) -> [Float] {
    (0..<4).flatMap { row in (0..<4).map { column in matrix[column][row] } }
}

private func rowMajor(_ matrix: simd_float3x3) -> [Float] {
    (0..<3).flatMap { row in (0..<3).map { column in matrix[column][row] } }
}

private func floatDepthData(_ buffer: CVPixelBuffer) -> Data? {
    guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_DepthFloat32 else {
        return nil
    }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    var output = Data(capacity: width * height * 4)
    for row in 0..<height {
        let values = base.advanced(by: row * stride).assumingMemoryBound(to: Float.self)
        for column in 0..<width {
            output.appendLittleEndian(values[column].bitPattern)
        }
    }
    return output
}

private func confidenceData(_ buffer: CVPixelBuffer?) -> Data {
    guard let buffer else { return Data() }
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddress(buffer) else { return Data() }
    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    var output = Data(capacity: width * height)
    for row in 0..<height {
        output.append(
            base.advanced(by: row * stride).assumingMemoryBound(to: UInt8.self),
            count: width
        )
    }
    return output
}

private final class TCPFrameSender {
    private let queue = DispatchQueue(label: "sgf.ipad.tcp")
    private let host: NWEndpoint.Host
    private let port: NWEndpoint.Port
    private let callbacks: StreamCallbacks
    private var connection: NWConnection?
    private var connectionGeneration: UInt64 = 0
    private var ready = false
    private var sendInFlight = false
    private var stopping = false
    private var reconnectScheduled = false

    init(host: String, port: UInt16, callbacks: StreamCallbacks) {
        self.host = NWEndpoint.Host(host)
        self.port = NWEndpoint.Port(rawValue: port)!
        self.callbacks = callbacks
        queue.async { [weak self] in
            self?.startConnection()
        }
    }

    private func startConnection() {
        guard !stopping, connection == nil else { return }
        connectionGeneration &+= 1
        let generation = connectionGeneration
        let candidate = NWConnection(host: host, port: port, using: .tcp)
        connection = candidate
        candidate.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            self.queue.async {
                self.handle(state, generation: generation)
            }
        }
        candidate.start(queue: queue)
    }

    private func handle(_ state: NWConnection.State, generation: UInt64) {
        guard !stopping, generation == connectionGeneration else { return }
        switch state {
        case .ready:
            ready = true
            Task { @MainActor in
                callbacks.status("Connected")
                callbacks.error(nil)
            }
        case .waiting(let error):
            ready = false
            Task { @MainActor in
                callbacks.status("Waiting for workstation")
                callbacks.error(error.localizedDescription)
            }
            // Network.framework parks refused connections in .waiting without
            // retrying promptly; force a fresh connection after 3 s.
            queue.asyncAfter(deadline: .now() + 3.0) { [weak self] in
                guard let self, !self.stopping, generation == self.connectionGeneration,
                      !self.ready else { return }
                self.replaceFailedConnection()
            }
        case .failed(let error):
            Task { @MainActor in
                callbacks.status("Reconnecting")
                callbacks.error(error.localizedDescription)
            }
            replaceFailedConnection()
        case .cancelled:
            replaceFailedConnection()
        default:
            ready = false
        }
    }

    private func replaceFailedConnection() {
        guard !stopping else { return }
        ready = false
        sendInFlight = false
        connection?.stateUpdateHandler = nil
        connection?.cancel()
        connection = nil
        guard !reconnectScheduled else { return }
        reconnectScheduled = true
        queue.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self else { return }
            self.reconnectScheduled = false
            self.startConnection()
        }
    }

    func sendLatest(_ packet: Data) -> Bool {
        var accepted = false
        queue.sync {
            guard ready, !sendInFlight, let connection else { return }
            let generation = connectionGeneration
            sendInFlight = true
            accepted = true
            connection.send(content: packet, completion: .contentProcessed { [weak self] error in
                guard let self else { return }
                self.queue.async {
                    guard generation == self.connectionGeneration else { return }
                    self.sendInFlight = false
                    if let error, !self.stopping {
                        Task { @MainActor in
                            self.callbacks.status("Reconnecting")
                            self.callbacks.error(error.localizedDescription)
                        }
                        self.replaceFailedConnection()
                    }
                }
            })
        }
        return accepted
    }

    func stop() {
        queue.async { [weak self] in
            guard let self else { return }
            self.stopping = true
            self.connectionGeneration &+= 1
            self.ready = false
            self.sendInFlight = false
            self.connection?.stateUpdateHandler = nil
            self.connection?.cancel()
            self.connection = nil
        }
    }
}

private enum FrameEncodingResult {
    case packet(Data, motionTimestampNS: UInt64)
    case unavailable
    case failed(String)
}

private final class ARKitFrameStreamer: NSObject, ARSessionDelegate {
    let session = ARSession()
    private let sender: TCPFrameSender
    private let localStore: LocalPacketStore?
    private let callbacks: StreamCallbacks
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let motionSampler = MotionSampler()
    private let encodeQueue = DispatchQueue(label: "sgf.ipad.encode")
    private let sessionID = UUID().uuidString
    private let encodingLock = NSLock()
    private var frameID: UInt64 = 0
    private var encodedFrames: UInt64 = 0
    private var skippedFrames: UInt64 = 0
    private var encoding = false
    private var acceptingFrames = true
    private var pendingFrames = 0
    private var lastPreviewTimestamp: TimeInterval = 0
    // Directional coverage: yaw x pitch bins over the sphere, from valid depth rays.
    private let covW = 48, covH = 24
    private var coverageGrid: [Float]?

    /// Downsampled live depth preview + guidance stats, throttled to ~3 Hz.
    private func makePreview(frame: ARFrame) {
        guard let sceneDepth = frame.sceneDepth else { return }
        let buffer = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let rowStride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.size

        let cols = 64, rows = 48
        var rgba = [UInt8](repeating: 0, count: cols * rows * 4)
        var validValues: [Double] = []
        validValues.reserveCapacity(cols * rows)
        let stepX = max(1, width / cols)
        let stepY = max(1, height / rows)
        let values = base.assumingMemoryBound(to: Float.self)
        for row in 0..<rows {
            let sy = min(height - 1, row * stepY)
            for col in 0..<cols {
                let sx = min(width - 1, col * stepX)
                let depth = values[sy * rowStride + sx]
                let offset = (row * cols + col) * 4
                if depth.isFinite && depth > 0.2 && depth < 6.0 {
                    validValues.append(Double(depth))
                    // near (0.4 m) = green, mid (2 m) = yellow, far (4 m) = red
                    let t = min(1, max(0, (Double(depth) - 0.4) / 3.6))
                    var r = 0.0, g = 1.0, b = 0.0
                    if t < 0.5 {
                        r = t * 2; g = 1.0
                    } else {
                        r = 1.0; g = (1 - t) * 2
                    }
                    rgba[offset] = UInt8(r * 255)
                    rgba[offset + 1] = UInt8(g * 255)
                    rgba[offset + 2] = UInt8(b * 255)
                    rgba[offset + 3] = 255
                } else {
                    rgba[offset] = 40
                    rgba[offset + 1] = 44
                    rgba[offset + 2] = 52
                    rgba[offset + 3] = 255
                }
            }
        }
        var stats = DepthPreviewStats()
        stats.validFraction = validValues.isEmpty ? 0 : Double(validValues.count) / Double(cols * rows)
        if !validValues.isEmpty {
            validValues.sort()
            stats.medianDepthM = validValues[validValues.count / 2]
        }
        switch frame.camera.trackingState {
        case .normal: stats.tracking = "normal"
        case .limited: stats.tracking = "limited"
        case .notAvailable: stats.tracking = "not_available"
        }
        var image: UIImage?
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        if let context = CGContext(data: &rgba, width: cols, height: rows, bitsPerComponent: 8,
                                   bytesPerRow: cols * 4, space: colorSpace,
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
           let cgImage = context.makeImage() {
            image = UIImage(cgImage: cgImage, scale: 1, orientation: PreviewStatsSampler.piOrientation())
        }

        // ---- directional coverage accumulation + strip rendering ----
        var coverage: UIImage?
        if coverageGrid == nil { coverageGrid = [Float](repeating: 0, count: covW * covH) }
        if var grid = coverageGrid {
            let intr = frame.camera.intrinsics
            let imgW = Float(CVPixelBufferGetWidth(frame.capturedImage))
            let imgH = Float(CVPixelBufferGetHeight(frame.capturedImage))
            let fx = intr[0][0] * Float(width) / imgW
            let fy = intr[1][1] * Float(height) / imgH
            let cx = intr[2][0] * Float(width) / imgW
            let cy = intr[2][1] * Float(height) / imgH
            let cam = frame.camera.transform
            let rotation = simd_float3x3(columns: (SIMD3(cam.columns.0.x, cam.columns.0.y, cam.columns.0.z),
                                                   SIMD3(cam.columns.1.x, cam.columns.1.y, cam.columns.1.z),
                                                   SIMD3(cam.columns.2.x, cam.columns.2.y, cam.columns.2.z)))
            for py in stride(from: 0, to: height, by: 16) {
                for px in stride(from: 0, to: width, by: 16) {
                    let depth = values[py * rowStride + px]
                    guard depth.isFinite, depth > 0.25, depth < 5.5 else { continue }
                    let dir = simd_normalize(SIMD3<Float>((Float(px) - cx) / fx,
                                                         -(Float(py) - cy) / fy, -1))
                    let world = rotation * dir
                    let yaw = atan2(world.x, -world.z)
                    let pitch = asin(max(-1, min(1, world.y)))
                    let xi = min(covW - 1, max(0, Int((yaw / (2 * .pi) + 0.5) * Float(covW))))
                    let yi = min(covH - 1, max(0, Int((pitch / .pi + 0.5) * Float(covH))))
                    grid[yi * covW + xi] += 1
                }
            }
            // camera forward direction as a cursor on the strip
            let forward = rotation * SIMD3<Float>(0, 0, -1)
            let fYaw = atan2(forward.x, -forward.z)
            var cursor = Int((fYaw / (2 * .pi) + 0.5) * Float(covW))
            cursor = ((cursor % covW) + covW) % covW

            var coveredBins = 0
            var cov = [UInt8](repeating: 0, count: covW * covH * 4)
            for row in 0..<covH {
                for col in 0..<covW {
                    let count = grid[row * covW + col]
                    if count > 0 { coveredBins += 1 }
                    let t = min(1, count / 8)
                    let offset = (row * covW + col) * 4
                    if col == cursor {
                        cov[offset] = 255; cov[offset + 1] = 255; cov[offset + 2] = 255
                    } else {
                        cov[offset] = UInt8(12 + 40 * t)
                        cov[offset + 1] = UInt8(20 + 215 * t)
                        cov[offset + 2] = UInt8(30 + 100 * t)
                    }
                    cov[offset + 3] = 255
                }
            }
            coverageGrid = grid
            stats.coveragePercent = Double(coveredBins) / Double(covW * covH)
            if let context = CGContext(data: &cov, width: covW, height: covH, bitsPerComponent: 8,
                                       bytesPerRow: covW * 4, space: colorSpace,
                                       bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
               let cgImage = context.makeImage() {
                coverage = UIImage(cgImage: cgImage)
            }
        }
        Task { @MainActor in
            let t = frame.camera.transform.columns.3
            callbacks.preview(image, coverage, stats, CGPoint(x: Double(t.x), y: Double(t.z)))
        }
    }

    init(serverHost: String, serverPort: UInt16, callbacks: StreamCallbacks,
         localStore: LocalPacketStore? = nil) {
        self.callbacks = callbacks
        self.localStore = localStore
        sender = TCPFrameSender(host: serverHost, port: serverPort, callbacks: callbacks)
        super.init()
        session.delegate = self
    }

    func start() throws {
        let configuration = ARWorldTrackingConfiguration()
        guard ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            throw NSError(
                domain: "Scan.iPad",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "This iPad does not expose LiDAR scene depth"]
            )
        }
        try motionSampler.start()
        configuration.frameSemantics.insert(.sceneDepth)
        configuration.worldAlignment = .gravity
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
        Task { @MainActor in callbacks.status("Starting ARKit") }
    }

    func stop(completion: @escaping @MainActor () -> Void) {
        session.pause()
        session.delegate = nil
        encodingLock.lock()
        acceptingFrames = false
        // Enqueued while holding the admission lock: all accepted frames precede this barrier.
        encodeQueue.async { [self] in
            localStore?.seal()
            motionSampler.stop()
            sender.stop()
            Task { @MainActor in completion() }
        }
        encodingLock.unlock()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if frame.timestamp - lastPreviewTimestamp >= 0.33 {
            lastPreviewTimestamp = frame.timestamp
            makePreview(frame: frame)
        }
        encodingLock.lock()
        guard acceptingFrames else { encodingLock.unlock(); return }
        if localStore != nil && pendingFrames >= 12 {
            acceptingFrames = false
            encodingLock.unlock()
            Task { @MainActor in
                self.callbacks.error(L10n.t("存储速度不足，已停止并保存已接收帧", "Storage cannot keep up; stopped and saving accepted frames"))
            }
            return
        }
        if localStore == nil {
            guard !encoding else {
                skippedFrames += 1
                let current = skippedFrames
                encodingLock.unlock()
                Task { @MainActor in callbacks.skipped(current) }
                return
            }
        }
        encoding = true
        let currentFrameID = frameID
        frameID += 1
        pendingFrames += 1
        encodeQueue.async { [self] in
            defer {
                self.encodingLock.lock()
                self.encoding = false
                self.pendingFrames -= 1
                self.encodingLock.unlock()
            }
            switch self.encode(frame: frame, frameID: currentFrameID) {
            case .packet(let packet, let motionTimestampNS):
                if let store = self.localStore {
                    do {
                        try store.write(packet)
                    } catch {
                        Task { @MainActor in
                            self.callbacks.error(L10n.t("录制写入失败，原始文件已保留：", "Recording write failed; originals retained: ") + error.localizedDescription)
                        }
                        return
                    }
                    self.motionSampler.commit(upTo: motionTimestampNS)
                    self.encodedFrames += 1
                    let current = self.encodedFrames
                    Task { @MainActor in
                        self.callbacks.encoded(current)
                        self.callbacks.error(nil)
                    }
                    return
                }
                if self.sender.sendLatest(packet) {
                    self.motionSampler.commit(upTo: motionTimestampNS)
                    self.encodedFrames += 1
                    let current = self.encodedFrames
                    Task { @MainActor in
                        self.callbacks.encoded(current)
                        self.callbacks.error(nil)
                    }
                } else {
                    self.skippedFrames += 1
                    let current = self.skippedFrames
                    Task { @MainActor in self.callbacks.skipped(current) }
                }
                return
            case .unavailable:
                self.skippedFrames += 1
                let current = self.skippedFrames
                Task { @MainActor in self.callbacks.skipped(current) }
                return
            case .failed(let reason):
                self.skippedFrames += 1
                let current = self.skippedFrames
                Task { @MainActor in
                    self.callbacks.skipped(current)
                    self.callbacks.error(reason)
                }
                return
            }
        }
        encodingLock.unlock()
    }

    private func encode(frame: ARFrame, frameID: UInt64) -> FrameEncodingResult {
        // ARKit can legitimately omit sceneDepth while tracking warms up. This
        // is a dropped sample, not a fatal encoder or network error.
        guard let sceneDepth = frame.sceneDepth else { return .unavailable }
        let colorBuffer = frame.capturedImage
        let colorImage = CIImage(cvPixelBuffer: colorBuffer)
        guard let colorCG = ciContext.createCGImage(colorImage, from: colorImage.extent),
              let color = UIImage(cgImage: colorCG).jpegData(compressionQuality: 0.88) else {
            return .failed("Camera RGB JPEG encoding failed")
        }
        guard let depth = floatDepthData(sceneDepth.depthMap) else {
            return .failed("ARKit sceneDepth is not a Float32 depth map")
        }
        let confidence = confidenceData(sceneDepth.confidenceMap)
        let tracking: String
        switch frame.camera.trackingState {
        case .normal: tracking = "normal"
        case .limited: tracking = "limited"
        case .notAvailable: tracking = "not_available"
        }
        let timestampNS = UInt64((frame.timestamp * 1_000_000_000.0).rounded())
        let motionSamples = motionSampler.snapshot(upTo: timestampNS)
        let header = FrameHeader(
            schema_version: 1,
            session_id: sessionID,
            frame_id: frameID,
            timestamp_ns: timestampNS,
            tracking_state: tracking,
            image_orientation: "sensor_native",
            color_width: CVPixelBufferGetWidth(colorBuffer),
            color_height: CVPixelBufferGetHeight(colorBuffer),
            depth_width: CVPixelBufferGetWidth(sceneDepth.depthMap),
            depth_height: CVPixelBufferGetHeight(sceneDepth.depthMap),
            intrinsics_reference_width: Int(frame.camera.imageResolution.width),
            intrinsics_reference_height: Int(frame.camera.imageResolution.height),
            camera_intrinsics: rowMajor(frame.camera.intrinsics),
            arkit_camera_to_world_m: rowMajor(frame.camera.transform),
            imu_samples: motionSamples,
            color_encoding: "jpeg",
            depth_encoding: "float32_le_meters",
            confidence_encoding: "uint8",
            color_bytes: color.count,
            depth_bytes: depth.count,
            confidence_bytes: confidence.count,
            crc32: CRCHeader(
                color: crc32(color),
                depth: crc32(depth),
                confidence: crc32(confidence)
            )
        )
        guard let headerData = try? JSONEncoder().encode(header),
              headerData.count <= 64 * 1024 else {
            return .failed("Frame metadata encoding failed")
        }
        var packet = Data(
            capacity: 12 + headerData.count + color.count + depth.count + confidence.count
        )
        packet.append(wireMagic)
        packet.appendBigEndian(UInt32(headerData.count))
        packet.append(headerData)
        packet.append(color)
        packet.append(depth)
        packet.append(confidence)
        return .packet(packet, motionTimestampNS: timestampNS)
    }
}


/// Depth preview + guidance stats for the pre-start camera preview.
private final class PreviewStatsSampler: NSObject, ARSessionDelegate {
    private let update: @MainActor (UIImage?, UIImage?, DepthPreviewStats, CGPoint) -> Void
    private var lastTimestamp: TimeInterval = 0

    init(update: @escaping @MainActor (UIImage?, UIImage?, DepthPreviewStats, CGPoint) -> Void) {
        self.update = update
        super.init()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard frame.timestamp - lastTimestamp >= 0.33, let sceneDepth = frame.sceneDepth else { return }
        lastTimestamp = frame.timestamp
        let buffer = sceneDepth.depthMap
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer)
        let rowStride = CVPixelBufferGetBytesPerRow(buffer) / MemoryLayout<Float>.size
        let values = base.assumingMemoryBound(to: Float.self)

        let cols = 64, rows = 48
        var rgba = [UInt8](repeating: 0, count: cols * rows * 4)
        var valid: [Double] = []
        for row in 0..<rows {
            let sy = min(height - 1, row * height / rows)
            for col in 0..<cols {
                let sx = min(width - 1, col * width / cols)
                let depth = values[sy * rowStride + sx]
                let offset = (row * cols + col) * 4
                if depth.isFinite, depth > 0.2, depth < 6.0 {
                    valid.append(Double(depth))
                    let t = min(1, max(0, (Double(depth) - 0.4) / 3.6))
                    var r = 0.0, g = 1.0
                    if t < 0.5 { r = t * 2 } else { g = (1 - t) * 2 }
                    rgba[offset] = UInt8(r * 255)
                    rgba[offset + 1] = UInt8(g * 255)
                    rgba[offset + 2] = 0
                } else {
                    rgba[offset] = 40; rgba[offset + 1] = 44; rgba[offset + 2] = 52
                }
                rgba[offset + 3] = 255
            }
        }
        var stats = DepthPreviewStats()
        stats.validFraction = valid.isEmpty ? 0 : Double(valid.count) / Double(cols * rows)
        if !valid.isEmpty {
            valid.sort()
            stats.medianDepthM = valid[valid.count / 2]
        }
        switch frame.camera.trackingState {
        case .normal: stats.tracking = "normal"
        case .limited: stats.tracking = "limited"
        case .notAvailable: stats.tracking = "not_available"
        }
        var image: UIImage?
        if let context = CGContext(data: &rgba, width: cols, height: rows, bitsPerComponent: 8,
                                   bytesPerRow: cols * 4, space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
           let cgImage = context.makeImage() {
            image = UIImage(cgImage: cgImage, scale: 1, orientation: PreviewStatsSampler.piOrientation())
        }
        let t = frame.camera.transform.columns.3
        Task { @MainActor in
            self.update(image, nil, stats, CGPoint(x: Double(t.x), y: Double(t.z)))
        }
    }
}


extension PreviewStatsSampler {
    /// Rotate the sensor-native (landscape) preview upright for the current UI orientation.
    static func piOrientation() -> UIImage.Orientation {
        let orientation = UIApplication.shared.connectedScenes
            .compactMap { ($0 as? UIWindowScene)?.interfaceOrientation }
            .first ?? .landscapeRight
        switch orientation {
        case .portrait: return .right
        case .portraitUpsideDown: return .left
        case .landscapeLeft: return .down
        case .landscapeRight: return .up
        @unknown default: return .up
        }
    }
}

