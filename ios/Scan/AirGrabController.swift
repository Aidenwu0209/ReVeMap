import AVFoundation
import Combine
import SwiftUI
import Vision

/// Frames stay on-device. Only normalized hand coordinates reach the WebGL viewer.
@MainActor
final class AirGrabController: ObservableObject {
    enum Mode: String { case move, rotate }
    @Published private(set) var mode: Mode = .move
    @Published private(set) var enabled = false
    @Published private(set) var starting = false
    @Published private(set) var status = AirGrabGesture.Phase.searching
    @Published var error: String?
    let packets = PassthroughSubject<AirGrabPacket, Never>()
    private(set) var sessionID = UUID().uuidString
    private var sequence = 0
    private var generation = 0
    private var orientation: UIInterfaceOrientation = .landscapeRight
    private let worker = AirGrabCameraWorker()
    private var lease: CaptureController?
    private var stopping = false
    private var lastFrame = Date.distantPast
    private var watchdog: Task<Void, Never>?

    func start(capture: CaptureController) {
        guard !enabled, !starting, !stopping else { return }
        guard !capture.isStreaming, !capture.isSavingRecording, !capture.handCameraInUse else {
            error = L10n.t("请先结束扫描，再开启隔空操控。", "Finish scanning before enabling Air Grab.")
            return
        }
        if capture.isPreviewing { capture.stopPreview() }
        capture.handCameraInUse = true
        lease = capture
        sessionID = UUID().uuidString
        let requestID = sessionID
        generation += 1
        enabled = true
        starting = true
        status = .searching
        error = nil
        Task { [weak self] in
            let allowed: Bool
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized: allowed = true
            case .notDetermined: allowed = await AVCaptureDevice.requestAccess(for: .video)
            default: allowed = false
            }
            guard let self, self.enabled, self.sessionID == requestID else { return }
            guard allowed else {
                self.error = L10n.t("请在系统设置中允许 Scan 使用相机，以识别手势。", "Allow camera access for Scan in Settings to recognize hand gestures.")
                self.stop()
                return
            }
            self.lastFrame = Date()
            self.worker.start(orientation: self.orientation, generation: self.generation, event: { [weak self] event, generation in
                Task { @MainActor in
                    guard let self, self.enabled, self.sessionID == requestID, self.generation == generation else { return }
                    self.lastFrame = Date()
                    if self.starting { self.starting = false }
                    self.emit(event)
                }
            }, failure: { [weak self] message in
                Task { @MainActor in
                    guard let self, self.sessionID == requestID, self.enabled else { return }
                    self.error = message
                    self.stop()
                }
            })
            self.watchdog = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .milliseconds(400))
                    guard !Task.isCancelled, let self, self.enabled else { return }
                    if Date().timeIntervalSince(self.lastFrame) > 1.2 {
                        self.rearm()
                        self.emit(.init(phase: .lost, point: nil))
                    }
                    if Date().timeIntervalSince(self.lastFrame) > 8 {
                        self.error = L10n.t("相机画面已中断，请重新开启隔空操控。", "Camera frames stopped. Turn Air Grab on again.")
                        self.stop()
                        return
                    }
                }
            }
        }
    }

    func rearm() {
        guard enabled else { return }
        generation += 1
        worker.rearm(generation: generation)
        emit(.init(phase: .reset, point: nil))
    }

    func setMode(_ value: Mode) {
        guard mode != value else { return }
        mode = value
        rearm()
    }

    func stop() {
        guard enabled || starting else { return }
        emit(.init(phase: .reset, point: nil))
        enabled = false
        starting = false
        sessionID = UUID().uuidString // Reject callbacks from a prior capture or permission request.
        watchdog?.cancel()
        watchdog = nil
        stopping = true
        let owner = lease
        lease = nil
        worker.stop { [weak self] in
            Task { @MainActor in
                owner?.handCameraInUse = false
                self?.stopping = false
            }
        }
    }

    func updateOrientation(_ value: UIInterfaceOrientation) {
        guard value != .unknown, value != orientation else { return }
        orientation = value
        generation += 1
        worker.updateOrientation(value, generation: generation)
        if enabled { emit(.init(phase: .reset, point: nil)) }
    }

    private func emit(_ event: AirGrabGesture.Event) {
        if event.phase != .uncertain, status != event.phase { status = event.phase }
        sequence += 1
        packets.send(AirGrabPacket(session: sessionID, sequence: sequence, phase: event.phase, mode: mode.rawValue,
                                   x: event.point.map { Double($0.x) }, y: event.point.map { Double($0.y) }))
        if [.began, .ended, .lost, .reset].contains(event.phase) {
            worker.recordBridge("native_" + event.phase.rawValue, generation: generation, mode: mode.rawValue, sequence: sequence)
        }
    }

    func viewerDidGrab() {
        guard enabled else { return }
        worker.recordBridge("viewer_grabbed", generation: generation, mode: mode.rawValue, sequence: sequence)
    }

    var instruction: String {
        if starting { return L10n.t("正在开启前置摄像头…", "Starting front camera…") }
        switch status {
        case .searching, .lost, .reset, .uncertain: return L10n.t("让拇指和食指进入前置镜头，先张开再捏合", "Show your thumb and index finger; open them, then pinch")
        case .needsOpen: return L10n.t("先张开拇指和食指，再捏合抓取", "Open thumb and index finger, then pinch to grab")
        case .ready: return mode == .rotate
            ? L10n.t("捏合旋转 · 左右转向，上下俯仰", "Pinch to rotate · Move left/right or up/down")
            : L10n.t("捏合抓起 · 移动跟随 · 松手悬停", "Pinch to lift · Move to carry · Release to hold")
        case .began, .changed: return mode == .rotate
            ? L10n.t("正在旋转 · 松手保持当前角度", "Rotating · Release to keep this angle")
            : L10n.t("已抓住 · 移动手指，松开后悬停", "Holding · Move your hand; release to hold in place")
        case .ended: return mode == .rotate
            ? L10n.t("角度已保持 · 再次捏合继续旋转", "Angle held · Pinch again to rotate")
            : L10n.t("已悬停 · 再次捏合可继续移动", "Object held in place · Pinch again to move")
        }
    }
}

/// All AVCaptureSession and Vision operations use one serial queue.
private final class AirGrabCameraWorker: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let queue = DispatchQueue(label: "Scan.AirGrab.camera", qos: .userInitiated)
    private let session = AVCaptureSession()
    private let request = VNDetectHumanHandPoseRequest()
    private var detector = AirGrabGesture()
    private var output: AVCaptureVideoDataOutput?
    private var event: ((AirGrabGesture.Event, Int) -> Void)?
    private var generation = 0
    private var failure: ((String) -> Void)?
    private var observers: [NSObjectProtocol] = []
    private var cadence = AirGrabFrameCadence()
    private var active = false
    private var visionFailures = 0
    private var diagnosticFrames: [[String: Any]] = []
    private var diagnosticCounts: [String: Int] = [:]
    private var diagnosticPhases: [String: Int] = [:]
    private var diagnosticBridgeCounts: [String: Int] = [:]
    private var diagnosticBridgeEvents: [[String: Any]] = []
    private var diagnosticCamera: [String: Any] = [:]
    private var diagnosticLastWrite = -Double.infinity
    private var diagnosticLastFrame: Double?
    private let diagnosticQueue = DispatchQueue(label: "Scan.AirGrab.diagnostics", qos: .utility)

    func start(orientation: UIInterfaceOrientation, generation: Int, event: @escaping (AirGrabGesture.Event, Int) -> Void,
               failure: @escaping (String) -> Void) {
        queue.async { [self] in
            self.generation = generation
            self.event = event
            self.failure = failure
            detector.reset()
            visionFailures = 0
            cadence.reset()
            diagnosticFrames = []; diagnosticCounts = [:]; diagnosticPhases = [:]
            diagnosticBridgeCounts = [:]; diagnosticBridgeEvents = []
            diagnosticCamera = [:]
            diagnosticLastWrite = -.infinity; diagnosticLastFrame = nil
            do {
                session.beginConfiguration()
                defer { session.commitConfiguration() }
                session.sessionPreset = .vga640x480
                guard let camera = AVCaptureDevice.DiscoverySession(
                    deviceTypes: [.builtInTrueDepthCamera, .builtInWideAngleCamera, .builtInUltraWideCamera],
                    mediaType: .video, position: .front).devices.first else {
                    throw CameraFailure.unavailable
                }
                let input = try AVCaptureDeviceInput(device: camera)
                let dimensions = CMVideoFormatDescriptionGetDimensions(camera.activeFormat.formatDescription)
                diagnosticCamera = ["name": camera.localizedName, "type": camera.deviceType.rawValue,
                    "formatWidth": dimensions.width, "formatHeight": dimensions.height,
                    "fieldOfView": camera.activeFormat.videoFieldOfView, "zoom": camera.videoZoomFactor]
                guard session.canAddInput(input) else { throw CameraFailure.unavailable }
                session.addInput(input)
                let video = AVCaptureVideoDataOutput()
                video.alwaysDiscardsLateVideoFrames = true
                video.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
                video.setSampleBufferDelegate(self, queue: queue)
                guard session.canAddOutput(video) else { throw CameraFailure.unavailable }
                session.addOutput(video)
                output = video
                configureOrientation(orientation)
                request.maximumHandCount = 2 // Reject ambiguous multi-hand frames instead of switching hands.
            } catch {
                tearDown()
                failure(L10n.t("无法开启前置相机，请确认相机未被占用。", "Front camera unavailable. Check whether another camera session is active."))
                return
            }
            for name in [AVCaptureSession.wasInterruptedNotification, AVCaptureSession.runtimeErrorNotification] {
                observers.append(NotificationCenter.default.addObserver(forName: name, object: session, queue: nil) { [weak self] _ in
                    self?.queue.async { [weak self] in
                        guard let self, self.active else { return }
                        self.failure?(L10n.t("相机已中断，请重新开启隔空操控。", "Camera interrupted. Turn Air Grab on again."))
                    }
                })
            }
            active = true
            session.startRunning()
            if !session.isRunning { failure(L10n.t("前置相机未能启动，请重试。", "Front camera could not start. Please retry.")) }
        }
    }

    func stop(completion: @escaping () -> Void) {
        queue.async { [self] in tearDown(); completion() }
    }

    func recordBridge(_ name: String, generation: Int, mode: String, sequence: Int) {
        queue.async { [self] in
            guard active, generation == self.generation else { return }
            diagnosticBridgeCounts[name, default: 0] += 1
            diagnosticBridgeEvents.append(["event": name, "time": Date().timeIntervalSince1970,
                                           "mode": mode, "sequence": sequence])
            if diagnosticBridgeEvents.count > 100 { diagnosticBridgeEvents.removeFirst() }
            writeDiagnostics(force: true)
        }
    }

    func rearm(generation: Int) { queue.async { [self] in self.generation = generation; detector.reset() } }

    func updateOrientation(_ orientation: UIInterfaceOrientation, generation: Int) {
        queue.async { [self] in self.generation = generation; configureOrientation(orientation); detector.reset() }
    }

    private func configureOrientation(_ orientation: UIInterfaceOrientation) {
        diagnosticCamera["interfaceOrientation"] = orientation.rawValue
        guard let connection = output?.connection(with: .video) else { return }
        // AVCapture resolves the physical sensor orientation (including landscape iPad cameras).
        let value: AVCaptureVideoOrientation
        switch orientation {
        case .portrait: value = .portrait
        case .portraitUpsideDown: value = .portraitUpsideDown
        case .landscapeLeft: value = .landscapeLeft
        default: value = .landscapeRight
        }
        if connection.isVideoOrientationSupported { connection.videoOrientation = value }
        if connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = true
        }
        diagnosticCamera["connectionOrientation"] = connection.videoOrientation.rawValue
        diagnosticCamera["mirrored"] = connection.isVideoMirrored
    }

    private func tearDown() {
        writeDiagnostics(force: true)
        active = false
        if session.isRunning { session.stopRunning() }
        observers.forEach(NotificationCenter.default.removeObserver)
        observers.removeAll()
        output?.setSampleBufferDelegate(nil, queue: nil)
        session.beginConfiguration()
        session.inputs.forEach(session.removeInput)
        session.outputs.forEach(session.removeOutput)
        session.commitConfiguration()
        output = nil
        event = nil
        failure = nil
        detector.reset()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard active else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        guard cadence.accepts(time) else { return }
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let started = ProcessInfo.processInfo.systemUptime
        var diagnostic: [String: Any] = ["time": Date().timeIntervalSince1970,
            "width": CVPixelBufferGetWidth(pixel), "height": CVPixelBufferGetHeight(pixel),
            "frameGapMs": diagnosticLastFrame.map { (started - $0) * 1000 } ?? 0]
        diagnosticLastFrame = started
        func publish(_ sample: AirGrabGesture.Sample?, reason: String) {
            let result = detector.process(sample)
            diagnostic["reason"] = reason
            diagnostic["phase"] = result.phase.rawValue
            diagnostic["holding"] = detector.pinching
            diagnostic["rawCenter"] = sample.map { [Double($0.center.x), Double($0.center.y)] }
            diagnostic["filteredCenter"] = result.point.map { [Double($0.x), Double($0.y)] }
            diagnostic["pinchBeginDistance"] = detector.pinchBeginDistance
            diagnostic["pinchReleaseDistance"] = detector.pinchReleaseDistance
            event?(result, generation)
        }
        defer {
            diagnostic["processingMs"] = (ProcessInfo.processInfo.systemUptime - started) * 1000
            diagnosticFrames.append(diagnostic)
            if diagnosticFrames.count > 600 { diagnosticFrames.removeFirst() }
            diagnosticCounts[diagnostic["reason"] as? String ?? "error", default: 0] += 1
            diagnosticPhases[diagnostic["phase"] as? String ?? "unknown", default: 0] += 1
            writeDiagnostics(force: false)
        }
        do {
            // The capture connection has already rotated and mirrored the pixel buffer.
            try VNImageRequestHandler(cvPixelBuffer: pixel, orientation: .up).perform([request])
            visionFailures = 0
            let hands = request.results ?? []
            diagnostic["hands"] = hands.count
            guard hands.count == 1, let hand = hands.first else {
                publish(nil, reason: hands.isEmpty ? "no_hand" : "multiple_hands"); return
            }
            let thumb = try hand.recognizedPoint(.thumbTip)
            let index = try hand.recognizedPoint(.indexTip)
            // Wrist/palm confidence is diagnostic only. It cannot veto clear fingertips.
            let wrist = try? hand.recognizedPoint(.wrist)
            let middle = try? hand.recognizedPoint(.middleMCP)
            diagnostic["confidence"] = [Double(thumb.confidence), Double(index.confidence),
                Double(wrist?.confidence ?? 0), Double(middle?.confidence ?? 0)]
            guard thumb.confidence >= Float(AirGrabGesture.minimumTipConfidence),
                  index.confidence >= Float(AirGrabGesture.minimumTipConfidence) else {
                publish(nil, reason: "low_confidence"); return
            }
            let sample = AirGrabGesture.sample(thumb: thumb.location, index: index.location,
                thumbConfidence: Double(thumb.confidence), indexConfidence: Double(index.confidence),
                width: Double(CVPixelBufferGetWidth(pixel)), height: Double(CVPixelBufferGetHeight(pixel)))
            diagnostic["pinchDistance"] = sample?.pinchDistance
            publish(sample, reason: sample == nil ? "invalid_coordinates" : "accepted")
        } catch {
            publish(nil, reason: "vision_error")
            visionFailures += 1
            if visionFailures >= 15 { failure?(L10n.t("手势识别暂不可用，请重新开启。", "Hand tracking is unavailable. Turn Air Grab on again.")) }
        }
    }

    private func writeDiagnostics(force: Bool) {
        guard !diagnosticFrames.isEmpty else { return }
        let now = ProcessInfo.processInfo.systemUptime
        guard force || now - diagnosticLastWrite >= 2 else { return }
        diagnosticLastWrite = now
        let report: [String: Any] = ["version": "air-diag-tips-v3", "updated": Date().timeIntervalSince1970,
            "minimumTipConfidence": AirGrabGesture.minimumTipConfidence, "wristRequired": false, "targetFPS": 30,
            "heldTrackingGraceMs": AirGrabGesture.heldTrackingGrace * 1000,
            "camera": diagnosticCamera, "counts": diagnosticCounts, "phases": diagnosticPhases,
            "bridgeCounts": diagnosticBridgeCounts, "bridgeEvents": diagnosticBridgeEvents,
            "frames": diagnosticFrames]
        diagnosticQueue.async {
            guard let data = try? JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]),
                  let folder = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else { return }
            try? data.write(to: folder.appendingPathComponent("air-grab-diagnostics.json"), options: .atomic)
        }
    }
    private enum CameraFailure: Error { case unavailable }
}

struct AirGrabOrientationReader: UIViewRepresentable {
    var onChange: (UIInterfaceOrientation) -> Void
    func makeUIView(context: Context) -> ReaderView { ReaderView() }
    func updateUIView(_ view: ReaderView, context: Context) { view.onChange = onChange; view.report() }
    final class ReaderView: UIView {
        var onChange: ((UIInterfaceOrientation) -> Void)?
        private var reported: UIInterfaceOrientation?
        override func didMoveToWindow() { super.didMoveToWindow(); report() }
        override func layoutSubviews() { super.layoutSubviews(); report() }
        func report() {
            guard let value = window?.windowScene?.effectiveGeometry.interfaceOrientation,
                  value != .unknown, value != reported else { return }
            reported = value
            DispatchQueue.main.async { [weak self] in self?.onChange?(value) }
        }
    }
}

private struct ScanRecordsVisibleKey: EnvironmentKey { static let defaultValue = true }
extension EnvironmentValues {
    var scanRecordsVisible: Bool {
        get { self[ScanRecordsVisibleKey.self] }
        set { self[ScanRecordsVisibleKey.self] = newValue }
    }
}
