import XCTest
import Foundation
@testable import ScanCore

private final class StubProtocol: URLProtocol {
    static var handler: (URLRequest) throws -> (Int, Data) = { _ in throw URLError(.notConnectedToInternet) }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status, data) = try Self.handler(request)
            let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self, didFailWithError: error) }
    }
    override func stopLoading() {}
}

@MainActor
final class ScanReliabilityTests: XCTestCase {
    private func network() -> URLSession {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubProtocol.self]
        return URLSession(configuration: config)
    }
    private func data(_ value: Any) throws -> Data { try JSONSerialization.data(withJSONObject: value) }

    func testInvalidPortsCannotTrapOrBecomeSavedConfiguration() {
        for invalid in ["", "0", "-1", "65536", "70010", "999999999999999999999", "abc"] {
            XCTAssertNil(ScanEndpoint.port(invalid))
        }
        XCTAssertEqual(ScanEndpoint.port(" 7001 "), 7001)
        XCTAssertEqual(ScanEndpoint.port("65535"), 65535)
        XCTAssertEqual(ScanEndpoint.restoredPort(-1), 7001)
        XCTAssertEqual(ScanEndpoint.restoredPort(70010), 7001)
        XCTAssertEqual(ScanEndpoint.restoredPort(12345), 12345)
    }

    func testLocalCaptureStartsOfflineWithoutAnyHTTPRequestOrValidEndpoint() async {
        StubProtocol.handler = { _ in XCTFail("Local capture must not call the workstation"); throw URLError(.notConnectedToInternet) }
        let session = network(); defer { session.invalidateAndCancel() }
        let settings = WorkstationSettings(network: session)
        var started = false
        await settings.beginCapture(local: true, host: "", port: nil, model: "offline-model") { _, _, local in
            started = local
        }
        XCTAssertTrue(started)
        XCTAssertNil(settings.startError)
    }

    func testModelListCanRecoverAfterFailedRequest() async throws {
        let session = network(); defer { session.invalidateAndCancel() }
        let settings = WorkstationSettings(network: session)
        StubProtocol.handler = { _ in throw URLError(.notConnectedToInternet) }
        await settings.refreshModels(host: "workstation")
        XCTAssertNotNil(settings.modelsError)
        let response = try data(["models": ["model-a", "model-b"]])
        StubProtocol.handler = { request in
            XCTAssertEqual(request.url?.path, "/api/models")
            return (200, response)
        }
        await settings.refreshModels(host: "workstation")
        XCTAssertEqual(settings.models, ["model-a", "model-b"])
        XCTAssertNil(settings.modelsError)
        XCTAssertFalse(settings.isRefreshing)
    }

    func testNewHostDoesNotReuseOldHostModelsOnFailure() async throws {
        let response = try data(["models": ["host-a-model"]])
        StubProtocol.handler = { _ in (200, response) }
        let session = network(); defer { session.invalidateAndCancel() }
        let settings = WorkstationSettings(network: session)
        await settings.refreshModels(host: "host-a")
        StubProtocol.handler = { _ in throw URLError(.cannotConnectToHost) }
        await settings.refreshModels(host: "host-b")
        XCTAssertTrue(settings.models.isEmpty)
        XCTAssertNotNil(settings.modelsError)
    }

    func testLiveCaptureRequiresSuccessfulMatchingModelReceipt() async throws {
        let token = String(repeating: "a", count: 48)
        let html = Data("token='\(token)'".utf8)
        for (status, acknowledged, expected) in [(500, "chosen", false), (200, "different", false), (200, "chosen", true)] {
            let reply = try data(["options": ["vlm": acknowledged]])
            StubProtocol.handler = { request in
                if request.url?.path == "/" { return (200, html) }
                XCTAssertEqual(request.url?.path, "/api/options")
                XCTAssertEqual(request.httpMethod, "POST")
                XCTAssertEqual(request.value(forHTTPHeaderField: "X-Scan-Token"), token)
                return (status, reply)
            }
            let session = network(); defer { session.invalidateAndCancel() }
            let settings = WorkstationSettings(network: session)
            var started = false
            await settings.beginCapture(local: false, host: "workstation", port: 7001, model: "chosen") { _, _, local in
                XCTAssertFalse(local); started = true
            }
            XCTAssertEqual(started, expected)
            XCTAssertEqual(settings.startError == nil, expected)
        }
    }

    func testCancelledStartNeverStartsCapture() async {
        let session = network(); defer { session.invalidateAndCancel() }
        let settings = WorkstationSettings(network: session)
        let task = Task {
            await settings.beginCapture(local: true, host: "", port: nil, model: "") { _, _, _ in
                XCTFail("A dismissed preview must not start a capture")
            }
        }
        task.cancel()
        await task.value
        XCTAssertFalse(settings.isStarting)
    }

    func testUploadRequiresModelReceiptAndRejectsDuplicateOperation() async throws {
        let token = String(repeating: "a", count: 48)
        let reply = try data(["options": ["vlm": "chosen"]])
        let session = network(); defer { session.invalidateAndCancel() }
        let settings = WorkstationSettings(network: session)
        StubProtocol.handler = { _ in (503, Data()) }
        await settings.beginUpload(host: "workstation", model: "chosen") { _ in
            XCTFail("Failed configuration must retain the recording without uploading")
        }
        XCTAssertNotNil(settings.startError)
        StubProtocol.handler = { request in
            request.url?.path == "/" ? (200, Data("token='\(token)'".utf8)) : (200, reply)
        }
        var uploads = 0
        await settings.beginUpload(host: "workstation", model: "chosen") { host in
            XCTAssertEqual(host, "workstation")
            uploads += 1
            await settings.beginUpload(host: "another-host", model: "different") { _ in
                XCTFail("An active upload must exclude duplicate configuration operations")
            }
        }
        XCTAssertEqual(uploads, 1)
        XCTAssertNil(settings.startError)
        XCTAssertFalse(settings.isStarting)
    }

    func testDisconnectRequiresPriorFramesAndLongStall() {
        var health = StreamStallWatchdog()
        XCTAssertFalse(health.connectionLost(frames: 0, status: "Waiting for workstation", now: 0))
        XCTAssertFalse(health.connectionLost(frames: 0, status: "Waiting for workstation", now: 100))
        XCTAssertFalse(health.connectionLost(frames: 10, status: "Connected", now: 101))
        XCTAssertFalse(health.connectionLost(frames: 10, status: "Reconnecting", now: 113))
        XCTAssertTrue(health.connectionLost(frames: 10, status: "Reconnecting", now: 114))
        XCTAssertFalse(health.connectionLost(frames: 11, status: "Connected", now: 115))
        XCTAssertFalse(health.connectionLost(frames: 11, status: "Reconnecting", now: 116))
    }

    func testDamagedRecordingDoesNotHideGoodRecordingAndQuarantinePreservesBytes() throws {
        let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let root = parent.appendingPathComponent("recordings")
        defer { try? FileManager.default.removeItem(at: parent) }
        let bad = root.appendingPathComponent("aaa-broken")
        try FileManager.default.createDirectory(at: bad, withIntermediateDirectories: true)
        let bytes = Data("SGFIPD01original-frame".utf8)
        try bytes.write(to: bad.appendingPathComponent("f000001.bin"))
        let good = try LocalPacketStore(root: root)
        try good.write(bytes); good.seal()
        let result = try LocalPacketStore.recoverPending(root: root)
        XCTAssertEqual(result.stores.count, 1)
        XCTAssertEqual(result.stores.first?.directory.resolvingSymlinksInPath(), good.directory.resolvingSymlinksInPath())
        XCTAssertEqual(result.issues.count, 1)
        try LocalPacketStore.quarantine(result.issues[0], root: root)
        let retained = try FileManager.default.contentsOfDirectory(at: parent.appendingPathComponent("recordings-recovery"), includingPropertiesForKeys: nil)
        XCTAssertEqual(retained.count, 1)
        XCTAssertEqual(try Data(contentsOf: retained[0].appendingPathComponent("f000001.bin")), bytes)
        XCTAssertEqual(try Data(contentsOf: good.directory.appendingPathComponent("f000000.bin")), bytes)
        XCTAssertTrue(try LocalPacketStore.recoverPending(root: root).issues.isEmpty)
    }

    func testOnlyDamagedRecordingsLeaveNoPendingStoreToBlockNewCapture() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let bad = root.appendingPathComponent("bad")
        try FileManager.default.createDirectory(at: bad, withIntermediateDirectories: true)
        try Data([1, 2, 3]).write(to: bad.appendingPathComponent("f000002.bin"))
        let result = try LocalPacketStore.recoverPending(root: root)
        XCTAssertEqual(result.issues.count, 1)
        XCTAssertTrue(result.stores.isEmpty)
        XCTAssertTrue(FileManager.default.fileExists(atPath: bad.path))
    }

    func testDetailPollingRefreshesCountsAndTimingWhilePreservingSelectedView() async throws {
        let session = network(); defer { session.invalidateAndCancel() }
        let store = SessionDetailStore(network: session)
        store.host = "workstation"; store.recordID = "scan_test"
        let first = try data(["status": "mapping", "capture": ["frames": 5], "view": "semantic_id"])
        let last = try data(["status": "completed", "capture": ["frames": 50], "view": "semantic_id",
                             "options": ["vlm": "chosen"],
                             "timeline": [["stage": "mapping", "seconds": 12.0]]])
        let catalog = try data(["revision": "1", "points": 100, "rgb_available": true, "classes": [], "instances": []] as [String: Any])
        let finalInstances = try data([["instance_id": 7, "semantic_name": "chair", "point_count": 90]])
        let finalCatalog = try data(["revision": "2", "points": 900, "rgb_available": true,
                                     "classes": [["kind": "class", "label_id": 5, "semantic_id": 5,
                                                   "semantic_name": "chair", "point_count": 90, "size_m": [1, 1, 1]]],
                                     "instances": []] as [String: Any])
        func install(_ status: Data, instances: Data = Data("[]".utf8), objects: Data) {
            StubProtocol.handler = { request in
                switch request.url!.lastPathComponent {
                case "status": return (200, status)
                case "instances": return (200, instances)
                case "objects": return (200, objects)
                default: throw URLError(.badURL)
                }
            }
        }
        install(first, objects: catalog)
        var pauses = 0
        await store.refreshUntilSettled {
            pauses += 1
            XCTAssertEqual(store.status?.capture?.frames, 5)
            await store.setView("rgb")
            install(last, instances: finalInstances, objects: finalCatalog)
        }
        XCTAssertEqual(pauses, 1)
        XCTAssertEqual(store.status?.capture?.frames, 50)
        XCTAssertEqual(store.status?.timeline?.first?.seconds, 12)
        XCTAssertEqual(store.status?.options?["vlm"]?.stringValue, "chosen")
        XCTAssertEqual(store.instances.first?.instance_id, 7)
        XCTAssertEqual(store.objects?.classes.first?.semantic_name, "chair")
        XCTAssertEqual(store.objects?.revision, "2")
        XCTAssertEqual(store.viewMode, "rgb")
        XCTAssertTrue(store.isTerminal)
    }

    func testDetailFailureRetriesAndCancellationStopsPolling() async {
        StubProtocol.handler = { _ in throw URLError(.notConnectedToInternet) }
        let session = network(); defer { session.invalidateAndCancel() }
        let store = SessionDetailStore(network: session)
        store.host = "workstation"; store.recordID = "scan_test"
        var pauses = 0
        await store.refreshUntilSettled {
            pauses += 1
            XCTAssertNotNil(store.errorText)
            throw CancellationError()
        }
        XCTAssertEqual(pauses, 1)
    }
}
