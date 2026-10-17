import Foundation
import Combine

enum ScanEndpoint {
    static func port(_ text: String) -> UInt16? {
        guard let value = UInt16(text.trimmingCharacters(in: .whitespacesAndNewlines)), value > 0 else { return nil }
        return value
    }

    static func restoredPort(_ value: Int) -> Int {
        (1...65535).contains(value) ? value : 7001
    }

    static func url(host: String, path: String) throws -> URL {
        var parts = URLComponents()
        parts.scheme = "http"
        parts.host = host.trimmingCharacters(in: .whitespacesAndNewlines)
        parts.port = 8765
        parts.path = path
        guard parts.host?.isEmpty == false, let url = parts.url else { throw URLError(.badURL) }
        return url
    }
}

@MainActor
final class WorkstationSettings: ObservableObject {
    @Published private(set) var models: [String] = []
    @Published private(set) var modelsError: String?
    @Published private(set) var isRefreshing = false
    @Published private(set) var isStarting = false
    @Published var startError: String?
    private let network: URLSession
    private var modelRevision = 0
    private var modelHost = ""

    init(network: URLSession = .shared) { self.network = network }

    func refreshModels(host: String) async {
        modelRevision += 1
        let revision = modelRevision
        if host != modelHost { models = []; modelHost = host }
        modelsError = nil
        isRefreshing = true
        defer { if revision == modelRevision { isRefreshing = false } }
        do {
            struct Response: Decodable { let models: [String] }
            let data = try await get(host: host, path: "/api/models")
            let response = try JSONDecoder().decode(Response.self, from: data)
            try Task.checkCancellation()
            guard revision == modelRevision else { return }
            models = response.models
            if models.isEmpty { modelsError = L10n.t("工作站未提供模型，请检查后重试", "No models returned. Check the workstation and retry.") }
        } catch {
            guard !Task.isCancelled, revision == modelRevision else { return }
            modelsError = L10n.t("模型列表加载失败，请点检查连接重试", "Models could not be loaded. Check the connection to retry.")
        }
    }

    private func get(host: String, path: String) async throws -> Data {
        let request = URLRequest(url: try ScanEndpoint.url(host: host, path: path),
                                 cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 10)
        let (data, response) = try await network.data(for: request)
        try check(response, data: data)
        return data
    }

    private func check(_ response: URLResponse, data: Data) throws {
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            throw RecordingFailure(message: body?["error"] as? String ?? L10n.t("工作站请求失败，请检查连接后重试", "Workstation request failed. Check the connection and retry."))
        }
    }

    func synchronizeModel(host: String, model: String) async throws {
        let html = String(decoding: try await get(host: host, path: "/"), as: UTF8.self)
        guard let range = html.range(of: "token='[0-9a-f]{48}'", options: .regularExpression) else {
            throw RecordingFailure(message: L10n.t("无法获取工作站凭据", "Cannot obtain workstation credentials."))
        }
        try Task.checkCancellation()
        var request = URLRequest(url: try ScanEndpoint.url(host: host, path: "/api/options"), timeoutInterval: 10)
        request.httpMethod = "POST"
        request.setValue(String(html[range].dropFirst(7).dropLast()), forHTTPHeaderField: "X-Scan-Token")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["vlm": model])
        let (data, response) = try await network.data(for: request)
        try check(response, data: data)
        let body = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard (body?["options"] as? [String: Any])?["vlm"] as? String == model else {
            throw RecordingFailure(message: L10n.t("工作站未确认所选模型，请重试", "The workstation did not confirm the selected model. Retry."))
        }
    }

    /// The local branch performs no HTTP requests; live capture requires a matching receipt.
    func beginCapture(local: Bool, host: String, port: UInt16?, model: String,
                      start: (String, UInt16, Bool) -> Void) async {
        guard !isStarting else { return }
        isStarting = true
        startError = nil
        defer { isStarting = false }
        do {
            try Task.checkCancellation()
            if local { start("", 7001, true); return }
            guard !host.isEmpty, let port, port > 0 else {
                throw RecordingFailure(message: L10n.t("请填写有效地址和 1–65535 的端口", "Enter a valid address and a port from 1 to 65535."))
            }
            try await synchronizeModel(host: host, model: model)
            try Task.checkCancellation()
            start(host, port, false)
        } catch {
            guard !Task.isCancelled else { return }
            startError = L10n.t("尚未开始扫描：", "Scan has not started: ") + error.localizedDescription
        }
    }

    func beginUpload(host: String, model: String, upload: (String) async -> Void) async {
        guard !isStarting else { return }
        isStarting = true
        startError = nil
        defer { isStarting = false }
        do {
            try Task.checkCancellation()
            try await synchronizeModel(host: host, model: model)
            try Task.checkCancellation()
            await upload(host)
        } catch {
            if !Task.isCancelled { startError = error.localizedDescription }
        }
    }
}

struct StreamStallWatchdog {
    private var lastCount: UInt64 = 0
    private var lastChange: TimeInterval?

    mutating func connectionLost(frames: UInt64, status: String, now: TimeInterval) -> Bool {
        if lastChange == nil || frames != lastCount {
            lastCount = frames
            lastChange = now
            return false
        }
        return frames > 0 && (status == "Waiting for workstation" || status == "Reconnecting")
            && now - (lastChange ?? now) > 12
    }
}
