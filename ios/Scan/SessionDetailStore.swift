import Foundation
import Combine

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
    private let network: URLSession
    private var revision = 0
    private var initializedView = false

    init(network: URLSession = .shared) { self.network = network }

    var isTerminal: Bool {
        guard let status else { return false }
        return ["completed", "failed", "cancelled", "interrupted", "incomplete", "captured"].contains(status.status)
    }

    func refreshUntilSettled(pause: () async throws -> Void = {
        try await Task.sleep(nanoseconds: 3_000_000_000)
    }) async {
        while !Task.isCancelled {
            await load()
            guard !Task.isCancelled, !isTerminal || errorText != nil else { return }
            do { try await pause() } catch { return }
        }
    }

    private func fetch<T: Decodable>(_ type: T.Type, url: URL) async throws -> T {
        let request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 10)
        let (data, response) = try await network.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        try Task.checkCancellation()
        return try JSONDecoder().decode(type, from: data)
    }

    func load() async {
        revision += 1
        let requestRevision = revision
        let requestHost = host, requestRecord = recordID
        let base = "/s/\(requestRecord)/api/"
        do {
            let decoded = try await fetch(DetailStatus.self, url: ScanEndpoint.url(host: requestHost, path: base + "status"))
            async let fetchedInstances = try? fetch([InstanceRow].self, url: ScanEndpoint.url(host: requestHost, path: base + "instances"))
            async let fetchedObjects = try? fetch(ObjectCatalog.self, url: ScanEndpoint.url(host: requestHost, path: base + "objects"))
            let (list, catalog) = await (fetchedInstances, fetchedObjects)
            guard !Task.isCancelled, revision == requestRevision, host == requestHost, recordID == requestRecord else { return }
            status = decoded
            errorText = list == nil ? L10n.t("实例加载失败，请刷新重试", "Instances could not be loaded. Refresh to retry.") : nil
            if !initializedView { viewMode = decoded.view ?? "semantic_id"; initializedView = true }
            if let list { instances = list }
            if let catalog { objects = catalog }
            objectError = catalog == nil ? L10n.t("暂时无法加载可选取物体", "Selectable objects are unavailable") : nil
        } catch {
            guard !Task.isCancelled, revision == requestRevision, host == requestHost, recordID == requestRecord else { return }
            errorText = L10n.t("无法加载会话数据", "Failed to load session")
        }
    }

    func loadObjects() async {
        await load()
    }

    func setView(_ mode: String) async {
        initializedView = true
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
        do {
            let (_, response) = try await network.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            try Task.checkCancellation()
            viewMode = mode
        } catch {
            if !Task.isCancelled { errorText = L10n.t("切换视图失败，请重试", "Could not change view. Retry.") }
        }
    }

    private func fetchToken() async -> String? {
        if let token { return token }
        guard let url = URL(string: "http://\(host):8765/") else { return nil }
        guard let (data, _) = try? await network.data(for: URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 10)),
              let html = String(data: data, encoding: .utf8),
              let range = html.range(of: "token='[0-9a-f]{48}'", options: .regularExpression)
        else { return nil }
        token = String(html[range].dropFirst(7).dropLast())
        return token
    }
}
