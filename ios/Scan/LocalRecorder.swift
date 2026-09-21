import Foundation
import CryptoKit

struct RecordingFailure: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

struct RecordedFrame: Codable {
    let bytes: Int
    let sha256: String
}

struct RecordingManifest: Codable {
    let id: String
    let frames: [RecordedFrame]
    var sha256: String {
        SHA256.hash(data: Data(frames.map { $0.sha256 + "\n" }.joined().utf8))
            .map { String(format: "%02x", $0) }.joined()
    }
}

struct UploadReceipt: Codable {
    let id: String
    let confirmed: Bool
    let frames: Int
    let bytes: Int
    let sha256: String
    let session: String
}

/// Atomic packets survive app restarts. A frozen manifest defines one retryable upload.
final class LocalPacketStore: @unchecked Sendable {
    let directory: URL
    private let lock = NSLock()
    private var count = 0
    private var bytes = 0
    private var sealed = false
    var packetCount: Int { lock.lock(); defer { lock.unlock() }; return count }
    var totalBytes: Int { lock.lock(); defer { lock.unlock() }; return bytes }

    static var root: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("recordings", isDirectory: true)
    }

    init(root: URL = LocalPacketStore.root) throws {
        directory = root.appendingPathComponent(UUID().uuidString.lowercased(), isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    init(recovering directory: URL) throws {
        self.directory = directory
        let files = try Self.packetFiles(directory)
        count = files.count
        bytes = try files.reduce(0) { total, file in
            total + (try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0)
        }
        sealed = true
    }

    static func pendingDirectories(root: URL = LocalPacketStore.root) throws -> [URL] {
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.isDirectoryKey])
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    private static func packetFiles(_ directory: URL) throws -> [URL] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: [.fileSizeKey])
            .filter { $0.pathExtension == "bin" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        for (i, file) in files.enumerated() {
            guard file.lastPathComponent == String(format: "f%06d.bin", i) else {
                throw RecordingFailure(message: L10n.t("录制帧不连续，原始文件已保留", "Recording has missing frames; originals retained"))
            }
        }
        return files
    }

    func write(_ packet: Data) throws {
        lock.lock(); defer { lock.unlock() }
        guard !sealed else { throw RecordingFailure(message: "Recording is sealed") }
        let url = directory.appendingPathComponent(String(format: "f%06d.bin", count))
        try packet.write(to: url, options: .atomic)
        let handle = try FileHandle(forWritingTo: url)
        defer { try? handle.close() }
        try handle.synchronize()
        count += 1
        bytes += packet.count
    }

    func seal() { lock.lock(); sealed = true; lock.unlock() }

    func manifest() throws -> RecordingManifest {
        lock.lock(); defer { lock.unlock() }
        guard sealed else { throw RecordingFailure(message: "Recording is still being saved") }
        let files = try Self.packetFiles(directory)
        guard !files.isEmpty else { throw RecordingFailure(message: L10n.t("录制没有可上传帧", "Recording has no frames")) }
        var frames: [RecordedFrame] = []
        for file in files {
            let data = try Data(contentsOf: file, options: .mappedIfSafe)
            guard data.count >= 12, data.prefix(8) == Data("SGFIPD01".utf8) else {
                throw RecordingFailure(message: L10n.t("录制帧损坏，原始文件已保留", "Invalid recording packet; originals retained"))
            }
            frames.append(RecordedFrame(bytes: data.count, sha256: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()))
        }
        let url = directory.appendingPathComponent("upload_manifest.json")
        if FileManager.default.fileExists(atPath: url.path) {
            let existing = try JSONDecoder().decode(RecordingManifest.self, from: Data(contentsOf: url))
            guard existing.frames.count == frames.count,
                  zip(existing.frames, frames).allSatisfy({ $0.bytes == $1.bytes && $0.sha256 == $1.sha256 }) else {
                throw RecordingFailure(message: L10n.t("录制在封存后发生变化，已停止上传", "Sealed recording changed; upload stopped"))
            }
            return existing
        }
        let manifest = RecordingManifest(id: UUID().uuidString.lowercased(), frames: frames)
        try JSONEncoder().encode(manifest).write(to: url, options: .atomic)
        return manifest
    }

    static func deleteRecording(directory: URL) throws {
        try FileManager.default.removeItem(at: directory)
    }
}

/// Each HTTP frame is committed with SHA256 and can be retried independently.
/// TCP v1 remains the live streaming protocol; packet contents are unchanged.
enum RecordingUploader {
    private struct Prepared: Decodable { let id: String; let sha256: String; let missing: [Int] }

    static func upload(store: LocalPacketStore, host: String, httpPort: Int = 8765,
                       progress: @escaping @MainActor (Int, Int) -> Void) async throws -> UploadReceipt {
        let manifest = try await Task.detached { try store.manifest() }.value
        try Task.checkCancellation()
        var components = URLComponents()
        components.scheme = "http"; components.host = host; components.port = httpPort
        guard let base = components.url else { throw URLError(.badURL) }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 60
        config.timeoutIntervalForResource = 300
        let session = URLSession(configuration: config)
        defer { session.invalidateAndCancel() }
        let (htmlData, response) = try await session.data(from: base)
        try check(response, data: htmlData)
        let html = String(decoding: htmlData, as: UTF8.self)
        guard let range = html.range(of: "token='[0-9a-f]+'", options: .regularExpression) else {
            throw RecordingFailure(message: L10n.t("无法读取工作站上传凭据", "Cannot read workstation upload token"))
        }
        let token = String(html[range].dropFirst(7).dropLast())
        func request(_ path: String, body: Data? = nil) -> URLRequest {
            var r = URLRequest(url: base.appendingPathComponent(path))
            r.httpMethod = "POST"
            r.setValue(token, forHTTPHeaderField: "X-Scan-Token")
            r.httpBody = body
            r.setValue("application/json", forHTTPHeaderField: "Content-Type")
            return r
        }
        let (data, reply) = try await session.data(for: request("api/uploads", body: JSONEncoder().encode(manifest)))
        try check(reply, data: data)
        let prepared = try JSONDecoder().decode(Prepared.self, from: data)
        guard prepared.id == manifest.id, prepared.sha256 == manifest.sha256,
              Set(prepared.missing).count == prepared.missing.count,
              prepared.missing.allSatisfy({ manifest.frames.indices.contains($0) }) else {
            throw RecordingFailure(message: "Upload manifest acknowledgement mismatch")
        }
        var sent = manifest.frames.count - prepared.missing.count
        await progress(sent, manifest.frames.count)
        for i in prepared.missing {
            try Task.checkCancellation()
            var r = request("api/uploads/\(manifest.id)/frames/\(i)")
            r.setValue("application/octet-stream", forHTTPHeaderField: "Content-Type")
            let file = store.directory.appendingPathComponent(String(format: "f%06d.bin", i))
            let (data, reply) = try await session.upload(for: r, fromFile: file)
            try check(reply, data: data)
            sent += 1
            await progress(sent, manifest.frames.count)
        }
        try Task.checkCancellation()
        let (receiptData, receiptReply) = try await session.data(for: request("api/uploads/\(manifest.id)/complete"))
        try check(receiptReply, data: receiptData)
        let receipt = try JSONDecoder().decode(UploadReceipt.self, from: receiptData)
        guard receipt.confirmed, receipt.id == manifest.id, receipt.frames == manifest.frames.count,
              receipt.bytes == manifest.frames.reduce(0, { $0 + $1.bytes }), receipt.sha256 == manifest.sha256,
              receipt.session.hasPrefix("scan_") else {
            throw RecordingFailure(message: L10n.t("接收回执不匹配，本地录制已保留", "Receipt mismatch; local recording retained"))
        }
        try receiptData.write(to: store.directory.appendingPathComponent("upload_receipt.json"), options: .atomic)
        return receipt
    }

    private static func check(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
            let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            throw RecordingFailure(message: body?["error"] as? String ?? L10n.t("工作站上传失败，请重试；录制已保留", "Upload failed; retry safely — recording retained"))
        }
    }
}
