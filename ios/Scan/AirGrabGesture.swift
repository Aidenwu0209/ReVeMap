import Foundation

/// Pinch control depends only on the two fingertips, not wrist or palm estimates.
struct AirGrabGesture {
    enum Phase: String, Codable { case searching, ready, needsOpen, began, changed, ended, uncertain, lost, reset }
    struct Sample {
        let center: CGPoint
        /// Fingertip separation as a fraction of the image's shorter side.
        let pinchDistance: Double
    }
    struct Event {
        let phase: Phase
        let point: CGPoint?
    }
    static let minimumTipConfidence = 0.30
    static let heldTrackingGrace: TimeInterval = 0.80
    private(set) var pinching = false
    private var armed = false
    private var openFrames = 0
    private var closedFrames = 0
    private var filtered: CGPoint?
    private var lastRawCenter: CGPoint?
    private var relocationCandidate: (point: CGPoint, time: TimeInterval)?
    private var lastGoodTime: TimeInterval?
    private var openDistances: [Double] = []
    var pinchBeginDistance: Double {
        guard !openDistances.isEmpty else { return 0.06 }
        let sorted = openDistances.sorted()
        return min(0.12, max(0.06, sorted[sorted.count / 2] * 0.30))
    }
    var pinchReleaseDistance: Double { max(0.12, pinchBeginDistance * 1.7) }
    private var trackingGrace: TimeInterval { pinching ? Self.heldTrackingGrace : 0.35 }

    mutating func reset() { self = Self() }

    static func sample(thumb: CGPoint, index: CGPoint, thumbConfidence: Double, indexConfidence: Double,
                       width: Double, height: Double) -> Sample? {
        guard width.isFinite, height.isFinite, width > 0, height > 0,
              thumbConfidence.isFinite, indexConfidence.isFinite,
              thumbConfidence >= minimumTipConfidence, indexConfidence >= minimumTipConfidence,
              thumb.x.isFinite, thumb.y.isFinite, index.x.isFinite, index.y.isFinite,
              (0...1).contains(thumb.x), (0...1).contains(thumb.y),
              (0...1).contains(index.x), (0...1).contains(index.y) else { return nil }
        let shortSide = min(width, height)
        let distance = hypot(Double(thumb.x - index.x) * width / shortSide,
                             Double(thumb.y - index.y) * height / shortSide)
        return Sample(center: CGPoint(x: (thumb.x + index.x) / 2, y: 1 - (thumb.y + index.y) / 2),
                      pinchDistance: distance)
    }

    mutating func process(_ sample: Sample?, at time: TimeInterval = ProcessInfo.processInfo.systemUptime) -> Event {
        guard time.isFinite else { reset(); return Event(phase: .lost, point: nil) }
        if let lastGoodTime {
            if time < lastGoodTime { reset(); return Event(phase: .lost, point: nil) }
            if sample != nil, time - lastGoodTime > trackingGrace { reset() }
        }
        guard let sample, sample.center.x.isFinite, sample.center.y.isFinite,
              (0...1).contains(sample.center.x), (0...1).contains(sample.center.y),
              sample.pinchDistance.isFinite, sample.pinchDistance >= 0 else { return missing(at: time) }
        // Compare observations, not the smoothed cursor: smoothing lag grows during a fast pan.
        // A larger displacement needs two nearby observations before we follow it again.
        if let previous = lastRawCenter,
           hypot(previous.x - sample.center.x, previous.y - sample.center.y) > 0.24 {
            let confirmed = relocationCandidate.map {
                time - $0.time <= 0.14 && time >= $0.time &&
                hypot($0.point.x - sample.center.x, $0.point.y - sample.center.y) <= 0.18
            } ?? false
            guard confirmed else {
                relocationCandidate = (sample.center, time)
                return missing(at: time)
            }
        }
        relocationCandidate = nil
        lastRawCenter = sample.center
        if let lastGoodTime, time - lastGoodTime > 0.14 { openFrames = 0; closedFrames = 0 }
        let point = filtered.map { CGPoint(x: $0.x + (sample.center.x - $0.x) * 0.38,
                                           y: $0.y + (sample.center.y - $0.y) * 0.38) } ?? sample.center
        filtered = point
        lastGoodTime = time
        // Learn scale from clearly open fingertips only; freeze it throughout a hold.
        if !pinching, sample.pinchDistance > max(0.10, pinchBeginDistance * 1.7) {
            openDistances.append(sample.pinchDistance)
            if openDistances.count > 9 { openDistances.removeFirst() }
        }
        let openDistance = pinching ? pinchReleaseDistance : max(0.10, pinchBeginDistance * 1.7)
        if sample.pinchDistance > openDistance { openFrames += 1; closedFrames = 0 }
        else if sample.pinchDistance < pinchBeginDistance { closedFrames += 1; openFrames = 0 }
        else { openFrames = 0; closedFrames = 0 }
        if pinching {
            if openFrames >= 3 {
                pinching = false
                armed = true
                return Event(phase: .ended, point: point)
            }
            return Event(phase: .changed, point: point)
        }
        if !armed {
            if openFrames >= 3 { armed = true }
            return Event(phase: armed ? .ready : .needsOpen, point: point)
        }
        if closedFrames >= 3 {
            pinching = true
            return Event(phase: .began, point: point)
        }
        return Event(phase: .ready, point: point)
    }

    private mutating func missing(at time: TimeInterval) -> Event {
        // Brief weak frames neither count as evidence nor erase already observed evidence.
        if let lastGoodTime, time - lastGoodTime > 0.14 { openFrames = 0; closedFrames = 0 }
        // Keep a confirmed grip through brief motion blur; never extrapolate missing joints.
        if let point = filtered, let lastGoodTime, time - lastGoodTime <= trackingGrace {
            return Event(phase: .uncertain, point: point)
        }
        let wasTracking = filtered != nil
        reset()
        return Event(phase: wasTracking ? .lost : .searching, point: nil)
    }
}

struct AirGrabFrameCadence {
    private var last = -Double.infinity
    mutating func reset() { last = -.infinity }
    mutating func accepts(_ timestamp: Double) -> Bool {
        guard timestamp.isFinite else { return false }
        if timestamp < last { reset() }
        guard timestamp - last >= 0.9 / 30.0 else { return false }
        last = timestamp
        return true
    }
}

struct AirGrabPacket: Encodable {
    let session: String
    let sequence: Int
    let phase: AirGrabGesture.Phase
    let mode: String
    let x: Double?
    let y: Double?
}
