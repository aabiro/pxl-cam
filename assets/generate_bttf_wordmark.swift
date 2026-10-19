import Foundation
import CoreText
import CoreGraphics
import AppKit

// 1. Load BTTF.ttf
let fontURL = URL(fileURLWithPath: "/Users/aaryn/Projects/pxl-cam/assets/BTTF.ttf")
var error: Unmanaged<CFError>?
CTFontManagerRegisterFontsForURL(fontURL as CFURL, .process, &error)

guard let descriptors = CTFontManagerCreateFontDescriptorsFromURL(fontURL as CFURL) as? [CTFontDescriptor],
      let firstDesc = descriptors.first else {
    fatalError("Failed to load BTTF.ttf")
}

let font = CTFontCreateWithFontDescriptor(firstDesc, 240.0, nil)

func reversePath(_ path: CGPath) -> CGPath {
    enum Seg {
        case line(CGPoint, CGPoint)
        case quad(CGPoint, CGPoint, CGPoint)
        case curve(CGPoint, CGPoint, CGPoint, CGPoint)
    }
    var subpaths: [[Seg]] = []
    var curSegs: [Seg] = []
    var startPt = CGPoint.zero
    var curPt = CGPoint.zero
    path.applyWithBlock { ptr in
        let el = ptr.pointee
        switch el.type {
        case .moveToPoint:
            if !curSegs.isEmpty { subpaths.append(curSegs); curSegs = [] }
            startPt = el.points[0]
            curPt = startPt
        case .addLineToPoint:
            let next = el.points[0]
            curSegs.append(.line(curPt, next))
            curPt = next
        case .addQuadCurveToPoint:
            let c = el.points[0], next = el.points[1]
            curSegs.append(.quad(curPt, c, next))
            curPt = next
        case .addCurveToPoint:
            let c1 = el.points[0], c2 = el.points[1], next = el.points[2]
            curSegs.append(.curve(curPt, c1, c2, next))
            curPt = next
        case .closeSubpath:
            if curPt != startPt {
                curSegs.append(.line(curPt, startPt))
            }
            if !curSegs.isEmpty { subpaths.append(curSegs); curSegs = [] }
            curPt = startPt
        @unknown default:
            break
        }
    }
    if !curSegs.isEmpty { subpaths.append(curSegs) }
    let out = CGMutablePath()
    for segs in subpaths {
        guard let last = segs.last else { continue }
        switch last {
        case .line(_, let b), .quad(_, _, let b), .curve(_, _, _, let b):
            out.move(to: b)
        }
        for seg in segs.reversed() {
            switch seg {
            case .line(let a, _):
                out.addLine(to: a)
            case .quad(let a, let c, _):
                out.addQuadCurve(to: a, control: c)
            case .curve(let a, let c1, let c2, _):
                out.addCurve(to: a, control1: c2, control2: c1)
            }
        }
        out.closeSubpath()
    }
    return out
}

func rawGlyphPath(_ ch: Character) -> CGPath {
    var unichars = Array(String(ch).utf16)
    var glyphs = [CGGlyph](repeating: 0, count: unichars.count)
    CTFontGetGlyphsForCharacters(font, &unichars, &glyphs, unichars.count)
    guard let p = CTFontCreatePathForGlyph(font, glyphs[0], nil) else {
        fatalError("No glyph for \(ch)")
    }
    var flip = CGAffineTransform(scaleX: 1.0, y: -1.0)
    // Reverse path after Y-flip so winding matches CoreGraphics strokingWithWidth (negative signed area)
    return reversePath(p.copy(using: &flip)!)
}

func warpPath(_ path: CGPath, mapPoint: @escaping (CGPoint) -> CGPoint) -> CGPath {
    let out = CGMutablePath()
    path.applyWithBlock { ptr in
        let el = ptr.pointee
        switch el.type {
        case .moveToPoint:
            out.move(to: mapPoint(el.points[0]))
        case .addLineToPoint:
            out.addLine(to: mapPoint(el.points[0]))
        case .addQuadCurveToPoint:
            out.addQuadCurve(to: mapPoint(el.points[1]), control: mapPoint(el.points[0]))
        case .addCurveToPoint:
            out.addCurve(to: mapPoint(el.points[2]), control1: mapPoint(el.points[0]), control2: mapPoint(el.points[1]))
        case .closeSubpath:
            out.closeSubpath()
        @unknown default:
            break
        }
    }
    return out
}

func pathToSVG(_ path: CGPath) -> String {
    var s = ""
    path.applyWithBlock { ptr in
        let el = ptr.pointee
        switch el.type {
        case .moveToPoint:
            let p = el.points[0]
            s += String(format: "M%.2f,%.2f ", p.x, p.y)
        case .addLineToPoint:
            let p = el.points[0]
            s += String(format: "L%.2f,%.2f ", p.x, p.y)
        case .addQuadCurveToPoint:
            let c = el.points[0], p = el.points[1]
            s += String(format: "Q%.2f,%.2f %.2f,%.2f ", c.x, c.y, p.x, p.y)
        case .addCurveToPoint:
            let c1 = el.points[0], c2 = el.points[1], p = el.points[2]
            s += String(format: "C%.2f,%.2f %.2f,%.2f %.2f,%.2f ", c1.x, c1.y, c2.x, c2.y, p.x, p.y)
        case .closeSubpath:
            s += "Z "
        @unknown default:
            break
        }
    }
    return s.trimmingCharacters(in: .whitespaces)
}

func mappedGlyph(_ ch: Character, x: CGFloat, y: CGFloat, w: CGFloat, h: CGFloat, topShiftX: CGFloat = 0, bottomShiftX: CGFloat = 0) -> CGPath {
    let raw = rawGlyphPath(ch)
    let bb = raw.boundingBoxOfPath
    let sx = w / bb.width
    let sy = h / bb.height
    return warpPath(raw) { pt in
        let nyNorm = (pt.y - bb.minY) / bb.height
        let shiftX = topShiftX * (1.0 - nyNorm) + bottomShiftX * nyNorm
        let nx = (pt.x - bb.minX) * sx + x + shiftX
        let ny = (pt.y - bb.minY) * sy + y
        return CGPoint(x: nx, y: ny)
    }
}

// Create polygon with negative signed area (counter-clockwise in Y-down) to match CGPath.copy(strokingWithWidth:)
func makePolygon(_ pts: [CGPoint]) -> CGPath {
    let p = CGMutablePath()
    guard !pts.isEmpty else { return p }
    var area: CGFloat = 0
    for i in 0..<pts.count {
        let a = pts[i]
        let b = pts[(i + 1) % pts.count]
        area += (a.x * b.y - b.x * a.y)
    }
    let ordered = area > 0 ? Array(pts.reversed()) : pts
    p.move(to: ordered[0])
    for pt in ordered.dropFirst() {
        p.addLine(to: pt)
    }
    p.closeSubpath()
    return p
}

// Load the exact traced Cyber Kitsune Fox path (normalized 1024x1024 space, y: 60..950, x: 284..740)
// and map it directly into the BTTF coordinate system (y: 34..376 -> scale = 342/890 = 0.38426966)
func loadFoxPath(foxRightX: CGFloat) -> (CGPath, (CGFloat, CGFloat) -> CGPoint) {
    let text = try! String(contentsOfFile: "/Users/aaryn/Projects/pxl-cam/assets/fox_traced_1024.path", encoding: .utf8)
    let scale: CGFloat = 342.0 / 890.0
    let mapPt: (CGFloat, CGFloat) -> CGPoint = { x, y in
        let mx = foxRightX - (740.0 - x) * scale
        let my = 34.0 + (y - 60.0) * scale
        return CGPoint(x: mx, y: my)
    }
    let p = CGMutablePath()
    for line in text.split(separator: "\n") {
        let parts = line.split(separator: " ")
        guard let cmd = parts.first else { continue }
        switch cmd {
        case "M":
            p.move(to: mapPt(CGFloat(Double(parts[1])!), CGFloat(Double(parts[2])!)))
        case "L":
            p.addLine(to: mapPt(CGFloat(Double(parts[1])!), CGFloat(Double(parts[2])!)))
        case "C":
            p.addCurve(
                to: mapPt(CGFloat(Double(parts[5])!), CGFloat(Double(parts[6])!)),
                control1: mapPt(CGFloat(Double(parts[1])!), CGFloat(Double(parts[2])!)),
                control2: mapPt(CGFloat(Double(parts[3])!), CGFloat(Double(parts[4])!))
            )
        case "Z":
            p.closeSubpath()
        default:
            break
        }
    }
    return (p, mapPt)
}

func renderPNG(fromSVG svgPath: String, width w: Int, height h: Int, darkBackground: Bool, to outPath: String) throws {
    guard let img = NSImage(contentsOfFile: svgPath) else { return }
    let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: w,
        pixelsHigh: h,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    )!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
    if darkBackground {
        NSColor(calibratedRed: 0.03, green: 0.03, blue: 0.05, alpha: 1.0).setFill()
        NSRect(x: 0, y: 0, width: w, height: h).fill()
    }
    img.draw(in: NSRect(x: 0, y: 0, width: w, height: h))
    NSGraphicsContext.restoreGraphicsState()
    if let pngData = rep.representation(using: .png, properties: [:]) {
        try pngData.write(to: URL(fileURLWithPath: outPath))
    }
}

// ============================================================================
// 1. TOP ROW (PXL): MELDED HORIZONTALLY INTO ONE CONTINUOUS CAMERA BODY TOP
// ============================================================================
let rawP = mappedGlyph("P", x: 64, y: 80, w: 148, h: 144, topShiftX: 4, bottomShiftX: -30)
let rawX = mappedGlyph("X", x: 160, y: 80, w: 162, h: 144, topShiftX: 0, bottomShiftX: -6)
let rawL = mappedGlyph("L", x: 284, y: 80, w: 156, h: 144, topShiftX: 0, bottomShiftX: 0)

let pentaprismOuter = makePolygon([
    CGPoint(x: 158, y: 82),
    CGPoint(x: 176, y: 34),
    CGPoint(x: 306, y: 34),
    CGPoint(x: 325, y: 82)
])
let viewfinderWindowHole = makePolygon([
    CGPoint(x: 186, y: 65),
    CGPoint(x: 195, y: 47),
    CGPoint(x: 287, y: 47),
    CGPoint(x: 296, y: 65)
])
let pentaprismMelded = pentaprismOuter.subtracting(viewfinderWindowHole)

let rawSpeedBars: CGPath = {
    let p = CGMutablePath()
    let topY: CGFloat = 80.0
    let botY: CGFloat = 180.0
    let bars: [(topX0: CGFloat, topX1: CGFloat, botX0: CGFloat, botX1: CGFloat)] = [
        (332.5, 365.5, 372.0, 394.0),
        (372.0, 401.5, 400.5, 420.0),
        (408.0, 434.0, 426.5, 440.0)
    ]
    for b in bars {
        p.addPath(makePolygon([
            CGPoint(x: b.topX0, y: topY),
            CGPoint(x: b.topX1, y: topY),
            CGPoint(x: b.botX1, y: botY),
            CGPoint(x: b.botX0, y: botY)
        ]))
    }
    return p
}()

// ============================================================================
// 2. BOTTOM ROW (CAM): MELDED HORIZONTALLY INTO ONE CONTINUOUS CAMERA BODY BASE
// ============================================================================
let rawCBase = mappedGlyph("c", x: 64, y: 228, w: 146, h: 148, topShiftX: -14, bottomShiftX: 0)
let cTopLeftShoulder = makePolygon([
    CGPoint(x: 78.0, y: 228.5),
    CGPoint(x: 98.0, y: 228.5),
    CGPoint(x: 98.0, y: 256.0),
    CGPoint(x: 77.5, y: 256.0)
])
let rawC = rawCBase.union(cTopLeftShoulder)
let rawA = mappedGlyph("a", x: 162, y: 228, w: 152, h: 148, topShiftX: 2, bottomShiftX: -4)
let rawM = mappedGlyph("m", x: 278, y: 228, w: 162, h: 148, topShiftX: 0, bottomShiftX: 16)

let vBridgeLeft = makePolygon([
    CGPoint(x: 112, y: 224),
    CGPoint(x: 136, y: 224),
    CGPoint(x: 136, y: 242),
    CGPoint(x: 112, y: 242)
])
let vBridgeMid = makePolygon([
    CGPoint(x: 194, y: 220),
    CGPoint(x: 234, y: 220),
    CGPoint(x: 234, y: 258),
    CGPoint(x: 194, y: 258)
])
let vBridgeRight = makePolygon([
    CGPoint(x: 312, y: 220),
    CGPoint(x: 365, y: 220),
    CGPoint(x: 365, y: 254),
    CGPoint(x: 312, y: 254)
])
let cBottomFlush = makePolygon([
    CGPoint(x: 115, y: 360),
    CGPoint(x: 196, y: 360),
    CGPoint(x: 196, y: 376),
    CGPoint(x: 115, y: 376)
])
let amBottomFlush = makePolygon([
    CGPoint(x: 262, y: 360),
    CGPoint(x: 296, y: 360),
    CGPoint(x: 296, y: 376),
    CGPoint(x: 262, y: 376)
])

let unionTopRaw = rawP
    .union(rawX)
    .union(rawL)
    .union(pentaprismMelded)
    .union(rawSpeedBars)
    .union(vBridgeLeft)
    .union(vBridgeMid)
    .union(vBridgeRight)

let unionBotRaw = rawC
    .union(rawA)
    .union(rawM)
    .union(cBottomFlush)
    .union(amBottomFlush)

let allLettersPath = unionTopRaw.union(unionBotRaw)

let wordmarkDarkTrench = allLettersPath.union(
    allLettersPath.copy(strokingWithWidth: 7.5, lineCap: .round, lineJoin: .round, miterLimit: 2.5)
)
let interiorCorePolygon = makePolygon([
    CGPoint(x: 84, y: 83.75),
    CGPoint(x: 420, y: 83.75),
    CGPoint(x: 426, y: 226),
    CGPoint(x: 420, y: 379.75),
    CGPoint(x: 84, y: 379.75)
])
let wordmarkChassisBase = wordmarkDarkTrench.union(interiorCorePolygon)

// ============================================================================
// 3. EXPORT STANDALONE BTTF WORDMARK
// ============================================================================
do {
    let wmCyanOuterStroke = wordmarkChassisBase.copy(strokingWithWidth: 18.0, lineCap: .round, lineJoin: .round, miterLimit: 2.5)
    let wmCyanHull = wordmarkChassisBase.union(wmCyanOuterStroke)
    let hullBB = wmCyanHull.boundingBoxOfPath
    let uniformPad: CGFloat = 8.0
    let vbX = hullBB.minX - uniformPad
    let vbY = hullBB.minY - uniformPad
    let vbW = hullBB.width + uniformPad * 2.0
    let vbH = hullBB.height + uniformPad * 2.0
    let outW = Int(round(vbW * 2.0))
    let outH = Int(round(vbH * 2.0))

    let svgContent = """
    <?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="\(String(format: "%.2f %.2f %.2f %.2f", vbX, vbY, vbW, vbH))" width="\(outW)" height="\(outH)">
      <defs>
        <linearGradient id="bttf_unified_glam" gradientUnits="userSpaceOnUse" x1="250" y1="34" x2="250" y2="376">
          <stop offset="0.00" stop-color="#F04E27"/>
          <stop offset="0.14" stop-color="#F04E27"/>
          <stop offset="0.30" stop-color="#FF8400"/>
          <stop offset="0.44" stop-color="#FFC91B"/>
          <stop offset="0.50" stop-color="#FFF36B"/>
          <stop offset="0.56" stop-color="#FFC91B"/>
          <stop offset="0.72" stop-color="#FF8400"/>
          <stop offset="0.88" stop-color="#F04E27"/>
          <stop offset="1.00" stop-color="#F04E27"/>
        </linearGradient>
        <linearGradient id="bttf_cyan" gradientUnits="userSpaceOnUse" x1="50" y1="30" x2="450" y2="385">
          <stop offset="0.00" stop-color="#38E4FF"/>
          <stop offset="0.30" stop-color="#00AAF2"/>
          <stop offset="0.70" stop-color="#0089CF"/>
          <stop offset="1.00" stop-color="#0072B5"/>
        </linearGradient>
      </defs>
      <path d="\(pathToSVG(wmCyanHull))" fill="url(#bttf_cyan)" stroke="#A3F6FF" stroke-width="2.0" stroke-linejoin="round"/>
      <path d="\(pathToSVG(wordmarkDarkTrench))" fill="#040811"/>
      <path d="\(pathToSVG(allLettersPath))" fill="url(#bttf_unified_glam)"/>
      <path d="\(pathToSVG(allLettersPath))" fill="none" stroke="#FFF6AA" stroke-width="1.4" stroke-linejoin="round" opacity="0.68"/>
      <g transform="translate(178, 40)">
        <polygon points="0,-10 2.2,-2.2 10,0 2.2,2.2 0,10 -2.2,2.2 -10,0 -2.2,-2.2" fill="#FFFFFF"/>
        <circle cx="0" cy="0" r="2.6" fill="#FFF7B0"/>
      </g>
    </svg>
    """
    let outSVGPath = "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_wordmark_bttf.svg"
    try svgContent.write(toFile: outSVGPath, atomically: true, encoding: .utf8)
    try renderPNG(fromSVG: outSVGPath, width: outW, height: outH, darkBackground: false, to: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_wordmark_bttf.png")
}

// ============================================================================
// 4. HELPER FOR FOX BTTF LAYERS (SHARED BY STANDALONE FOX & UNIFIED LOCKUP)
// ============================================================================
func buildFoxBTTFLayers(foxRightX: CGFloat, cyanEars: Bool) -> (foxPath: CGPath, foxDarkTrench: CGPath, foxChassisBase: CGPath, mapPt: (CGFloat, CGFloat) -> CGPoint) {
    let (foxPath, mapPt) = loadFoxPath(foxRightX: foxRightX)

    var foxDarkCore = makePolygon([
        mapPt(330, 326), mapPt(695, 326),
        mapPt(650, 415), mapPt(665, 510),
        mapPt(512, 665),
        mapPt(360, 510), mapPt(375, 415)
    ])
    if !cyanEars {
        let leftEarDark = makePolygon([
            mapPt(385, 105), mapPt(455, 195), mapPt(420, 238), mapPt(380, 238)
        ])
        let rightEarDark = makePolygon([
            mapPt(640, 105), mapPt(645, 238), mapPt(605, 238), mapPt(570, 195)
        ])
        foxDarkCore = foxDarkCore.union(leftEarDark).union(rightEarDark)
    }

    let foxDarkTrench = foxPath
        .union(foxPath.copy(strokingWithWidth: 5.2, lineCap: .round, lineJoin: .round, miterLimit: 2.5))
        .union(foxDarkCore)

    var foxCyanPoints: [CGPoint] = []
    if cyanEars {
        foxCyanPoints.append(contentsOf: [
            mapPt(395, 105), mapPt(455, 200), mapPt(512, 165), mapPt(570, 200), mapPt(630, 105),
            mapPt(640, 240)
        ])
    } else {
        foxCyanPoints.append(contentsOf: [
            mapPt(365, 245), mapPt(660, 245)
        ])
    }
    foxCyanPoints.append(contentsOf: [
        mapPt(660, 250), mapPt(630, 335),
        mapPt(700, 340), mapPt(645, 415), mapPt(735, 425),
        mapPt(720, 775), mapPt(650, 835), mapPt(580, 895), mapPt(512, 945),
        mapPt(445, 895), mapPt(375, 835), mapPt(305, 775),
        mapPt(290, 425), mapPt(380, 415), mapPt(325, 340),
        mapPt(395, 335), mapPt(365, 250)
    ])
    if cyanEars {
        foxCyanPoints.append(mapPt(385, 240))
    }
    let foxCyanCore = makePolygon(foxCyanPoints)
    let foxChassisBase = foxDarkTrench.union(foxCyanCore)
    return (foxPath, foxDarkTrench, foxChassisBase, mapPt)
}

// ============================================================================
// 5. EXPORT STANDALONE 80S GLAM BTTF FOX ICON (EXACT SAME GRADIENT & STYLE)
// ============================================================================
func exportStandaloneBTTFFox(svgPath: String, pngPath: String, darkPreviewPath: String?, cyanEars: Bool) throws {
    let (foxPath, foxDarkTrench, foxChassisBase, mapPt) = buildFoxBTTFLayers(foxRightX: 187.5, cyanEars: cyanEars)
    let foxCyanStroke = foxChassisBase.copy(strokingWithWidth: 18.0, lineCap: .round, lineJoin: .round, miterLimit: 2.5)
    let foxCyanHull = foxChassisBase.union(foxCyanStroke)

    let hullBB = foxCyanHull.boundingBoxOfPath
    let pad: CGFloat = 10.0
    let vbH = hullBB.height + pad * 2.0
    let vbW = vbH // Square 1:1 viewBox for standalone icon
    let vbX = hullBB.midX - vbW / 2.0
    let vbY = hullBB.minY - pad

    let glint1 = mapPt(365, 242)
    let glint2 = mapPt(652, 495)

    let svg = """
    <?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="\(String(format: "%.2f %.2f %.2f %.2f", vbX, vbY, vbW, vbH))" width="1024" height="1024">
      <defs>
        <linearGradient id="bttf_unified_glam" gradientUnits="userSpaceOnUse" x1="100" y1="34" x2="100" y2="376">
          <stop offset="0.00" stop-color="#F04E27"/>
          <stop offset="0.14" stop-color="#F04E27"/>
          <stop offset="0.30" stop-color="#FF8400"/>
          <stop offset="0.44" stop-color="#FFC91B"/>
          <stop offset="0.50" stop-color="#FFF36B"/>
          <stop offset="0.56" stop-color="#FFC91B"/>
          <stop offset="0.72" stop-color="#FF8400"/>
          <stop offset="0.88" stop-color="#F04E27"/>
          <stop offset="1.00" stop-color="#F04E27"/>
        </linearGradient>
        <linearGradient id="bttf_cyan" gradientUnits="userSpaceOnUse" x1="0" y1="30" x2="200" y2="385">
          <stop offset="0.00" stop-color="#38E4FF"/>
          <stop offset="0.30" stop-color="#00AAF2"/>
          <stop offset="0.70" stop-color="#0089CF"/>
          <stop offset="1.00" stop-color="#0072B5"/>
        </linearGradient>
      </defs>
      <path d="\(pathToSVG(foxCyanHull))" fill="url(#bttf_cyan)" stroke="#A3F6FF" stroke-width="2.0" stroke-linejoin="round"/>
      <path d="\(pathToSVG(foxDarkTrench))" fill="#040811"/>
      <path d="\(pathToSVG(foxPath))" fill="url(#bttf_unified_glam)"/>
      <path d="\(pathToSVG(foxPath))" fill="none" stroke="#FFF6AA" stroke-width="1.35" stroke-linejoin="round" opacity="0.68"/>
      <g transform="translate(\(String(format: "%.1f, %.1f", glint1.x, glint1.y)))">
        <polygon points="0,-9 2.0,-2.0 9,0 2.0,2.0 0,9 -2.0,2.0 -9,0 -2.0,-2.0" fill="#FFFFFF"/>
        <circle cx="0" cy="0" r="2.3" fill="#FFF7B0"/>
      </g>
      <g transform="translate(\(String(format: "%.1f, %.1f", glint2.x, glint2.y)))">
        <polygon points="0,-8 1.8,-1.8 8,0 1.8,1.8 0,8 -1.8,1.8 -8,0 -1.8,-1.8" fill="#FFFFFF"/>
        <circle cx="0" cy="0" r="2.0" fill="#FFF7B0"/>
      </g>
    </svg>
    """
    try svg.write(toFile: svgPath, atomically: true, encoding: .utf8)
    try renderPNG(fromSVG: svgPath, width: 1024, height: 1024, darkBackground: false, to: pngPath)
    if let darkPreviewPath = darkPreviewPath {
        try renderPNG(fromSVG: svgPath, width: 1024, height: 1024, darkBackground: true, to: darkPreviewPath)
    }
}

try exportStandaloneBTTFFox(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_bttf_glam.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_bttf_glam.png",
    darkPreviewPath: "/Users/aaryn/.gemini/antigravity/brain/bf1fd10a-c879-4158-8108-0d88754929dc/pxl_cam_logo_bttf_glam_preview.png",
    cyanEars: false
)
try exportStandaloneBTTFFox(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/candidate_4_coral_cyber_fox_bttf_unified_vector.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/candidate_4_coral_cyber_fox_bttf_unified_vector.png",
    darkPreviewPath: nil,
    cyanEars: false
)
try exportStandaloneBTTFFox(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/candidate_4_coral_cyber_fox_bttf_cyan_ears_vector.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/candidate_4_coral_cyber_fox_bttf_cyan_ears_vector.png",
    darkPreviewPath: nil,
    cyanEars: true
)

// ============================================================================
// 6. EXPORT UNIFIED COHESIVE FOX + WORDMARK LOCKUP (ONE UNIT, SAME GRADIENT)
// ============================================================================
func exportUnifiedLockup(
    svgPath: String,
    pngPath: String,
    darkPreviewPath: String?,
    foxRightX: CGFloat,
    fuseHulls: Bool,
    cyanEars: Bool
) throws {
    let (foxPath, foxDarkTrench, foxChassisBase, mapPt) = buildFoxBTTFLayers(foxRightX: foxRightX, cyanEars: cyanEars)
    let combinedFaces = foxPath.union(allLettersPath)
    let darkInnerTrench = foxDarkTrench.union(wordmarkDarkTrench)

    var cyanChassisBase = foxChassisBase.union(wordmarkChassisBase)
    if fuseHulls {
        // Fill the narrow interior pocket between the Fox's right collar tip (mapPt(701, 340))
        // and rightmost bottom pixel square (mapPt(712, 765)) and P/C's left stem,
        // so the unified cyan hull has ZERO artificial corners and flows directly from the collar tip and pixel step!
        let collarTip = mapPt(701, 340)
        let rightPixelBottom = mapPt(712, 765)
        let waistBridge = makePolygon([
            collarTip,
            CGPoint(x: 78.0, y: collarTip.y),
            CGPoint(x: 64.0, y: rightPixelBottom.y),
            rightPixelBottom,
            mapPt(650, 415)
        ])
        cyanChassisBase = cyanChassisBase.union(waistBridge)
    }

    let cyanOuterStroke = cyanChassisBase.copy(strokingWithWidth: 18.0, lineCap: .round, lineJoin: .round, miterLimit: 2.5)
    let unifiedCyanHull = cyanChassisBase.union(cyanOuterStroke)

    let hullBB = unifiedCyanHull.boundingBoxOfPath
    let uniformPad: CGFloat = 10.0
    let vbX = hullBB.minX - uniformPad
    let vbY = hullBB.minY - uniformPad
    let vbW = hullBB.width + uniformPad * 2.0
    let vbH = hullBB.height + uniformPad * 2.0

    let outW = Int(round(vbW * 2.5))
    let outH = Int(round(vbH * 2.5))

    let foxGlintPt1 = mapPt(365, 242)
    let foxGlintPt2 = mapPt(652, 495)

    let svg = """
    <?xml version="1.0" encoding="UTF-8"?>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="\(String(format: "%.2f %.2f %.2f %.2f", vbX, vbY, vbW, vbH))" width="\(outW)" height="\(outH)">
      <defs>
        <!-- Continuous Dual-Mirrored Back to the Future Gradient Shared Across Both Fox & PXL/CAM -->
        <linearGradient id="bttf_unified_glam" gradientUnits="userSpaceOnUse" x1="150" y1="34" x2="150" y2="376">
          <stop offset="0.00" stop-color="#F04E27"/>
          <stop offset="0.14" stop-color="#F04E27"/>
          <stop offset="0.30" stop-color="#FF8400"/>
          <stop offset="0.44" stop-color="#FFC91B"/>
          <stop offset="0.50" stop-color="#FFF36B"/>
          <stop offset="0.56" stop-color="#FFC91B"/>
          <stop offset="0.72" stop-color="#FF8400"/>
          <stop offset="0.88" stop-color="#F04E27"/>
          <stop offset="1.00" stop-color="#F04E27"/>
        </linearGradient>

        <!-- Continuous 80s Glam Electric Cyan Hull Gradient -->
        <linearGradient id="bttf_cyan" gradientUnits="userSpaceOnUse" x1="-140" y1="30" x2="450" y2="385">
          <stop offset="0.00" stop-color="#38E4FF"/>
          <stop offset="0.30" stop-color="#00AAF2"/>
          <stop offset="0.70" stop-color="#0089CF"/>
          <stop offset="1.00" stop-color="#0072B5"/>
        </linearGradient>
      </defs>

      <!-- 1. Unified One-Unit BTTF Electric Cyan Hull -->
      <path d="\(pathToSVG(unifiedCyanHull))" fill="url(#bttf_cyan)" stroke="#A3F6FF" stroke-width="2.0" stroke-linejoin="round"/>

      <!-- 2. Unified Dark Inner Bevel Trench -->
      <path d="\(pathToSVG(darkInnerTrench))" fill="#040811"/>

      <!-- 3. Single Unified Path for Fox + PXL/CAM Filled with the Exact Same Continuous Gradient -->
      <path d="\(pathToSVG(combinedFaces))" fill="url(#bttf_unified_glam)"/>
      <path d="\(pathToSVG(combinedFaces))" fill="none" stroke="#FFF6AA" stroke-width="1.35" stroke-linejoin="round" opacity="0.68"/>

      <!-- 4. 80s Glam Specular Starburst Sparks Across Both Fox & Wordmark -->
      <g transform="translate(178, 40)">
        <polygon points="0,-10 2.2,-2.2 10,0 2.2,2.2 0,10 -2.2,2.2 -10,0 -2.2,-2.2" fill="#FFFFFF"/>
        <circle cx="0" cy="0" r="2.6" fill="#FFF7B0"/>
      </g>
      <g transform="translate(\(String(format: "%.1f, %.1f", foxGlintPt1.x, foxGlintPt1.y)))">
        <polygon points="0,-9 2.0,-2.0 9,0 2.0,2.0 0,9 -2.0,2.0 -9,0 -2.0,-2.0" fill="#FFFFFF"/>
        <circle cx="0" cy="0" r="2.3" fill="#FFF7B0"/>
      </g>
      <g transform="translate(\(String(format: "%.1f, %.1f", foxGlintPt2.x, foxGlintPt2.y)))">
        <polygon points="0,-8 1.8,-1.8 8,0 1.8,1.8 0,8 -1.8,1.8 -8,0 -1.8,-1.8" fill="#FFFFFF"/>
        <circle cx="0" cy="0" r="2.0" fill="#FFF7B0"/>
      </g>
    </svg>
    """
    try svg.write(toFile: svgPath, atomically: true, encoding: .utf8)
    try renderPNG(fromSVG: svgPath, width: outW, height: outH, darkBackground: false, to: pngPath)
    if let darkPreviewPath = darkPreviewPath {
        try renderPNG(fromSVG: svgPath, width: outW, height: outH, darkBackground: true, to: darkPreviewPath)
    }
    print("Exported \(svgPath) (\(outW)x\(outH))")
}

// Primary unified lockup (One Unit Cohesive Lockup, Dark Inner Ears so Cyan 80s Shades Pop)
try exportUnifiedLockup(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_top_left_overlay.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_top_left_overlay.png",
    darkPreviewPath: "/Users/aaryn/.gemini/antigravity/brain/bf1fd10a-c879-4158-8108-0d88754929dc/pxl_cam_top_left_overlay_preview.png",
    foxRightX: 56.0,
    fuseHulls: true,
    cyanEars: false
)

// Candidate variations in assets/candidates/
try exportUnifiedLockup(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/pxl_cam_top_left_overlay_bttf_unified_dark_ears.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/pxl_cam_top_left_overlay_bttf_unified_dark_ears.png",
    darkPreviewPath: nil,
    foxRightX: 56.0,
    fuseHulls: true,
    cyanEars: false
)
try exportUnifiedLockup(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/pxl_cam_top_left_overlay_bttf_unified_cyan_ears.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/pxl_cam_top_left_overlay_bttf_unified_cyan_ears.png",
    darkPreviewPath: "/Users/aaryn/.gemini/antigravity/brain/bf1fd10a-c879-4158-8108-0d88754929dc/pxl_cam_top_left_overlay_cyan_ears_preview.png",
    foxRightX: 56.0,
    fuseHulls: true,
    cyanEars: true
)
try exportUnifiedLockup(
    svgPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/pxl_cam_top_left_overlay_bttf_sidebyside.svg",
    pngPath: "/Users/aaryn/Projects/pxl-cam/assets/candidates/pxl_cam_top_left_overlay_bttf_sidebyside.png",
    darkPreviewPath: "/Users/aaryn/.gemini/antigravity/brain/bf1fd10a-c879-4158-8108-0d88754929dc/pxl_cam_top_left_overlay_sidebyside_preview.png",
    foxRightX: 38.0,
    fuseHulls: false,
    cyanEars: false
)
