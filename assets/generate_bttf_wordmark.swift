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

func splitSubpaths(_ path: CGPath) -> [CGPath] {
    var result: [CGPath] = []
    var current = CGMutablePath()
    var hasElements = false
    path.applyWithBlock { ptr in
        let el = ptr.pointee
        switch el.type {
        case .moveToPoint:
            if hasElements {
                result.append(current)
                current = CGMutablePath()
            }
            current.move(to: el.points[0])
            hasElements = true
        case .addLineToPoint:
            current.addLine(to: el.points[0])
        case .addQuadCurveToPoint:
            current.addQuadCurve(to: el.points[1], control: el.points[0])
        case .addCurveToPoint:
            current.addCurve(to: el.points[2], control1: el.points[0], control2: el.points[1])
        case .closeSubpath:
            current.closeSubpath()
        @unknown default:
            break
        }
    }
    if hasElements {
        result.append(current)
    }
    return result
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

// Keep all edges level and true so the border is 100% uniform on all 4 sides
func bttfWarp(_ pt: CGPoint) -> CGPoint {
    return pt
}

// ============================================================================
// 1. TOP ROW (PXL): MELDED HORIZONTALLY INTO ONE CONTINUOUS CAMERA BODY TOP
// ============================================================================
// P's left edge runs smoothly from x=68 (top) to x=72 (waist), meeting C's upper-left at x=72 with zero step!
let rawP = mappedGlyph("P", x: 64, y: 80, w: 148, h: 144, topShiftX: 4, bottomShiftX: -30)
let rawX = mappedGlyph("X", x: 160, y: 80, w: 162, h: 144, topShiftX: 0, bottomShiftX: -6)
let rawL = mappedGlyph("L", x: 284, y: 80, w: 156, h: 144, topShiftX: 0, bottomShiftX: 0)

// Camera Pentaprism Viewfinder Jut-Out & Top Notch MELDED directly onto the top of PXL (at y=80)!
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
let pentaprismMelded = pentaprismOuter
    .subtracting(viewfinderWindowHole)

// Precision BTTF chevron speed-bars in the top-right camera notch:
// - Top of speed-bars (topY = 80.0) is 100% flush with the top of P (y=80) and L's stem (y=80)!
// - Rightmost bar's outer edge runs from x=434 (top) to x=440 (waist), symmetric with P/C on the left!
// - botY = 180.0 leaves a uniform 6.5px dark bevel channel above L's foot (matching the 6.5px channels between bars!)
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
// C's upper-left meets P's bottom-left at x=72 and runs to x=64 at bottom; M runs from x=440 at waist to x=434 at bottom!
let rawCBase = mappedGlyph("c", x: 64, y: 228, w: 146, h: 148, topShiftX: -14, bottomShiftX: 0)
// Square off C's top-left outer shoulder strictly below y=228.5 so the 4.5px horizontal dark bevel line
// between P (y=224) and C (y=228.5) remains 100% intact while eliminating the left-waist step!
let cTopLeftShoulder = makePolygon([
    CGPoint(x: 78.0, y: 228.5),
    CGPoint(x: 98.0, y: 228.5),
    CGPoint(x: 98.0, y: 256.0),
    CGPoint(x: 77.5, y: 256.0)
])
let rawC = rawCBase.union(cTopLeftShoulder)
let rawA = mappedGlyph("a", x: 162, y: 228, w: 152, h: 148, topShiftX: 2, bottomShiftX: -4)
let rawM = mappedGlyph("m", x: 278, y: 228, w: 162, h: 148, topShiftX: 0, bottomShiftX: 16)

// Internal vertical fusion bridges so PXL and CAM meld into one unit while keeping P and C distinct on the left
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
// Flush bottom-edge bridge between C's lower bar and A's left leg (starting inside C at x=115 so C's left curve stays round)
let cBottomFlush = makePolygon([
    CGPoint(x: 115, y: 360),
    CGPoint(x: 196, y: 360),
    CGPoint(x: 196, y: 376),
    CGPoint(x: 115, y: 376)
])
// Flush bottom-edge bridge between A's right leg and M's left leg
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

let meldedAllLettersRaw = unionTopRaw.union(unionBotRaw)
let allLettersPath = warpPath(meldedAllLettersRaw, mapPoint: bttfWarp)

// ============================================================================
// 3. 100% UNIFORM-DISTANCE BACK TO THE FUTURE #0089CF ELECTRIC CYAN BORDER
// ============================================================================
// Stroke allLettersPath with 7.5px (3.75px radius) for the dark bevel trench, and stroke the camera chassis
// directly with 18.0px (9.0px radius) for unifiedCyanHull so every single point on all 4 sides of the border
// is at an exact, mathematically invariant 9.0px distance from the content!
let darkInnerTrench = allLettersPath.union(
    allLettersPath.copy(strokingWithWidth: 7.5, lineCap: .round, lineJoin: .round, miterLimit: 2.5)
)

// Camera body interior core polygon spans the interior counters down to y=376 (flush with the bottom of C, A, M)
// so A and M's bottom arches have the same electric cyan background as P, X, L, C, A and the bottom border is 100% uniform!
let interiorCorePolygon = makePolygon([
    CGPoint(x: 84, y: 83.75),
    CGPoint(x: 420, y: 83.75),
    CGPoint(x: 426, y: 226),
    CGPoint(x: 420, y: 379.75),
    CGPoint(x: 84, y: 379.75)
])

let cyanChassisBase = darkInnerTrench.union(interiorCorePolygon)
let cyanOuterStroke = cyanChassisBase.copy(strokingWithWidth: 18.0, lineCap: .round, lineJoin: .round, miterLimit: 2.5)
let unifiedCyanHull = cyanChassisBase.union(cyanOuterStroke)

let svgCyanHull = pathToSVG(unifiedCyanHull)
let svgDarkTrench = pathToSVG(darkInnerTrench)
let svgAllLetters = pathToSVG(allLettersPath)

// Compute exact bounding box of unifiedCyanHull (+ 1.0px half-stroke of #A3F6FF) with uniform padding on all 4 sides!
let hullBB = unifiedCyanHull.boundingBoxOfPath
let uniformPad: CGFloat = 8.0
let vbX = hullBB.minX - uniformPad
let vbY = hullBB.minY - uniformPad
let vbW = hullBB.width + uniformPad * 2.0
let vbH = hullBB.height + uniformPad * 2.0

let svgBodyElements = """
  <defs>
    <!-- Unified Dual-Mirrored Back to the Future Gradient (Fire-Orange -> Gold Center -> Fire-Orange) -->
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

    <!-- 80s Glam Electric Cyan Outer Hull Gradient (#0089CF + neon cyan top highlight) -->
    <linearGradient id="bttf_cyan" gradientUnits="userSpaceOnUse" x1="50" y1="30" x2="450" y2="385">
      <stop offset="0.00" stop-color="#38E4FF"/>
      <stop offset="0.30" stop-color="#00AAF2"/>
      <stop offset="0.70" stop-color="#0089CF"/>
      <stop offset="1.00" stop-color="#0072B5"/>
    </linearGradient>
  </defs>

  <!-- 1. Uniform-Distance BTTF Electric Cyan Camera-Shaped Hull (#0089CF) -->
  <path d="\(svgCyanHull)" fill="url(#bttf_cyan)" stroke="#A3F6FF" stroke-width="2.0" stroke-linejoin="round"/>

  <!-- 2. Uniform-Distance Dark Inner Bevel Trench -->
  <path d="\(svgDarkTrench)" fill="#040811"/>

  <!-- 3. ONE UNIT MELDED LETTERS: PXL on top of CAM + Pentaprism Jut-Out + Top Notch + BTTF Speed-Bars -->
  <path d="\(svgAllLetters)" fill="url(#bttf_unified_glam)"/>
  <path d="\(svgAllLetters)" fill="none" stroke="#FFF6AA" stroke-width="1.4" stroke-linejoin="round" opacity="0.68"/>

  <!-- 4. 80s Glam Specular Starburst Sparks (Kept inside the letter faces) -->
  <g transform="translate(178, 40)">
    <polygon points="0,-10 2.2,-2.2 10,0 2.2,2.2 0,10 -2.2,2.2 -10,0 -2.2,-2.2" fill="#FFFFFF"/>
    <circle cx="0" cy="0" r="2.6" fill="#FFF7B0"/>
  </g>
"""

let outW = Int(round(vbW * 2.0))
let outH = Int(round(vbH * 2.0))

let svgContent = """
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="\(String(format: "%.2f %.2f %.2f %.2f", vbX, vbY, vbW, vbH))" width="\(outW)" height="\(outH)">
\(svgBodyElements)
</svg>
"""

let outSVGPath = "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_wordmark_bttf.svg"
try svgContent.write(toFile: outSVGPath, atomically: true, encoding: .utf8)

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

try renderPNG(fromSVG: outSVGPath, width: outW, height: outH, darkBackground: false, to: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_wordmark_bttf.png")
try renderPNG(fromSVG: outSVGPath, width: outW, height: outH, darkBackground: true, to: "/Users/aaryn/.gemini/antigravity/brain/60b30b83-5b0f-42ea-af4d-25258f00186a/pxl_cam_wordmark_bttf_preview.png")

// Helper to find exact non-transparent pixel bounding box of a bitmap
func nonTransparentBounds(_ img: NSImage) -> NSRect {
    guard let tiff = img.tiffRepresentation,
          let rep = NSBitmapImageRep(data: tiff) else {
        return NSRect(origin: .zero, size: img.size)
    }
    let w = rep.pixelsWide
    let h = rep.pixelsHigh
    var minX = w, maxX = 0, minY = h, maxY = 0
    for y in 0..<h {
        for x in 0..<w {
            if let c = rep.colorAt(x: x, y: y), c.alphaComponent > 0.02 {
                if x < minX { minX = x }
                if x > maxX { maxX = x }
                if y < minY { minY = y }
                if y > maxY { maxY = y }
            }
        }
    }
    if minX > maxX { return NSRect(x: 0, y: 0, width: w, height: h) }
    // Convert from top-down bitmap y to AppKit bottom-up y
    let appKitMinY = h - 1 - maxY
    let appKitH = maxY - minY + 1
    return NSRect(x: minX, y: appKitMinY, width: maxX - minX + 1, height: appKitH)
}

// Create the combined Top-Left Overlay Lockup with EXACT uniform margin on all 4 sides
if let foxImg = NSImage(contentsOfFile: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_scaled_transparent.png"),
   let wordmarkImg = NSImage(contentsOfFile: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_wordmark_bttf.png") {
    let foxCrop = nonTransparentBounds(foxImg)
    let wmCrop = nonTransparentBounds(wordmarkImg)

    let targetContentH: CGFloat = 720.0
    let foxDrawW = round(targetContentH * (foxCrop.width / foxCrop.height))
    let wmDrawW = round(targetContentH * (wmCrop.width / wmCrop.height))
    let gapBetween: CGFloat = 28.0
    let uniformMargin: CGFloat = 36.0

    let lockupW = Int(round(uniformMargin * 2.0 + foxDrawW + gapBetween + wmDrawW))
    let lockupH = Int(round(uniformMargin * 2.0 + targetContentH))

    func renderLockup(darkBackground: Bool, drawBorderPreview: Bool, to outPath: String) throws {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil,
            pixelsWide: lockupW,
            pixelsHigh: lockupH,
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
            NSRect(x: 0, y: 0, width: lockupW, height: lockupH).fill()
        }
        foxImg.draw(
            in: NSRect(x: uniformMargin, y: uniformMargin, width: foxDrawW, height: targetContentH),
            from: foxCrop,
            operation: .sourceOver,
            fraction: 1.0
        )
        wordmarkImg.draw(
            in: NSRect(x: uniformMargin + foxDrawW + gapBetween, y: uniformMargin, width: wmDrawW, height: targetContentH),
            from: wmCrop,
            operation: .sourceOver,
            fraction: 1.0
        )
        if drawBorderPreview {
            let borderPath = NSBezierPath(roundedRect: NSRect(x: 4, y: 4, width: CGFloat(lockupW) - 8, height: CGFloat(lockupH) - 8), xRadius: 44, yRadius: 44)
            borderPath.lineWidth = 4.0
            NSColor(calibratedRed: 1.0, green: 0.52, blue: 0.0, alpha: 0.45).setStroke()
            borderPath.stroke()
        }
        NSGraphicsContext.restoreGraphicsState()
        if let pngData = rep.representation(using: .png, properties: [:]) {
            try pngData.write(to: URL(fileURLWithPath: outPath))
        }
    }

    try renderLockup(darkBackground: false, drawBorderPreview: false, to: "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_top_left_overlay.png")
    try renderLockup(darkBackground: true, drawBorderPreview: true, to: "/Users/aaryn/.gemini/antigravity/brain/60b30b83-5b0f-42ea-af4d-25258f00186a/pxl_cam_top_left_overlay_preview.png")
    print("Lockup dimensions: \(lockupW)x\(lockupH) (aspect: \(String(format: "%.4f", Double(lockupW)/Double(lockupH))))")
}

print("Exported uniform-border BTTF wordmark (\(outW)x\(outH)) and top-left overlay lockups!")

