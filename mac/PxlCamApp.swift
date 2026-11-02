import Cocoa
import Metal
import MetalKit
import CoreVideo
import CoreMedia
import VideoToolbox
import CoreImage
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Vision
import AVFoundation
import Foundation
import Darwin

// MARK: - Metal Shader Source (Zero-Copy Texture + Low-Light Shadow Lift + Studio ISP)
private let metalShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

struct ISPUniforms {
    float2 texelSize;
    int ispEnabled;
    int mirrorHorizontal;
    int flipVertical;
    int rotationQuadrants; // 0, 1 (90 CW), 2 (180), 3 (270 CW)
    float brightnessGain;
    float flashDimFactor;
    float shadowLift;
    float contrastBoost;
    float saturationBoost;
    float sharpenStrength;
    float denoiseStrength;
    float warmthShift;
};

vertex VertexOut vertexPassthrough(uint vertexID [[vertex_id]],
                                   constant ISPUniforms &u [[buffer(0)]]) {
    const float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    const float2 coords[4] = {
        float2(0.0, 1.0),
        float2(1.0, 1.0),
        float2(0.0, 0.0),
        float2(1.0, 0.0)
    };
    VertexOut out;
    out.position = float4(positions[vertexID], 0.0, 1.0);
    float2 uv = coords[vertexID];

    if (u.mirrorHorizontal != 0) {
        uv.x = 1.0 - uv.x;
    }
    if (u.flipVertical != 0) {
        uv.y = 1.0 - uv.y;
    }
    if (u.rotationQuadrants == 1) {
        uv = float2(uv.y, 1.0 - uv.x);
    } else if (u.rotationQuadrants == 2) {
        uv = float2(1.0 - uv.x, 1.0 - uv.y);
    } else if (u.rotationQuadrants == 3) {
        uv = float2(1.0 - uv.y, uv.x);
    }

    out.texCoord = uv;
    return out;
}

// Multi-scale (1.0px + 2.0px) impulse + YCbCr chroma/luma neighborhood sampler:
// Preserves 100% of 1px details, text, catchlights, and glasses frames while removing isolated sensor hot pixels
// and low-light 4:2:0 chroma blotches.
inline float3 sampleMultiScaleNeighborhood(texture2d<float> tex,
                                           sampler s,
                                           float2 uv,
                                           float2 texel,
                                           thread float3 &outMin1,
                                           thread float3 &outMax1,
                                           thread float3 &outBlur1,
                                           thread float3 &outBlur2,
                                           thread float3 &outCleanChroma) {
    constexpr float3 kLumaW = float3(0.299, 0.587, 0.114);
    float3 c = tex.sample(s, uv).rgb;
    float2 dx1 = float2(texel.x, 0.0);
    float2 dy1 = float2(0.0, texel.y);

    // Inner 8-tap ring (radius 1.0px)
    float3 i0 = tex.sample(s, uv + dx1).rgb;
    float3 i1 = tex.sample(s, uv - dx1).rgb;
    float3 i2 = tex.sample(s, uv + dy1).rgb;
    float3 i3 = tex.sample(s, uv - dy1).rgb;
    float3 i4 = tex.sample(s, uv + dx1 + dy1).rgb;
    float3 i5 = tex.sample(s, uv - dx1 + dy1).rgb;
    float3 i6 = tex.sample(s, uv + dx1 - dy1).rgb;
    float3 i7 = tex.sample(s, uv - dx1 - dy1).rgb;

    float3 max1 = max(max(max(i0, i1), max(i2, i3)), max(max(i4, i5), max(i6, i7)));
    float3 min1 = min(min(min(i0, i1), min(i2, i3)), min(min(i4, i5), min(i6, i7)));

    // Only clamp extreme isolated single-pixel sensor spikes (>0.045 outside all 8 neighbors)
    // so genuine 1px fine texture, hair, text, and eye catchlights remain 100% intact.
    float3 cleanCenter = clamp(c, min1 - 0.045, max1 + 0.045);
    outMin1 = min1;
    outMax1 = max1;
    outBlur1 = 0.16 * (i0 + i1 + i2 + i3) + 0.09 * (i4 + i5 + i6 + i7);

    // Outer 8-tap ring (radius 1.85px) for chroma denoise and low-frequency detail reference
    float2 dx2 = dx1 * 1.85;
    float2 dy2 = dy1 * 1.85;
    float3 o0 = tex.sample(s, uv + dx2).rgb;
    float3 o1 = tex.sample(s, uv - dx2).rgb;
    float3 o2 = tex.sample(s, uv + dy2).rgb;
    float3 o3 = tex.sample(s, uv - dy2).rgb;
    float3 o4 = tex.sample(s, uv + dx2 * 0.75 + dy2 * 0.75).rgb;
    float3 o5 = tex.sample(s, uv - dx2 * 0.75 + dy2 * 0.75).rgb;
    float3 o6 = tex.sample(s, uv + dx2 * 0.75 - dy2 * 0.75).rgb;
    float3 o7 = tex.sample(s, uv - dx2 * 0.75 - dy2 * 0.75).rgb;

    outBlur2 = 0.16 * (o0 + o1 + o2 + o3) + 0.09 * (o4 + o5 + o6 + o7);

    // Edge-guided chroma denoise (suppresses CMOS chroma speckle without softening luma edges)
    float centerY = dot(cleanCenter, kLumaW);
    float3 chromaSum = (cleanCenter - centerY) * 2.2;
    float wSum = 2.2;
    float3 chromaTaps[8] = { i0, i1, i2, i3, o0, o1, o2, o3 };
    for (int k = 0; k < 8; ++k) {
        float tapY = dot(chromaTaps[k], kLumaW);
        float w = exp(-abs(tapY - centerY) * 22.0);
        chromaSum += (chromaTaps[k] - tapY) * w;
        wSum += w;
    }
    outCleanChroma = chromaSum / max(wSum, 1e-4);
    return cleanCenter;
}

fragment float4 passthroughFragment(VertexOut in [[stage_in]],
                                    texture2d<float> tex [[texture(0)]]) {
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    return float4(tex.sample(s, in.texCoord).rgb, 1.0);
}

fragment float4 studioISPFragment(VertexOut in [[stage_in]],
                                  texture2d<float> tex [[texture(0)]],
                                  texture2d<float> texPrev [[texture(1)]],
                                  constant ISPUniforms &u [[buffer(0)]]) {
    constexpr sampler s(mag_filter::linear, min_filter::linear, address::clamp_to_edge);
    constexpr float3 kLumaW = float3(0.299, 0.587, 0.114);

    float3 min1 = float3(0.0);
    float3 max1 = float3(1.0);
    float3 blur1 = float3(0.0);
    float3 blur2 = float3(0.0);
    float3 cleanChroma = float3(0.0);

    float3 cleanCurr = sampleMultiScaleNeighborhood(tex, s, in.texCoord, u.texelSize, min1, max1, blur1, blur2, cleanChroma);
    float rawY = dot(cleanCurr, kLumaW);

    // Reconstruct with denoised chroma first so color speckle never enters luma stages
    float chromaCleanMix = (u.ispEnabled != 0) ? clamp(0.30 + u.denoiseStrength * 0.50, 0.0, 0.85) : 0.0;
    float3 baseChroma = mix(cleanCurr - rawY, cleanChroma, chromaCleanMix);
    float3 working = clamp(float3(rawY) + baseChroma, 0.0, 1.0);

    // 1. Luminance-Preserving Filmic Shadow Lift & Exposure Curve FIRST (so recursive texPrev comparison is in same tone space)
    float preToneY = dot(working, kLumaW);
    float3 preToneChroma = working - preToneY;

    // Suppress CMOS dark-current color cast only in ultra-deep black floor (< 0.035)
    float shadowNeutralW = smoothstep(0.005, 0.040, preToneY);
    preToneChroma *= (0.50 + 0.50 * shadowNeutralW);

    float dimW = clamp(u.flashDimFactor, 0.05, 1.0);
    float effShadowLift = u.shadowLift * dimW * dimW;

    float liftedY = preToneY;
    if (effShadowLift > 0.001) {
        float gamma = 1.0 / (1.0 + effShadowLift * 1.15);
        float powerY = pow(clamp(liftedY, 0.0, 1.0), gamma);
        float maxRatio = 1.0 + effShadowLift * 3.8;
        liftedY = min(powerY, liftedY * maxRatio);
    }

    // Exposure / Brightness Gain with smooth filmic highlight roll-off
    float exposedY = liftedY * u.brightnessGain;
    if (u.brightnessGain > 1.12) {
        float shoulderK = (u.brightnessGain - 1.0) * 0.20;
        exposedY = (exposedY * (1.0 + exposedY * shoulderK * 0.35)) / (1.0 + exposedY * shoulderK);
    }
    exposedY = clamp(exposedY, 0.0, 1.0);

    // Toe-Preserving S-Curve Contrast (anchored at midtone 0.40)
    float contrastedY = exposedY;
    if (abs(u.contrastBoost - 1.0) > 0.005) {
        float pivot = 0.40;
        if (exposedY <= pivot && exposedY > 1e-5) {
            contrastedY = pivot * pow(exposedY / pivot, u.contrastBoost);
        } else if (exposedY > pivot) {
            float normHi = (exposedY - pivot) / (1.0 - pivot);
            contrastedY = pivot + (1.0 - pivot) * pow(clamp(normHi, 0.0, 1.0), 1.0 / max(u.contrastBoost, 0.7));
        }
    }
    contrastedY = clamp(contrastedY, 0.0, 1.0);

    float lumaRatio = contrastedY / max(preToneY, 0.022);
    float chromaScale = min(lumaRatio, 1.0 + (lumaRatio - 1.0) * 0.58) * u.saturationBoost;
    float3 graded = clamp(float3(contrastedY) + preToneChroma * chromaScale, 0.0, 1.0);

    if (u.ispEnabled != 0) {
        // Map neighborhood blurs into the same tone space for edge-accurate bilateral & unsharp mask
        float3 gradedBlur1 = clamp(blur1 * lumaRatio, 0.0, 1.0);
        float3 gradedBlur2 = clamp(blur2 * lumaRatio, 0.0, 1.0);

        // 2. Fine Edge-Preserving Bilateral Luma Micro-Denoise (tight edge threshold preserves all texture)
        float3 multiBlur = mix(gradedBlur1, gradedBlur2, 0.32);
        float3 spatialDiff = graded - multiBlur;
        float edgeSigma = mix(340.0, 560.0, smoothstep(0.05, 0.30, contrastedY));
        float spatialW = exp(-dot(spatialDiff, spatialDiff) * edgeSigma);
        float3 denoised = mix(graded, multiBlur, spatialW * clamp(u.denoiseStrength * 0.36, 0.0, 0.32));

        // 3. Luma-Adaptive Unsharp Mask (restores crisp optical micro-contrast sharper than scrcpy)
        float3 detailBase = mix(gradedBlur1, gradedBlur2, 0.22);
        float3 highFreq = denoised - detailBase;
        float noiseCore = mix(0.010, 0.003, smoothstep(0.06, 0.28, contrastedY));
        float3 coredHF = sign(highFreq) * max(abs(highFreq) - noiseCore, 0.0);
        float shadowSharpenDamp = smoothstep(0.03, 0.14, contrastedY);
        float3 sharpened = denoised + coredHF * (u.sharpenStrength * 1.65 * (0.45 + 0.55 * shadowSharpenDamp));
        float3 postSpatial = clamp(sharpened, 0.0, 1.0);

        // 4. Recursive IIR Motion-Adaptive Temporal 3DNR (texPrev is the previous ISP-processed output frame in screen coords)
        float3 prevSample = texPrev.sample(s, in.position.xy * u.texelSize).rgb;
        float3 motionDiff = abs(postSpatial - prevSample);
        float motionMagSq = dot(motionDiff, motionDiff);
        float temporalAlpha = exp(-motionMagSq * 240.0) * clamp(u.denoiseStrength * 0.52, 0.0, 0.48);
        graded = mix(postSpatial, prevSample, temporalAlpha);
    }

    if (abs(u.warmthShift) > 0.001) {
        graded.r = clamp(graded.r + u.warmthShift * 0.14, 0.0, 1.0);
        graded.g = clamp(graded.g + u.warmthShift * 0.02, 0.0, 1.0);
        graded.b = clamp(graded.b - u.warmthShift * 0.16, 0.0, 1.0);
    }

    if (dimW < 0.999) {
        graded = clamp(graded * dimW, 0.0, 1.0);
    }

    return float4(graded, 1.0);
}
"""

struct ISPUniforms {
    var texelSize: SIMD2<Float>
    var ispEnabled: Int32
    var mirrorHorizontal: Int32
    var flipVertical: Int32
    var rotationQuadrants: Int32
    var brightnessGain: Float
    var flashDimFactor: Float
    var shadowLift: Float
    var contrastBoost: Float
    var saturationBoost: Float
    var sharpenStrength: Float
    var denoiseStrength: Float
    var warmthShift: Float
}

// MARK: - Pxl Cam Scene Analysis Metrics
struct FrameSceneStats {
    let rawMeanLuma: Float        // 0.0 .. 1.0 center-weighted mean luminance
    let avgBrightness255: Float   // 0.0 .. 255.0 mean luminance (pxl-cam scale)
    let exposureBias: Float       // -0.5 .. +0.5 HSV Value deviation from 0.5
    let contrastStd255: Float     // 0.0 .. 128.0 std(gray) on 0..255 scale (<42 = flat)
    let laplacianSharpness: Float // Variance of 4-neighbor Laplacian operator (<80 = soft)
    let gaussianNoiseStd: Float   // High-freq residual std(gray - blur3x3) on 0..255 scale (>6 = noisy)
    let hsvSaturation255: Float   // 0.0 .. 255.0 mean HSV saturation (<68 = desaturated)
    let p10Luma: Float            // 10th percentile shadow luminance (0..1)
    let p50Luma: Float            // 50th percentile median luminance (0..1)
    let p90Luma: Float            // 90th percentile highlight luminance (0..1)
    let darkPixelFraction: Float  // Fraction of pixels < 0.16 luma
    let brightClipFraction: Float // Fraction of pixels > 0.94 luma
    let redGreenRatio: Float      // Warm/cool color balance R/G
    let blueGreenRatio: Float     // Warm/cool color balance B/G
    let sceneComplexity: Float    // 0.0 .. 1.5 composite complexity score
    let faceCount: Int            // Detected faces via Apple Vision
    let faceCoverage: Float       // Fraction of frame area occupied by faces (0..1)
}

// MARK: - Metal Video Renderer (Pure OBS Capture Target + Offscreen Minimized-Safe Studio Pipeline)
final class MetalVideoView: MTKView, MTKViewDelegate {
    private var commandQueue: MTLCommandQueue!
    private var pipelineState: MTLRenderPipelineState!
    private var passthroughPipelineState: MTLRenderPipelineState!
    private var textureCache: CVMetalTextureCache!
    private let textureLock = NSLock()
    private var currentPixelBuffer: CVPixelBuffer?
    private var previousPixelBuffer: CVPixelBuffer?
    private var latestProcessedPixelBuffer: CVPixelBuffer?
    private var previousProcessedPixelBuffer: CVPixelBuffer?
    private var latestJPEGData: Data?
    private var cachedStreamJPEG: Data?
    private var cachedStreamJPEGFrameID: UInt64 = 0
    private(set) var processedFrameID: UInt64 = 0
    private var pixelBufferPool: CVPixelBufferPool?
    private var poolWidth: Int = 0
    private var poolHeight: Int = 0
    private var ispOutputPool: CVPixelBufferPool?
    private var ispPoolWidth: Int = 0
    private var ispPoolHeight: Int = 0

    // Real-Time MP4 Video Recording State (AVAssetWriter)
    private let recordLock = NSLock()
    private var assetWriter: AVAssetWriter?
    private var assetWriterInput: AVAssetWriterInput?
    private var pixelBufferAdaptor: AVAssetWriterInputPixelBufferAdaptor?
    private(set) var isRecording: Bool = false
    private(set) var recordingFileURL: URL?
    private var recordingStartTime: CFAbsoluteTime = 0
    private var recordingFirstPTS: CFAbsoluteTime?
    private var recordingLastPTSUs: Int64 = -1
    private var recordingWidth: Int = 1920
    private var recordingHeight: Int = 1080
    private(set) var recordedFrameCount: Int = 0

    // Hardware VideoToolbox H.264 Decompression State (for scrcpy-style MediaCodec Surface stream)
    private let decoderLock = NSLock()
    private var vtSession: VTDecompressionSession?
    private var vtFormatDesc: CMVideoFormatDescription?
    private var currentSPS: [UInt8] = []
    private var currentPPS: [UInt8] = []
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // Live Shader Parameters (calibrated for native Android sRGB + ADRC tone curve)
    var ispEnabled: Bool = true
    var mirrorHorizontal: Bool = false
    var flipVertical: Bool = false
    var rotationQuadrants: Int = 0
    var brightnessGain: Float = 1.12
    var flashDimFactor: Float = 1.0
    var shadowLift: Float = 0.16
    var contrastBoost: Float = 1.05
    var saturationBoost: Float = 1.04
    var sharpenStrength: Float = 0.52
    var denoiseStrength: Float = 0.38
    var warmthShift: Float = 0.0

    var frameWidth: Int = 1920
    var frameHeight: Int = 1080

    override var acceptsFirstResponder: Bool { true }
    var onKeyPress: ((NSEvent) -> Bool)?

    init(frame frameRect: NSRect) {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is required")
        }
        super.init(frame: frameRect, device: dev)
        self.delegate = self
        self.framebufferOnly = true
        self.colorPixelFormat = .bgra8Unorm
        self.clearColor = MTLClearColor(red: 0.03, green: 0.03, blue: 0.04, alpha: 1.0)
        self.preferredFramesPerSecond = 60
        self.isPaused = false
        self.enableSetNeedsDisplay = false

        setupMetal(device: dev)
    }

    required init(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    override func keyDown(with event: NSEvent) {
        if let handler = onKeyPress, handler(event) {
            return
        }
        super.keyDown(with: event)
    }

    private func setupMetal(device: MTLDevice) {
        commandQueue = device.makeCommandQueue()
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &textureCache)

        do {
            let library = try device.makeLibrary(source: metalShaderSource, options: nil)
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = library.makeFunction(name: "vertexPassthrough")
            desc.fragmentFunction = library.makeFunction(name: "studioISPFragment")
            desc.colorAttachments[0].pixelFormat = self.colorPixelFormat
            pipelineState = try device.makeRenderPipelineState(descriptor: desc)

            let passDesc = MTLRenderPipelineDescriptor()
            passDesc.vertexFunction = library.makeFunction(name: "vertexPassthrough")
            passDesc.fragmentFunction = library.makeFunction(name: "passthroughFragment")
            passDesc.colorAttachments[0].pixelFormat = self.colorPixelFormat
            passthroughPipelineState = try device.makeRenderPipelineState(descriptor: passDesc)
        } catch {
            fatalError("Failed to compile Metal Studio ISP shader: \(error)")
        }
    }

    private func ensurePool(width: Int, height: Int) -> CVPixelBufferPool? {
        if let pool = pixelBufferPool, poolWidth == width, poolHeight == height {
            return pool
        }
        let poolAttrs: [CFString: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey: 6
        ]
        let bufferAttrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        var newPool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &newPool)
        pixelBufferPool = newPool
        poolWidth = width
        poolHeight = height
        return newPool
    }

    private func ensureISPOutputPool(width: Int, height: Int) -> CVPixelBufferPool? {
        if let pool = ispOutputPool, ispPoolWidth == width, ispPoolHeight == height {
            return pool
        }
        let poolAttrs: [CFString: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey: 6
        ]
        let bufferAttrs: [CFString: Any] = [
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
        ]
        var newPool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(kCFAllocatorDefault, poolAttrs as CFDictionary, bufferAttrs as CFDictionary, &newPool)
        ispOutputPool = newPool
        ispPoolWidth = width
        ispPoolHeight = height
        return newPool
    }

    /// Renders the current camera frame through the full Metal Studio ISP shader offscreen
    /// on every incoming frame (whether the window is visible or minimized in the macOS Dock),
    /// maintaining recursive IIR temporal 3DNR history (`previousProcessedPixelBuffer`).
    @discardableResult
    func renderStudioProcessedPixelBuffer(targetWidth: Int? = nil, targetHeight: Int? = nil, updateHistory: Bool = false) -> CVPixelBuffer? {
        textureLock.lock()
        guard let pb = currentPixelBuffer else {
            textureLock.unlock()
            return nil
        }
        let srcW = CVPixelBufferGetWidth(pb)
        let srcH = CVPixelBufferGetHeight(pb)
        let rotQ = ((rotationQuadrants % 4) + 4) % 4
        let isPortrait = (rotQ % 2) == 1
        let outW = targetWidth ?? (isPortrait ? srcH : srcW)
        let outH = targetHeight ?? (isPortrait ? srcW : srcH)
        let prevProcessed = latestProcessedPixelBuffer
        guard let outPool = ensureISPOutputPool(width: outW, height: outH) else {
            textureLock.unlock()
            return nil
        }
        var outPB: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, outPool, &outPB)
        textureLock.unlock()

        guard status == kCVReturnSuccess, let dstPixelBuffer = outPB else { return nil }

        var srcCvTex: CVMetalTexture?
        let resSrc = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            pb,
            nil,
            .bgra8Unorm,
            srcW,
            srcH,
            0,
            &srcCvTex
        )
        var prevCvTex: CVMetalTexture?
        if let prevPB = prevProcessed,
           CVPixelBufferGetWidth(prevPB) == outW,
           CVPixelBufferGetHeight(prevPB) == outH {
            _ = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                textureCache,
                prevPB,
                nil,
                .bgra8Unorm,
                outW,
                outH,
                0,
                &prevCvTex
            )
        }
        var dstCvTex: CVMetalTexture?
        let resDst = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            textureCache,
            dstPixelBuffer,
            nil,
            .bgra8Unorm,
            outW,
            outH,
            0,
            &dstCvTex
        )

        guard resSrc == kCVReturnSuccess,
              resDst == kCVReturnSuccess,
              let sCv = srcCvTex,
              let dCv = dstCvTex,
              let srcTex = CVMetalTextureGetTexture(sCv),
              let dstTex = CVMetalTextureGetTexture(dCv) else {
            return nil
        }
        let prevTex = (prevCvTex != nil ? CVMetalTextureGetTexture(prevCvTex!) : nil) ?? srcTex

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = dstTex
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeRenderCommandEncoder(descriptor: rpd) else {
            return nil
        }

        var uniforms = ISPUniforms(
            texelSize: SIMD2<Float>(1.0 / Float(max(outW, 1)), 1.0 / Float(max(outH, 1))),
            ispEnabled: ispEnabled ? 1 : 0,
            mirrorHorizontal: mirrorHorizontal ? 1 : 0,
            flipVertical: flipVertical ? 1 : 0,
            rotationQuadrants: Int32(rotQ),
            brightnessGain: brightnessGain,
            flashDimFactor: flashDimFactor,
            shadowLift: shadowLift,
            contrastBoost: contrastBoost,
            saturationBoost: saturationBoost,
            sharpenStrength: sharpenStrength,
            denoiseStrength: denoiseStrength,
            warmthShift: warmthShift
        )
        encoder.setRenderPipelineState(pipelineState)
        encoder.setVertexBytes(&uniforms, length: MemoryLayout<ISPUniforms>.stride, index: 0)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<ISPUniforms>.stride, index: 0)
        encoder.setFragmentTexture(srcTex, index: 0)
        encoder.setFragmentTexture(prevTex, index: 1)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if updateHistory {
            textureLock.lock()
            self.previousProcessedPixelBuffer = self.latestProcessedPixelBuffer
            self.latestProcessedPixelBuffer = dstPixelBuffer
            self.processedFrameID &+= 1
            textureLock.unlock()
        }

        return dstPixelBuffer
    }

    /// Returns fast Metal-accelerated JPEG data for the OBS 1080p Direct Stream (`/stream.mjpg` & `/frame.jpg`),
    /// caching per frame ID so multiple clients (OBS Browser Source + Dock preview) share a single encode.
    func latestStreamJPEGData(quality: Float = 0.92) -> Data? {
        textureLock.lock()
        let fid = processedFrameID
        if let cached = cachedStreamJPEG, cachedStreamJPEGFrameID == fid {
            textureLock.unlock()
            return cached
        }
        let pb = latestProcessedPixelBuffer ?? currentPixelBuffer
        textureLock.unlock()

        guard let pixelBuffer = pb else { return nil }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        let opts: [CIImageRepresentationOption: Any] = [
            kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality
        ]
        guard let jpegData = ciContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: opts) else {
            return nil
        }
        textureLock.lock()
        cachedStreamJPEG = jpegData
        cachedStreamJPEGFrameID = fid
        textureLock.unlock()
        return jpegData
    }

    func latestSnapshotJPEG() -> (data: Data, width: Int, height: Int)? {
        if let processedPB = renderStudioProcessedPixelBuffer(updateHistory: false) {
            let w = CVPixelBufferGetWidth(processedPB)
            let h = CVPixelBufferGetHeight(processedPB)
            let ciImage = CIImage(cvPixelBuffer: processedPB)
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
            let opts: [CIImageRepresentationOption: Any] = [
                kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.96
            ]
            if let jpegData = ciContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: opts) {
                return (jpegData, w, h)
            }
        }
        textureLock.lock()
        let pb = currentPixelBuffer
        textureLock.unlock()
        guard let pixelBuffer = pb else { return nil }
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        guard let data = ciContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: [:]) else {
            return nil
        }
        return (data, w, h)
    }

    var recordingDurationSeconds: Double {
        recordLock.lock()
        defer { recordLock.unlock() }
        guard isRecording else { return 0.0 }
        return max(0.0, CFAbsoluteTimeGetCurrent() - recordingStartTime)
    }

    func startRecording(to outputURL: URL) -> (started: Bool, width: Int, height: Int) {
        textureLock.lock()
        let srcW = max(2, frameWidth)
        let srcH = max(2, frameHeight)
        let rotQ = ((rotationQuadrants % 4) + 4) % 4
        let isPortrait = (rotQ % 2) == 1
        let recW = ((isPortrait ? srcH : srcW) / 2) * 2
        let recH = ((isPortrait ? srcW : srcH) / 2) * 2
        textureLock.unlock()

        recordLock.lock()
        defer { recordLock.unlock() }
        guard !isRecording else { return (false, recW, recH) }

        do {
            let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
            let compressionProps: [String: Any] = [
                AVVideoAverageBitRateKey: 16_000_000,
                AVVideoExpectedSourceFrameRateKey: 30,
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel
            ]
            let videoSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: recW,
                AVVideoHeightKey: recH,
                AVVideoCompressionPropertiesKey: compressionProps
            ]
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            input.expectsMediaDataInRealTime = true

            let sourceBufferAttrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
                kCVPixelBufferWidthKey as String: recW,
                kCVPixelBufferHeightKey as String: recH,
                kCVPixelBufferMetalCompatibilityKey as String: true
            ]
            let adaptor = AVAssetWriterInputPixelBufferAdaptor(
                assetWriterInput: input,
                sourcePixelBufferAttributes: sourceBufferAttrs
            )

            guard writer.canAdd(input) else {
                return (false, recW, recH)
            }
            writer.add(input)
            guard writer.startWriting() else {
                return (false, recW, recH)
            }
            writer.startSession(atSourceTime: .zero)

            self.assetWriter = writer
            self.assetWriterInput = input
            self.pixelBufferAdaptor = adaptor
            self.recordingFileURL = outputURL
            self.recordingStartTime = CFAbsoluteTimeGetCurrent()
            self.recordingFirstPTS = nil
            self.recordingLastPTSUs = -1
            self.recordingWidth = recW
            self.recordingHeight = recH
            self.recordedFrameCount = 0
            self.isRecording = true
            return (true, recW, recH)
        } catch {
            return (false, recW, recH)
        }
    }

    func stopRecording(completion: @escaping (URL?, Double, Int) -> Void) {
        recordLock.lock()
        guard isRecording, let writer = assetWriter, let input = assetWriterInput else {
            recordLock.unlock()
            completion(nil, 0, 0)
            return
        }
        let url = recordingFileURL
        let duration = max(0.0, CFAbsoluteTimeGetCurrent() - recordingStartTime)
        let frames = recordedFrameCount
        isRecording = false
        assetWriter = nil
        assetWriterInput = nil
        pixelBufferAdaptor = nil
        recordingFileURL = nil
        recordingFirstPTS = nil
        recordingLastPTSUs = -1
        recordLock.unlock()

        input.markAsFinished()
        writer.finishWriting {
            DispatchQueue.main.async {
                if writer.status == .completed {
                    completion(url, duration, frames)
                } else {
                    completion(nil, duration, frames)
                }
            }
        }
    }

    private func appendCurrentFrameToRecordingIfNeeded() {
        recordLock.lock()
        let active = isRecording
        let recW = recordingWidth
        let recH = recordingHeight
        recordLock.unlock()
        guard active else { return }

        textureLock.lock()
        let existingProcessed = latestProcessedPixelBuffer
        textureLock.unlock()

        let processedPB: CVPixelBuffer
        if let existing = existingProcessed,
           CVPixelBufferGetWidth(existing) == recW,
           CVPixelBufferGetHeight(existing) == recH {
            processedPB = existing
        } else if let rendered = renderStudioProcessedPixelBuffer(targetWidth: recW, targetHeight: recH, updateHistory: false) {
            processedPB = rendered
        } else {
            return
        }

        recordLock.lock()
        guard isRecording,
              let writer = assetWriter,
              writer.status == .writing,
              let input = assetWriterInput,
              input.isReadyForMoreMediaData,
              let adaptor = pixelBufferAdaptor else {
            recordLock.unlock()
            return
        }

        let now = CFAbsoluteTimeGetCurrent()
        let firstTime: CFAbsoluteTime
        if let existingFirst = recordingFirstPTS {
            firstTime = existingFirst
        } else {
            firstTime = now
            recordingFirstPTS = now
        }
        var ptsUs = Int64((now - firstTime) * 1_000_000.0)
        if ptsUs <= recordingLastPTSUs {
            ptsUs = recordingLastPTSUs + 1_000 // guarantee strictly increasing PTS
        }
        recordingLastPTSUs = ptsUs
        let presentationTime = CMTime(value: ptsUs, timescale: 1_000_000)
        if adaptor.append(processedPB, withPresentationTime: presentationTime) {
            recordedFrameCount += 1
        }
        recordLock.unlock()
    }

    /// Runs the pxl-cam multi-signal frame analysis engine on the raw decoded camera frame:
    /// - Luminance mean, std-dev contrast, and shadow/median/highlight percentiles (p10/p50/p90)
    /// - HSV Saturation mean and Exposure Bias (mean(V) - 0.5)
    /// - 4-neighbor Laplacian variance for sharpness estimation
    /// - High-frequency Gaussian residual std-dev in non-edge regions for ISO noise estimation
    /// - Apple Vision VNDetectFaceRectanglesRequest for portrait/skin-tone protection
    func analyzeCurrentFrame(detectFaces: Bool = true) -> FrameSceneStats? {
        textureLock.lock()
        guard let pixelBuffer = currentPixelBuffer else {
            textureLock.unlock()
            return nil
        }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)

        guard let baseAddr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            textureLock.unlock()
            return nil
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard width > 32, height > 32 else {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            textureLock.unlock()
            return nil
        }

        let ptr = baseAddr.assumingMemoryBound(to: UInt8.self)
        let gridX = 72
        let gridY = 40
        var lumaSamples: [Float] = []
        lumaSamples.reserveCapacity(gridX * gridY)

        var weightedLumaSum: Float = 0.0
        var weightSum: Float = 0.0
        var unweightedLumaSum: Float = 0.0
        var darkCount: Int = 0
        var clipCount: Int = 0
        var totalCount: Int = 0

        var sumV: Float = 0.0
        var sumSat: Float = 0.0
        var sumR: Float = 0.0
        var sumG: Float = 0.0
        var sumB: Float = 0.0

        var laplacianSum: Float = 0.0
        var laplacianSqSum: Float = 0.0
        var residualSqSum: Float = 0.0
        var residualCount: Int = 0

        @inline(__always)
        func lumaAt(_ x: Int, _ y: Int) -> Float {
            let row = ptr.advanced(by: y * bytesPerRow)
            let p = row.advanced(by: x * 4)
            return 0.114 * Float(p[0]) + 0.587 * Float(p[1]) + 0.299 * Float(p[2]) // 0..255 scale
        }

        for gy in 0..<gridY {
            let y = 2 + (gy * (height - 5)) / max(gridY - 1, 1)
            let rowPtr = ptr.advanced(by: y * bytesPerRow)
            let ny = Float(gy) / Float(gridY - 1)
            let centerY = (ny >= 0.18 && ny <= 0.82)

            for gx in 0..<gridX {
                let x = 2 + (gx * (width - 5)) / max(gridX - 1, 1)
                let px = rowPtr.advanced(by: x * 4)
                // BGRA8 layout
                let b255 = Float(px[0])
                let g255 = Float(px[1])
                let r255 = Float(px[2])

                let b = b255 / 255.0
                let g = g255 / 255.0
                let r = r255 / 255.0
                let luma01 = 0.2126 * r + 0.7152 * g + 0.0722 * b
                let luma255 = 0.299 * r255 + 0.587 * g255 + 0.114 * b255

                let maxC = max(r, max(g, b))
                let minC = min(r, min(g, b))
                let sat = (maxC > 0.001) ? ((maxC - minC) / maxC) : 0.0

                let nx = Float(gx) / Float(gridX - 1)
                let isCenter = centerY && (nx >= 0.18 && nx <= 0.82)
                let weight: Float = isCenter ? 1.65 : 0.82

                weightedLumaSum += luma01 * weight
                weightSum += weight
                unweightedLumaSum += luma255
                lumaSamples.append(luma01)
                totalCount += 1

                if luma01 < 0.16 { darkCount += 1 }
                if luma01 > 0.94 { clipCount += 1 }

                sumV += maxC
                sumSat += sat * 255.0
                sumR += r
                sumG += g
                sumB += b

                // 4-neighbor Laplacian & 3x3 residual noise (pxl-cam smart_config metrics)
                let lTop = lumaAt(x, y - 1)
                let lBot = lumaAt(x, y + 1)
                let lLeft = lumaAt(x - 1, y)
                let lRight = lumaAt(x + 1, y)

                let lap = 4.0 * luma255 - (lTop + lBot + lLeft + lRight)
                laplacianSum += lap
                laplacianSqSum += lap * lap

                // High-frequency residual against local 5-tap Gaussian approximation
                let localBlur = (4.0 * luma255 + lTop + lBot + lLeft + lRight) / 8.0
                let residual = luma255 - localBlur
                if abs(lap) < 38.0 { // Flat/non-edge region -> measures sensor ISO grain
                    residualSqSum += residual * residual
                    residualCount += 1
                }
            }
        }

        CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        textureLock.unlock()

        guard totalCount > 0 else { return nil }
        let nFloat = Float(totalCount)
        let avgBright255 = unweightedLumaSum / nFloat
        let meanLuma01 = weightedLumaSum / max(weightSum, 1.0)
        let expBias = (sumV / nFloat) - 0.5
        let avgSat255 = sumSat / nFloat

        // Contrast std-dev on 0..255 scale
        var varAccum: Float = 0.0
        for s in lumaSamples {
            let d = (s * 255.0) - avgBright255
            varAccum += d * d
        }
        let contrastStd = sqrt(varAccum / nFloat)

        // Laplacian variance (sharpness metric in pxl-cam)
        let lapMean = laplacianSum / nFloat
        let lapVar = max(0.0, (laplacianSqSum / nFloat) - (lapMean * lapMean))

        // Gaussian residual std-dev (noise metric in pxl-cam)
        let noiseStd = (residualCount > 0) ? sqrt(residualSqSum / Float(residualCount)) : 1.5

        lumaSamples.sort()
        let p10 = lumaSamples[min(max(Int(nFloat * 0.10), 0), totalCount - 1)]
        let p50 = lumaSamples[min(max(Int(nFloat * 0.50), 0), totalCount - 1)]
        let p90 = lumaSamples[min(max(Int(nFloat * 0.90), 0), totalCount - 1)]
        let darkFrac = Float(darkCount) / nFloat
        let clipFrac = Float(clipCount) / nFloat
        let safeG = max(sumG, 0.001)

        let complexity = min(max(0.6 * (lapVar / 150.0) + 0.4 * (contrastStd / 60.0), 0.0), 1.5)

        // Optional Apple Vision face detection for portrait-aware exposure & skin protection
        var detectedFaces = 0
        var faceAreaRatio: Float = 0.0
        if detectFaces {
            let req = VNDetectFaceRectanglesRequest()
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            try? handler.perform([req])
            if let results = req.results {
                detectedFaces = results.count
                for face in results {
                    let bbox = face.boundingBox
                    faceAreaRatio += Float(bbox.width * bbox.height)
                }
                faceAreaRatio = min(faceAreaRatio, 1.0)
            }
        }

        return FrameSceneStats(
            rawMeanLuma: meanLuma01,
            avgBrightness255: avgBright255,
            exposureBias: expBias,
            contrastStd255: contrastStd,
            laplacianSharpness: lapVar,
            gaussianNoiseStd: noiseStd,
            hsvSaturation255: avgSat255,
            p10Luma: p10,
            p50Luma: p50,
            p90Luma: p90,
            darkPixelFraction: darkFrac,
            brightClipFraction: clipFrac,
            redGreenRatio: sumR / safeG,
            blueGreenRatio: sumB / safeG,
            sceneComplexity: complexity,
            faceCount: detectedFaces,
            faceCoverage: faceAreaRatio
        )
    }

    func updateWithJPEGData(_ data: Data, width: Int, height: Int) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            return
        }
        let w = cgImage.width
        let h = cgImage.height

        textureLock.lock()
        self.latestJPEGData = data
        guard let pool = ensurePool(width: w, height: h) else {
            textureLock.unlock()
            return
        }
        var pb: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pb)
        textureLock.unlock()

        guard status == kCVReturnSuccess, let pixelBuffer = pb else { return }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        if let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) {
            let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
            if let ctx = CGContext(
                data: baseAddress,
                width: w,
                height: h,
                bitsPerComponent: 8,
                bytesPerRow: bytesPerRow,
                space: colorSpace,
                bitmapInfo: bitmapInfo
            ) {
                ctx.interpolationQuality = .none
                ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
        }
        CVPixelBufferUnlockBaseAddress(pixelBuffer, [])

        textureLock.lock()
        if let prev = self.currentPixelBuffer,
           CVPixelBufferGetWidth(prev) == w,
           CVPixelBufferGetHeight(prev) == h {
            self.previousPixelBuffer = prev
        } else {
            self.previousPixelBuffer = pixelBuffer
        }
        self.currentPixelBuffer = pixelBuffer
        self.frameWidth = w
        self.frameHeight = h
        textureLock.unlock()
        _ = renderStudioProcessedPixelBuffer(updateHistory: true)
        appendCurrentFrameToRecordingIfNeeded()
    }

    private func extractAnnexBNALRanges(from bytes: UnsafeBufferPointer<UInt8>) -> [Range<Int>] {
        let count = bytes.count
        guard count >= 4 else { return [] }
        var nalStarts: [Int] = []
        var startCodePositions: [Int] = []
        var i = 0
        while i < count - 3 {
            if bytes[i] == 0 && bytes[i + 1] == 0 {
                if bytes[i + 2] == 1 {
                    startCodePositions.append(i)
                    nalStarts.append(i + 3)
                    i += 3
                    continue
                } else if i + 3 < count && bytes[i + 2] == 0 && bytes[i + 3] == 1 {
                    startCodePositions.append(i)
                    nalStarts.append(i + 4)
                    i += 4
                    continue
                }
            }
            i += 1
        }
        var ranges: [Range<Int>] = []
        for idx in 0..<nalStarts.count {
            let start = nalStarts[idx]
            let end = (idx + 1 < startCodePositions.count) ? startCodePositions[idx + 1] : count
            if start < end {
                ranges.append(start..<end)
            }
        }
        return ranges
    }

    func updateWithH264Data(_ data: Data, width: Int, height: Int) {
        decoderLock.lock()
        defer { decoderLock.unlock() }

        var spsCandidate: [UInt8]?
        var ppsCandidate: [UInt8]?
        var avccData = Data()

        data.withUnsafeBytes { rawBuf in
            guard let base = rawBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return }
            let bufPtr = UnsafeBufferPointer(start: base, count: data.count)
            let nalRanges = extractAnnexBNALRanges(from: bufPtr)
            for r in nalRanges {
                let nalHeader = bufPtr[r.lowerBound]
                let nalType = nalHeader & 0x1F
                if nalType == 7 {
                    spsCandidate = Array(bufPtr[r])
                } else if nalType == 8 {
                    ppsCandidate = Array(bufPtr[r])
                } else if nalType == 1 || nalType == 5 || nalType == 6 {
                    var lenBE = UInt32(r.count).bigEndian
                    withUnsafeBytes(of: &lenBE) { avccData.append(contentsOf: $0) }
                    avccData.append(base.advanced(by: r.lowerBound), count: r.count)
                }
            }
        }

        if let sps = spsCandidate, let pps = ppsCandidate {
            if vtSession == nil || vtFormatDesc == nil || sps != currentSPS || pps != currentPPS {
                if let oldSession = vtSession {
                    VTDecompressionSessionInvalidate(oldSession)
                    vtSession = nil
                }
                vtFormatDesc = nil

                var formatDescOut: CMFormatDescription?
                let paramStatus: OSStatus = sps.withUnsafeBufferPointer { spsPtr in
                    pps.withUnsafeBufferPointer { ppsPtr in
                        guard let spsBase = spsPtr.baseAddress, let ppsBase = ppsPtr.baseAddress else {
                            return -1
                        }
                        let pointers: [UnsafePointer<UInt8>] = [spsBase, ppsBase]
                        let sizes: [Int] = [sps.count, pps.count]
                        return pointers.withUnsafeBufferPointer { ptrArray in
                            sizes.withUnsafeBufferPointer { sizeArray in
                                CMVideoFormatDescriptionCreateFromH264ParameterSets(
                                    allocator: kCFAllocatorDefault,
                                    parameterSetCount: 2,
                                    parameterSetPointers: ptrArray.baseAddress!,
                                    parameterSetSizes: sizeArray.baseAddress!,
                                    nalUnitHeaderLength: 4,
                                    formatDescriptionOut: &formatDescOut
                                )
                            }
                        }
                    }
                }

                if paramStatus == noErr, let fmtDesc = formatDescOut {
                    let destAttrs: [CFString: Any] = [
                        kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
                        kCVPixelBufferMetalCompatibilityKey: true,
                        kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary
                    ]
                    var newSession: VTDecompressionSession?
                    let sessStatus = VTDecompressionSessionCreate(
                        allocator: kCFAllocatorDefault,
                        formatDescription: fmtDesc,
                        decoderSpecification: nil,
                        imageBufferAttributes: destAttrs as CFDictionary,
                        outputCallback: nil,
                        decompressionSessionOut: &newSession
                    )
                    if sessStatus == noErr, let s = newSession {
                        vtFormatDesc = fmtDesc
                        vtSession = s
                        currentSPS = sps
                        currentPPS = pps
                    }
                }
            }
        }

        guard !avccData.isEmpty, let session = vtSession, let formatDesc = vtFormatDesc else {
            return
        }

        var blockBuffer: CMBlockBuffer?
        let bbStatus = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: avccData.count,
            blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: avccData.count,
            flags: kCMBlockBufferAssureMemoryNowFlag,
            blockBufferOut: &blockBuffer
        )
        guard bbStatus == kCMBlockBufferNoErr, let blockBuf = blockBuffer else { return }

        let repStatus = avccData.withUnsafeBytes { rawPtr -> OSStatus in
            guard let base = rawPtr.baseAddress else { return -1 }
            return CMBlockBufferReplaceDataBytes(
                with: base,
                blockBuffer: blockBuf,
                offsetIntoDestination: 0,
                dataLength: avccData.count
            )
        }
        guard repStatus == kCMBlockBufferNoErr else { return }

        var sampleBuffer: CMSampleBuffer?
        var sampleSize = avccData.count
        let sbStatus = CMSampleBufferCreateReady(
            allocator: kCFAllocatorDefault,
            dataBuffer: blockBuf,
            formatDescription: formatDesc,
            sampleCount: 1,
            sampleTimingEntryCount: 0,
            sampleTimingArray: nil,
            sampleSizeEntryCount: 1,
            sampleSizeArray: &sampleSize,
            sampleBufferOut: &sampleBuffer
        )
        guard sbStatus == noErr, let sampleBuf = sampleBuffer else { return }

        var infoFlags = VTDecodeInfoFlags()
        VTDecompressionSessionDecodeFrame(
            session,
            sampleBuffer: sampleBuf,
            flags: [],
            infoFlagsOut: &infoFlags
        ) { [weak self] status, _, imageBuffer, _, _ in
            guard status == noErr, let self = self, let pixelBuffer = imageBuffer else { return }
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)
            self.textureLock.lock()
            if let prev = self.currentPixelBuffer,
               CVPixelBufferGetWidth(prev) == w,
               CVPixelBufferGetHeight(prev) == h {
                self.previousPixelBuffer = prev
            } else {
                self.previousPixelBuffer = pixelBuffer
            }
            self.currentPixelBuffer = pixelBuffer
            self.frameWidth = w
            self.frameHeight = h
            self.latestJPEGData = nil
            self.textureLock.unlock()
            _ = self.renderStudioProcessedPixelBuffer(updateHistory: true)
            self.appendCurrentFrameToRecordingIfNeeded()
        }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        textureLock.lock()
        let processedPB = latestProcessedPixelBuffer
        let rawPB = currentPixelBuffer
        let prevPB = previousProcessedPixelBuffer ?? previousPixelBuffer ?? currentPixelBuffer
        let w = frameWidth
        let h = frameHeight
        textureLock.unlock()

        guard let drawable = currentDrawable,
              let rpd = currentRenderPassDescriptor,
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeRenderCommandEncoder(descriptor: rpd) else {
            return
        }

        if let readyPB = processedPB {
            let pw = CVPixelBufferGetWidth(readyPB)
            let ph = CVPixelBufferGetHeight(readyPB)
            var cvTex: CVMetalTexture?
            let res = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                textureCache,
                readyPB,
                nil,
                .bgra8Unorm,
                pw,
                ph,
                0,
                &cvTex
            )
            if res == kCVReturnSuccess, let cvTex = cvTex, let mtlTex = CVMetalTextureGetTexture(cvTex) {
                var passthroughUniforms = ISPUniforms(
                    texelSize: SIMD2<Float>(1.0 / Float(max(pw, 1)), 1.0 / Float(max(ph, 1))),
                    ispEnabled: 0,
                    mirrorHorizontal: 0,
                    flipVertical: 0,
                    rotationQuadrants: 0,
                    brightnessGain: 1.0,
                    flashDimFactor: 1.0,
                    shadowLift: 0.0,
                    contrastBoost: 1.0,
                    saturationBoost: 1.0,
                    sharpenStrength: 0.0,
                    denoiseStrength: 0.0,
                    warmthShift: 0.0
                )
                encoder.setRenderPipelineState(passthroughPipelineState)
                encoder.setVertexBytes(&passthroughUniforms, length: MemoryLayout<ISPUniforms>.stride, index: 0)
                encoder.setFragmentTexture(mtlTex, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
        } else if let pixelBuffer = rawPB {
            var cvTex: CVMetalTexture?
            let res = CVMetalTextureCacheCreateTextureFromImage(
                kCFAllocatorDefault,
                textureCache,
                pixelBuffer,
                nil,
                .bgra8Unorm,
                w,
                h,
                0,
                &cvTex
            )
            var prevCvTex: CVMetalTexture?
            if let prevPixelBuffer = prevPB {
                _ = CVMetalTextureCacheCreateTextureFromImage(
                    kCFAllocatorDefault,
                    textureCache,
                    prevPixelBuffer,
                    nil,
                    .bgra8Unorm,
                    CVPixelBufferGetWidth(prevPixelBuffer),
                    CVPixelBufferGetHeight(prevPixelBuffer),
                    0,
                    &prevCvTex
                )
            }
            if res == kCVReturnSuccess, let cvTex = cvTex, let mtlTex = CVMetalTextureGetTexture(cvTex) {
                let prevMtlTex = (prevCvTex != nil ? CVMetalTextureGetTexture(prevCvTex!) : nil) ?? mtlTex
                var uniforms = ISPUniforms(
                    texelSize: SIMD2<Float>(1.0 / Float(max(w, 1)), 1.0 / Float(max(h, 1))),
                    ispEnabled: ispEnabled ? 1 : 0,
                    mirrorHorizontal: mirrorHorizontal ? 1 : 0,
                    flipVertical: flipVertical ? 1 : 0,
                    rotationQuadrants: Int32(rotationQuadrants % 4),
                    brightnessGain: brightnessGain,
                    flashDimFactor: flashDimFactor,
                    shadowLift: shadowLift,
                    contrastBoost: contrastBoost,
                    saturationBoost: saturationBoost,
                    sharpenStrength: sharpenStrength,
                    denoiseStrength: denoiseStrength,
                    warmthShift: warmthShift
                )
                encoder.setRenderPipelineState(pipelineState)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<ISPUniforms>.stride, index: 0)
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<ISPUniforms>.stride, index: 0)
                encoder.setFragmentTexture(mtlTex, index: 0)
                encoder.setFragmentTexture(prevMtlTex, index: 1)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
        }

        encoder.endEncoding()
        cmdBuf.present(drawable)
        cmdBuf.commit()
    }
}


// MARK: - Child Overlay Pill Window (Invisible to OBS)
final class OverlayPillWindow: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

// MARK: - Main Application Controller
final class PxlCamAppDelegate: NSResponder, NSApplicationDelegate, NSWindowDelegate {
    private var videoWindow: NSWindow!
    private var overlayPillWindow: OverlayPillWindow!
    private var topLeftBrandWindow: OverlayPillWindow!
    private var settingsWindow: NSPanel!
    private var hoverTooltipWindow: NSPanel!
    private var hoverTooltipLabel: NSTextField!
    private var videoView: MetalVideoView!

    // Minimal Overlay Pill Controls
    private var pillStatusDot: NSTextField!
    private var pillStatusLabel: NSTextField!
    private var pillFlipBtn: NSButton!
    private var pillFlashBtn: NSButton!
    private var pillZoomSlider: NSSlider!
    private var pillZoomLabel: NSTextField!
    private var pillAFBtn: NSButton!
    private var pillSnapBtn: NSButton!
    private var pillRecordBtn: NSButton!
    private var pillWandBtn: NSButton!
    private var pillGearBtn: NSButton!

    // Separate Settings Inspector Controls
    private var aiWandHeroBtn: NSButton!
    private var aiTelemetryLabel: NSTextField!
    private var lensPopup: NSPopUpButton!
    private var resPopup: NSPopUpButton!
    private var fpsModePopup: NSPopUpButton!
    private var flashSegment: NSSegmentedControl!
    private var flashDimmerSlider: NSSlider!
    private var flashDimmerLabel: NSTextField!
    private var topZoomSlider: NSSlider!
    private var topZoomLabel: NSTextField!
    private var wbPopup: NSPopUpButton!
    private var afModePopup: NSPopUpButton!
    private var aeLockCheck: NSButton!
    private var awbLockCheck: NSButton!
    private var evSlider: NSSlider!
    private var evLabel: NSTextField!
    private var qualitySlider: NSSlider!
    private var qualityLabel: NSTextField!

    // GPU ISP Sliders in Settings Window
    private var ispCheck: NSButton!
    private var brightnessSlider: NSSlider!
    private var brightnessLabel: NSTextField!
    private var shadowSlider: NSSlider!
    private var shadowLabel: NSTextField!
    private var contrastSlider: NSSlider!
    private var contrastLabel: NSTextField!
    private var saturationSlider: NSSlider!
    private var saturationLabel: NSTextField!
    private var denoiseSlider: NSSlider!
    private var denoiseLabel: NSTextField!
    private var sharpnessSlider: NSSlider!
    private var sharpnessLabel: NSTextField!

    // Transform & Device Controls in Settings Window
    private var autoRotateCheck: NSButton!
    private var mirrorCheck: NSButton!
    private var flipVCheck: NSButton!
    private var screenOffCheck: NSButton!
    private var recordBtn: NSButton!

    // State
    private var currentLens: Int = 0       // 0 = Back, 1 = Front
    private var currentResIndex: Int = 2   // 0 = 4K (3840x2160), 1 = QHD+ (3264x1836 / 2560x1440), 2 = 1080p, 3 = 720p
    private var currentFlashMode: Int = 0  // 0 = Off, 1 = On (Torch), 2 = Auto
    private var currentFlashStrength: Int = 100 // 0..100% Torch / Flash Dimmer Level
    private var currentWBMode: Int = 0
    private var currentAFMode: Int = 0
    private var currentFPSMode: Int = 0    // 0 = Low-Light Auto (default), 1 = 30fps Lock, 2 = Night Boost
    private var currentZoom: Float = 1.0
    private var currentEV: Int = 0
    private var currentPreset: Int = 1     // 0=Natural, 1=Studio, 2=Broadcast, 3=Cinema, 4=Clean, 5=Warm, 6=Crisp
    private var isWandModeActive: Bool = false
    private var isAFActive: Bool = false
    private var snapActiveUntil: CFAbsoluteTime = 0.0
    private weak var hoveredPillButton: NSView?
    private var hudVisible: Bool = true
    private var controlsHideTimer: Timer?
    private var controlsEventMonitor: Any?
    private var topLeftBrandAspect: CGFloat = 1208.0 / 792.0
    private var wandBannerUntil: CFAbsoluteTime = 0.0
    private var wandBannerText: String = ""
    private var sensorRotationQuadrants: Int = 0
    private var manualRotationQuadrants: Int = 0
    private var autoRotateEnabled: Bool = true
    private var isWindowPortrait: Bool = false
    private var appNapActivityToken: NSObjectProtocol?
    private var obsHTTPServerSocket: Int32 = -1

    // Networking & Telemetry
    private let commandQueue = DispatchQueue(label: "com.pxlcam.app.cmd")
    private var activeClientSocket: Int32 = -1
    private var isConnected: Bool = false
    private var lastAutoServerLaunchAttempt: CFAbsoluteTime = 0.0
    private var frameCounter: Int = 0
    private var lastFPSCheck = CFAbsoluteTimeGetCurrent()
    private var measuredFPS: Double = 0.0
    private var lastFrameBytes: Int = 0

    func applicationDidFinishLaunching(_ notification: Notification) {
        UserDefaults.standard.register(defaults: ["NSInitialToolTipDelay": 90])
        NSApp.setActivationPolicy(.regular)
        NSApp.appearance = NSAppearance(named: .darkAqua)
        // Prevent macOS App Nap so Metal 1080p ISP & OBS streaming run at full 30fps even when minimized in the Dock!
        appNapActivityToken = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .latencyCritical],
            reason: "Pxl Cam Live 1080p Studio Stream & OBS Custom Dock"
        )
        applyCustomApplicationIcon()
        buildHoverTooltipWindow()
        buildVideoWindow()
        buildTopLeftBrandOverlay()
        buildMinimalOverlayPill()
        buildSettingsInspectorWindow()
        setupControlsAutoHideTimer()
        startStreamReceiverThread()
        startLocalOBSBridgeServer()
        startTelemetryTimer()
        NSApp.activate(ignoringOtherApps: true)
    }

    private func applyCustomApplicationIcon() {
        let fm = FileManager.default
        let candidatePaths: [String] = [
            Bundle.main.path(forResource: "AppIcon", ofType: "icns"),
            "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_rounded.png",
            "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_rounded_transparent.png",
            "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_noborder.png"
        ].compactMap { $0 }

        for path in candidatePaths where fm.fileExists(atPath: path) {
            if let img = NSImage(contentsOfFile: path) {
                NSApp.applicationIconImage = img
                break
            }
        }
    }

    // MARK: - 1. Pure Video Window (Captured by OBS)
    private func buildVideoWindow() {
        let initialRect = NSRect(x: 80, y: 140, width: 1152, height: 648)
        videoWindow = NSWindow(
            contentRect: initialRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        videoWindow.title = "MotoG_OBS_Camera"
        videoWindow.backgroundColor = .black
        videoWindow.contentAspectRatio = NSSize(width: 16, height: 9)
        videoWindow.collectionBehavior = [.fullScreenPrimary, .managed]
        videoWindow.delegate = self
        videoWindow.sharingType = .readOnly
        videoWindow.acceptsMouseMovedEvents = true

        videoView = MetalVideoView(frame: initialRect)
        videoView.autoresizingMask = [.width, .height]
        videoView.onKeyPress = { [weak self] event in
            self?.handleKeyCommand(event) ?? false
        }

        videoWindow.contentView = videoView
        videoWindow.makeKeyAndOrderFront(nil)
        videoWindow.makeFirstResponder(videoView)
    }

    // MARK: - 1B. Top-Left Coral Fox + 80's Glam BTTF PXL/CAM Camera Wordmark Overlay
    private func buildTopLeftBrandOverlay() {
        let candidates = [
            "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_top_left_overlay.png",
            Bundle.main.resourcePath.map { "\($0)/pxl_cam_top_left_overlay.png" } ?? ""
        ]
        var loadedImg: NSImage?
        for path in candidates where !path.isEmpty {
            if let img = NSImage(contentsOfFile: path) {
                loadedImg = img
                if img.size.height > 0 {
                    topLeftBrandAspect = img.size.width / img.size.height
                }
                break
            }
        }

        let brandHeight: CGFloat = 96
        let brandWidth: CGFloat = round(brandHeight * topLeftBrandAspect)
        let contentRect = videoWindow.contentRect(forFrameRect: videoWindow.frame)
        let brandRect = NSRect(
            x: contentRect.origin.x + 16,
            y: contentRect.maxY - brandHeight - 14,
            width: brandWidth,
            height: brandHeight
        )

        topLeftBrandWindow = OverlayPillWindow(
            contentRect: brandRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        topLeftBrandWindow.isOpaque = false
        topLeftBrandWindow.backgroundColor = .clear
        topLeftBrandWindow.hasShadow = true
        topLeftBrandWindow.level = .floating
        topLeftBrandWindow.ignoresMouseEvents = true
        topLeftBrandWindow.sharingType = .readOnly

        let container = NSView(frame: NSRect(x: 0, y: 0, width: brandWidth, height: brandHeight))
        container.wantsLayer = true
        container.layer?.backgroundColor = NSColor(calibratedRed: 0.03, green: 0.04, blue: 0.07, alpha: 0.48).cgColor
        container.layer?.cornerRadius = 11
        container.layer?.borderWidth = 1.0
        container.layer?.borderColor = NSColor(calibratedRed: 1.0, green: 0.52, blue: 0.0, alpha: 0.35).cgColor

        let imgView = NSImageView(frame: container.bounds)
        imgView.autoresizingMask = [.width, .height]
        imgView.imageScaling = .scaleAxesIndependently
        imgView.imageAlignment = .alignCenter
        imgView.image = loadedImg

        container.addSubview(imgView)
        topLeftBrandWindow.contentView = container
        videoWindow.addChildWindow(topLeftBrandWindow, ordered: .above)
    }

    // MARK: - 2. Sleek Translucent Floating Overlay Pill (With ✨ Magic Wand & ⚙️ Settings Gear)
    private func buildMinimalOverlayPill() {
        let pillWidth: CGFloat = 668
        let pillHeight: CGFloat = 42
        let contentRect = videoWindow.contentRect(forFrameRect: videoWindow.frame)
        let pillRect = NSRect(
            x: contentRect.origin.x + (contentRect.width - pillWidth) / 2.0,
            y: contentRect.origin.y + 16,
            width: pillWidth,
            height: pillHeight
        )

        overlayPillWindow = OverlayPillWindow(
            contentRect: pillRect,
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        overlayPillWindow.isOpaque = false
        overlayPillWindow.backgroundColor = .clear
        overlayPillWindow.hasShadow = true
        overlayPillWindow.level = .floating
        overlayPillWindow.sharingType = .readOnly
        overlayPillWindow.acceptsMouseMovedEvents = true
        overlayPillWindow.contentMinSize = NSSize(width: pillWidth, height: pillHeight)
        overlayPillWindow.contentMaxSize = NSSize(width: pillWidth, height: pillHeight)

        let glassView = NSVisualEffectView(frame: NSRect(x: 0, y: 0, width: pillWidth, height: pillHeight))
        glassView.autoresizingMask = [.width, .height]
        glassView.material = .hudWindow
        glassView.blendingMode = .withinWindow
        glassView.state = .active
        glassView.appearance = NSAppearance(named: .vibrantDark)
        glassView.wantsLayer = true
        glassView.layer?.cornerRadius = pillHeight / 2.0
        glassView.layer?.masksToBounds = true
        glassView.layer?.backgroundColor = NSColor(calibratedWhite: 0.06, alpha: 0.55).cgColor
        glassView.layer?.borderWidth = 1.0
        glassView.layer?.borderColor = NSColor.white.withAlphaComponent(0.16).cgColor

        let stack = NSStackView(frame: glassView.bounds.insetBy(dx: 14, dy: 6))
        stack.translatesAutoresizingMaskIntoConstraints = false
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.distribution = .fill
        stack.spacing = 7

        pillStatusDot = NSTextField(labelWithString: "●")
        pillStatusDot.font = NSFont.systemFont(ofSize: 12, weight: .bold)
        pillStatusDot.textColor = NSColor.systemGreen
        pillStatusDot.setContentHuggingPriority(.required, for: .horizontal)
        pillStatusDot.setContentCompressionResistancePriority(.required, for: .horizontal)
        attachHoverTooltip(pillStatusDot, "🟢 Live Connection Status Indicator")
        stack.addArrangedSubview(pillStatusDot)

        pillStatusLabel = NSTextField(labelWithString: "1920×1080 • 30 fps")
        pillStatusLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 11.5, weight: .semibold)
        pillStatusLabel.textColor = NSColor.white.withAlphaComponent(0.92)
        pillStatusLabel.lineBreakMode = .byTruncatingTail
        pillStatusLabel.maximumNumberOfLines = 1
        pillStatusLabel.usesSingleLineMode = true
        pillStatusLabel.cell?.wraps = false
        pillStatusLabel.cell?.isScrollable = false
        pillStatusLabel.cell?.truncatesLastVisibleLine = true
        pillStatusLabel.setContentHuggingPriority(NSLayoutConstraint.Priority(1), for: .horizontal)
        pillStatusLabel.setContentCompressionResistancePriority(NSLayoutConstraint.Priority(1), for: .horizontal)
        attachHoverTooltip(pillStatusLabel, "📡 Active Stream Resolution, FPS & Bitrate Telemetry")
        stack.addArrangedSubview(pillStatusLabel)

        pillFlipBtn = makePillIconButton(symbol: "camera.rotate.fill", tooltip: "🔄 Flip Camera Lens (Back Wide ↔ Front Selfie) [F]", action: #selector(onQuickFlipLens))
        stack.addArrangedSubview(pillFlipBtn)

        pillFlashBtn = makePillIconButton(symbol: "bolt.slash.fill", tooltip: "⚡ Cycle Flash Mode: Off → Torch On → Auto Flash [T]", action: #selector(onQuickCycleFlash))
        stack.addArrangedSubview(pillFlashBtn)

        // Compact Zoom Level Slider right on the Floating HUD Pill
        pillZoomSlider = NSSlider(value: 1.0, minValue: 1.0, maxValue: 6.0, target: self, action: #selector(onPillZoomSliderChanged))
        pillZoomSlider.isContinuous = true
        pillZoomSlider.controlSize = .small
        attachHoverTooltip(pillZoomSlider, "🔍 Hardware Sensor Zoom Level (1.0× – 6.0×) [+ / -]")
        pillZoomSlider.translatesAutoresizingMaskIntoConstraints = false
        pillZoomSlider.widthAnchor.constraint(equalToConstant: 72).isActive = true
        pillZoomSlider.setContentHuggingPriority(.required, for: .horizontal)
        pillZoomSlider.setContentCompressionResistancePriority(.required, for: .horizontal)
        stack.addArrangedSubview(pillZoomSlider)

        pillZoomLabel = NSTextField(labelWithString: "1.0×")
        pillZoomLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 10.5, weight: .bold)
        pillZoomLabel.textColor = NSColor.systemOrange
        pillZoomLabel.alignment = .right
        attachHoverTooltip(pillZoomLabel, "🔍 Current Hardware Zoom Ratio [+ / -]")
        pillZoomLabel.translatesAutoresizingMaskIntoConstraints = false
        pillZoomLabel.widthAnchor.constraint(equalToConstant: 30).isActive = true
        pillZoomLabel.setContentHuggingPriority(.required, for: .horizontal)
        pillZoomLabel.setContentCompressionResistancePriority(.required, for: .horizontal)
        stack.addArrangedSubview(pillZoomLabel)

        pillAFBtn = makePillIconButton(symbol: "scope", tooltip: "🎯 Toggle / Trigger Hardware Autofocus Lock & Sweep [A]", action: #selector(onTriggerAF))
        stack.addArrangedSubview(pillAFBtn)

        pillSnapBtn = makePillIconButton(symbol: "camera.aperture", tooltip: "📸 Save Full-Res JPEG Snapshot to ~/Downloads [P]", action: #selector(onTakeSnapshot))
        stack.addArrangedSubview(pillSnapBtn)

        pillRecordBtn = makePillIconButton(symbol: "record.circle", tooltip: "🔴 Start / Stop H.264 MP4 Video Recording to ~/Downloads [V]", action: #selector(onToggleRecording))
        pillRecordBtn.contentTintColor = NSColor.systemRed
        stack.addArrangedSubview(pillRecordBtn)

        // Magic Wand Button (Pxl Cam Dual-Track Auto-Fix!)
        pillWandBtn = makePillIconButton(symbol: "wand.and.stars", tooltip: "✨ AI Magic Wand: Toggle Auto-Tune Exposure, WB & Metal Studio ISP [W]", action: #selector(onPillWandButtonClicked))
        pillWandBtn.contentTintColor = NSColor.systemYellow
        stack.addArrangedSubview(pillWandBtn)

        // Settings Gear Button (Opens off-screen Settings Inspector!)
        pillGearBtn = makePillIconButton(symbol: "gearshape.fill", tooltip: "⚙️ Open / Close Pxl Cam Studio Controls [ , ]", action: #selector(onToggleSettingsWindow))
        pillGearBtn.contentTintColor = NSColor.systemCyan
        stack.addArrangedSubview(pillGearBtn)

        glassView.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: glassView.leadingAnchor, constant: 14),
            stack.trailingAnchor.constraint(equalTo: glassView.trailingAnchor, constant: -14),
            stack.topAnchor.constraint(equalTo: glassView.topAnchor, constant: 6),
            stack.bottomAnchor.constraint(equalTo: glassView.bottomAnchor, constant: -6)
        ])
        overlayPillWindow.contentView = glassView
        videoWindow.addChildWindow(overlayPillWindow, ordered: .above)
        repositionOverlayPill()
        updateDockButtonStates()
    }

    private func makePillIconButton(symbol: String, tooltip: String, action: Selector) -> NSButton {
        let btn = NSButton(frame: NSRect(x: 0, y: 0, width: 32, height: 28))
        btn.bezelStyle = .texturedRounded
        btn.isBordered = false
        btn.wantsLayer = true
        btn.layer?.cornerRadius = 14
        btn.layer?.backgroundColor = NSColor.white.withAlphaComponent(0.10).cgColor
        if let img = NSImage(systemSymbolName: symbol, accessibilityDescription: tooltip) {
            let cfg = NSImage.SymbolConfiguration(pointSize: 13, weight: .semibold)
            btn.image = img.withSymbolConfiguration(cfg)
        } else {
            btn.title = "⚙️"
        }
        btn.contentTintColor = .white
        btn.target = self
        btn.action = action
        btn.translatesAutoresizingMaskIntoConstraints = false
        btn.setContentHuggingPriority(.required, for: .horizontal)
        btn.setContentCompressionResistancePriority(.required, for: .horizontal)
        NSLayoutConstraint.activate([
            btn.widthAnchor.constraint(equalToConstant: 32),
            btn.heightAnchor.constraint(equalToConstant: 28)
        ])
        attachHoverTooltip(btn, tooltip, isPillIcon: true)
        return btn
    }

    /// Styles a dock pill icon button with a persistent glowing outline and active fill when ON,
    /// plus responsive hover feedback without stripping the active outline on mouse exit.
    private func stylePillButton(_ btn: NSButton?, isActive: Bool, activeColor: NSColor, inactiveTint: NSColor = .white) {
        guard let btn = btn else { return }
        let isHovered = (hoveredPillButton === btn)
        btn.wantsLayer = true
        btn.layer?.cornerRadius = 14
        if isActive {
            btn.layer?.borderWidth = isHovered ? 2.2 : 1.8
            btn.layer?.borderColor = (isHovered ? NSColor.white : activeColor).withAlphaComponent(0.98).cgColor
            if activeColor == .systemRed {
                btn.layer?.backgroundColor = NSColor.systemRed.withAlphaComponent(isHovered ? 0.92 : 0.82).cgColor
                btn.contentTintColor = .white
            } else {
                btn.layer?.backgroundColor = activeColor.withAlphaComponent(isHovered ? 0.38 : 0.28).cgColor
                btn.contentTintColor = activeColor
            }
        } else if isHovered {
            btn.layer?.borderWidth = 1.4
            btn.layer?.borderColor = NSColor.systemOrange.withAlphaComponent(0.92).cgColor
            btn.layer?.backgroundColor = NSColor.white.withAlphaComponent(0.16).cgColor
            btn.contentTintColor = inactiveTint
        } else {
            btn.layer?.borderWidth = 0.0
            btn.layer?.borderColor = NSColor.clear.cgColor
            btn.layer?.backgroundColor = NSColor.white.withAlphaComponent(0.09).cgColor
            btn.contentTintColor = inactiveTint
        }
    }

    /// Synchronizes the ON / OFF visual state and persistent active outline of all 7 dock buttons.
    private func updateDockButtonStates() {
        let now = CFAbsoluteTimeGetCurrent()
        // 1. Flip Lens Button: ON when Front Camera (1) is active
        stylePillButton(
            pillFlipBtn,
            isActive: currentLens == 1,
            activeColor: .systemOrange,
            inactiveTint: .white
        )

        // 2. Flash / Torch Button: ON when Torch (1) or Auto Flash (2) is active
        let flashColor: NSColor = (currentFlashMode == 2) ? .systemCyan : .systemYellow
        stylePillButton(
            pillFlashBtn,
            isActive: currentFlashMode != 0,
            activeColor: flashColor,
            inactiveTint: .white
        )

        // 3. Autofocus Button: ON when AF Lock / Sweep is active or non-default AF mode is active
        stylePillButton(
            pillAFBtn,
            isActive: isAFActive || currentAFMode != 0,
            activeColor: .systemGreen,
            inactiveTint: .white
        )

        // 4. Snapshot Button: Illuminates active cyan outline when a snapshot is captured
        stylePillButton(
            pillSnapBtn,
            isActive: now < snapActiveUntil,
            activeColor: .systemCyan,
            inactiveTint: .white
        )

        // 5. Record MP4 Button: ON with persistent red/white outline while recording
        let isRec = (videoView?.isRecording == true)
        stylePillButton(
            pillRecordBtn,
            isActive: isRec,
            activeColor: .systemRed,
            inactiveTint: .systemRed
        )

        // 6. AI Magic Wand Button: ON with persistent golden-yellow outline while AI Wand mode is active
        stylePillButton(
            pillWandBtn,
            isActive: isWandModeActive,
            activeColor: .systemYellow,
            inactiveTint: NSColor.systemYellow.withAlphaComponent(0.88)
        )
        if let aiWandHeroBtn = aiWandHeroBtn {
            aiWandHeroBtn.wantsLayer = true
            aiWandHeroBtn.layer?.cornerRadius = 6
            aiWandHeroBtn.layer?.borderWidth = isWandModeActive ? 1.5 : 0.0
            aiWandHeroBtn.layer?.borderColor = isWandModeActive ? NSColor.systemYellow.cgColor : NSColor.clear.cgColor
        }

        // 7. Settings Gear Button: ON with persistent cyan outline while Studio Controls window is open
        let isSettingsOpen = (settingsWindow?.isVisible == true)
        stylePillButton(
            pillGearBtn,
            isActive: isSettingsOpen,
            activeColor: .systemCyan,
            inactiveTint: NSColor.systemCyan.withAlphaComponent(0.88)
        )
    }

    // MARK: - 2A. Instant Studio Hover Tooltip System (Works on Non-Activating HUD & Settings Panels)
    private func buildHoverTooltipWindow() {
        hoverTooltipWindow = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 220, height: 28),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        hoverTooltipWindow.isOpaque = false
        hoverTooltipWindow.backgroundColor = .clear
        hoverTooltipWindow.hasShadow = true
        hoverTooltipWindow.level = .statusBar
        hoverTooltipWindow.ignoresMouseEvents = true
        hoverTooltipWindow.sharingType = .readOnly

        let bubble = NSView(frame: NSRect(x: 0, y: 0, width: 220, height: 28))
        bubble.autoresizingMask = [.width, .height]
        bubble.wantsLayer = true
        bubble.layer?.cornerRadius = 7.0
        bubble.layer?.masksToBounds = true
        bubble.layer?.backgroundColor = NSColor(calibratedRed: 0.05, green: 0.06, blue: 0.09, alpha: 0.96).cgColor
        bubble.layer?.borderWidth = 1.0
        bubble.layer?.borderColor = NSColor.systemOrange.withAlphaComponent(0.85).cgColor

        hoverTooltipLabel = NSTextField(labelWithString: "")
        hoverTooltipLabel.font = NSFont.systemFont(ofSize: 11.5, weight: .semibold)
        hoverTooltipLabel.textColor = NSColor(calibratedWhite: 0.97, alpha: 1.0)
        hoverTooltipLabel.alignment = .center
        hoverTooltipLabel.frame = NSRect(x: 10, y: 5, width: 200, height: 18)
        bubble.addSubview(hoverTooltipLabel)

        hoverTooltipWindow.contentView = bubble
        hoverTooltipWindow.orderOut(nil)
    }

    private func attachHoverTooltip(_ view: NSView, _ text: String, isPillIcon: Bool = false) {
        view.toolTip = text
        let area = NSTrackingArea(
            rect: .zero,
            options: [.mouseEnteredAndExited, .inVisibleRect, .activeAlways],
            owner: self,
            userInfo: [
                "tip": text,
                "view": view,
                "pill": isPillIcon
            ]
        )
        view.addTrackingArea(area)
    }

    private func showHoverTooltipBubble(text: String, for view: NSView) {
        guard let win = view.window, win.isVisible, win.alphaValue > 0.1 else { return }
        hoverTooltipLabel.stringValue = text
        hoverTooltipLabel.sizeToFit()
        let textSize = hoverTooltipLabel.frame.size
        let padX: CGFloat = 12
        let padY: CGFloat = 5
        let bubbleW = ceil(textSize.width + padX * 2)
        let bubbleH = ceil(max(26, textSize.height + padY * 2))

        hoverTooltipLabel.frame = NSRect(
            x: padX,
            y: round((bubbleH - textSize.height) / 2.0),
            width: ceil(textSize.width),
            height: ceil(textSize.height)
        )

        let viewRectInWindow = view.convert(view.bounds, to: nil)
        let viewRectOnScreen = win.convertToScreen(viewRectInWindow)

        var tipX = round(viewRectOnScreen.midX - bubbleW / 2.0)
        var tipY = round(viewRectOnScreen.maxY + 7.0)

        if let screenFrame = (win.screen ?? NSScreen.main)?.visibleFrame {
            tipX = max(screenFrame.minX + 8, min(tipX, screenFrame.maxX - bubbleW - 8))
            if tipY + bubbleH > screenFrame.maxY - 6 {
                tipY = round(viewRectOnScreen.minY - bubbleH - 7.0)
            }
        }

        hoverTooltipWindow.setFrame(NSRect(x: tipX, y: tipY, width: bubbleW, height: bubbleH), display: true)
        hoverTooltipWindow.orderFrontRegardless()
    }

    override func mouseEntered(with event: NSEvent) {
        guard let info = event.trackingArea?.userInfo,
              let tip = info["tip"] as? String,
              let view = info["view"] as? NSView else {
            return
        }
        showControlsAndResetTimer()
        if let isPill = info["pill"] as? Bool, isPill {
            hoveredPillButton = view
            updateDockButtonStates()
        }
        showHoverTooltipBubble(text: tip, for: view)
    }

    override func mouseExited(with event: NSEvent) {
        if let info = event.trackingArea?.userInfo,
           let view = info["view"] as? NSView,
           let isPill = info["pill"] as? Bool, isPill {
            if hoveredPillButton === view {
                hoveredPillButton = nil
            }
            updateDockButtonStates()
        }
        hoverTooltipWindow?.orderOut(nil)
    }

    private func repositionOverlayPill() {
        guard let videoWindow = videoWindow, let overlayPillWindow = overlayPillWindow else { return }
        let contentRect = videoWindow.contentRect(forFrameRect: videoWindow.frame)
        let pillWidth: CGFloat = min(668, max(480, contentRect.width - 20))
        let pillHeight: CGFloat = 42
        let pillX = round(contentRect.origin.x + (contentRect.width - pillWidth) / 2.0)
        let pillY = round(contentRect.origin.y + 16)
        overlayPillWindow.contentMinSize = NSSize(width: pillWidth, height: pillHeight)
        overlayPillWindow.contentMaxSize = NSSize(width: pillWidth, height: pillHeight)
        overlayPillWindow.setFrame(NSRect(x: pillX, y: pillY, width: pillWidth, height: pillHeight), display: true)

        if let topLeftBrandWindow = topLeftBrandWindow {
            let brandHeight: CGFloat = 96
            let brandWidth: CGFloat = round(brandHeight * topLeftBrandAspect)
            let brandX = contentRect.origin.x + 16
            let brandY = contentRect.maxY - brandHeight - 14
            topLeftBrandWindow.setFrame(NSRect(x: brandX, y: brandY, width: brandWidth, height: brandHeight), display: true)
        }
    }

    // MARK: - 2B. Auto-Hiding Controls Timer (3.0s Inactivity Like aarynfans-mvp)
    private func setupControlsAutoHideTimer() {
        resetControlsHideTimer()
        controlsEventMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.mouseMoved, .leftMouseDown, .rightMouseDown, .otherMouseDown, .leftMouseDragged, .scrollWheel, .keyDown]
        ) { [weak self] event in
            self?.showControlsAndResetTimer()
            return event
        }
    }

    private func showControlsAndResetTimer() {
        guard hudVisible, let overlayPillWindow = overlayPillWindow else { return }
        if overlayPillWindow.alphaValue < 0.99 || overlayPillWindow.ignoresMouseEvents || (topLeftBrandWindow?.alphaValue ?? 1.0) < 0.99 {
            overlayPillWindow.ignoresMouseEvents = false
            NSAnimationContext.runAnimationGroup { ctx in
                ctx.duration = 0.30
                ctx.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
                overlayPillWindow.animator().alphaValue = 1.0
                topLeftBrandWindow?.animator().alphaValue = 1.0
            }
        }
        resetControlsHideTimer()
    }

    private func resetControlsHideTimer() {
        controlsHideTimer?.invalidate()
        controlsHideTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { [weak self] _ in
            self?.hideControlsAnimated()
        }
    }

    private func hideControlsAnimated() {
        guard hudVisible, let overlayPillWindow = overlayPillWindow else { return }
        // Keep controls visible if settings inspector is open or cursor is directly hovering inside the pill
        if settingsWindow?.isVisible == true || overlayPillWindow.frame.contains(NSEvent.mouseLocation) {
            resetControlsHideTimer()
            return
        }
        overlayPillWindow.ignoresMouseEvents = true
        NSAnimationContext.runAnimationGroup { ctx in
            ctx.duration = 0.30
            ctx.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
            overlayPillWindow.animator().alphaValue = 0.0
            topLeftBrandWindow?.animator().alphaValue = 0.0
        }
    }

    func windowDidResize(_ notification: Notification) {
        repositionOverlayPill()
    }

    func windowDidMove(_ notification: Notification) {
        repositionOverlayPill()
    }

    func windowWillClose(_ notification: Notification) {
        if let closingWin = notification.object as? NSWindow, closingWin === settingsWindow {
            DispatchQueue.main.async { [weak self] in
                self?.updateDockButtonStates()
            }
        }
    }

    // MARK: - 3. Separate Off-Screen Settings Window (Opened via ⚙️ Gear)
    private func buildSettingsInspectorWindow() {
        let videoFrame = videoWindow.frame
        let inspectorWidth: CGFloat = 396
        let inspectorHeight: CGFloat = 746
        let inspectorX = min(videoFrame.maxX + 14, (NSScreen.main?.visibleFrame.maxX ?? 1600) - inspectorWidth - 10)
        let inspectorY = max(videoFrame.minY, 60)

        settingsWindow = NSPanel(
            contentRect: NSRect(x: inspectorX, y: inspectorY, width: inspectorWidth, height: inspectorHeight),
            styleMask: [.titled, .closable, .utilityWindow, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        settingsWindow.title = "⚙️ Pxl Cam — Studio Controls"
        settingsWindow.appearance = NSAppearance(named: .darkAqua)
        settingsWindow.isFloatingPanel = true
        settingsWindow.becomesKeyOnlyIfNeeded = false
        settingsWindow.sharingType = .readOnly
        settingsWindow.acceptsMouseMovedEvents = true
        settingsWindow.delegate = self

        let bgView = NSVisualEffectView(frame: NSRect(x: 0, y: 0, width: inspectorWidth, height: inspectorHeight))
        bgView.autoresizingMask = [.width, .height]
        bgView.material = .sidebar
        bgView.blendingMode = .behindWindow
        bgView.state = .active

        let mainStack = NSStackView(frame: NSRect(x: 18, y: 14, width: inspectorWidth - 36, height: inspectorHeight - 28))
        mainStack.autoresizingMask = [.width, .height]
        mainStack.orientation = .vertical
        mainStack.alignment = .leading
        mainStack.distribution = .equalSpacing
        mainStack.spacing = 8

        // --- SECTION 0: PXL CAM AI MAGIC WAND & RESET DEFAULTS ---
        let wandHeaderRow = NSStackView()
        wandHeaderRow.orientation = .horizontal
        wandHeaderRow.alignment = .centerY
        wandHeaderRow.spacing = 8
        aiWandHeroBtn = NSButton(title: "✨ Auto-Fix Scene (AI) [W]", target: self, action: #selector(onMagicWandAutoTune))
        aiWandHeroBtn.bezelStyle = .rounded
        aiWandHeroBtn.contentTintColor = .systemYellow
        aiWandHeroBtn.font = NSFont.systemFont(ofSize: 12, weight: .semibold)
        attachHoverTooltip(aiWandHeroBtn, "✨ Analyze live scene and auto-tune exposure, white balance & Metal Studio ISP [W]")

        let topResetDefaultsBtn = NSButton(title: "↺ Reset Defaults [D]", target: self, action: #selector(onResetAllDefaults))
        topResetDefaultsBtn.bezelStyle = .rounded
        topResetDefaultsBtn.contentTintColor = .systemCyan
        topResetDefaultsBtn.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        attachHoverTooltip(topResetDefaultsBtn, "↺ Reset all Camera2 hardware, Flash, WB, Zoom, EV, Orientation & Metal Studio ISP settings to defaults [D]")

        wandHeaderRow.addArrangedSubview(aiWandHeroBtn)
        wandHeaderRow.addArrangedSubview(topResetDefaultsBtn)
        mainStack.addArrangedSubview(wandHeaderRow)

        aiTelemetryLabel = NSTextField(labelWithString: "Dual-Track AI: Ready (Laplacian + Noise + HSV + Face Vision)")
        aiTelemetryLabel.font = NSFont.monospacedDigitSystemFont(ofSize: 10.5, weight: .medium)
        aiTelemetryLabel.textColor = NSColor.systemYellow.withAlphaComponent(0.92)
        aiTelemetryLabel.lineBreakMode = .byTruncatingTail
        aiTelemetryLabel.translatesAutoresizingMaskIntoConstraints = false
        aiTelemetryLabel.widthAnchor.constraint(equalToConstant: 356).isActive = true
        attachHoverTooltip(aiTelemetryLabel, "📊 Real-time Dual-Track AI Scene Telemetry (Sharpness, Noise, Saturation & Face Detection)")
        mainStack.addArrangedSubview(aiTelemetryLabel)

        mainStack.addArrangedSubview(makeDivider())

        // --- SECTION 1: CAMERA & FORMAT ---
        mainStack.addArrangedSubview(makeSectionHeader("CAMERA & RESOLUTION"))

        lensPopup = makePopup(["Back Camera (0 — Wide)", "Front Camera (1 — Selfie)"], selected: 0, action: #selector(onLensPopupChanged))
        mainStack.addArrangedSubview(makeFormRow("Camera Lens:", control: lensPopup, tooltip: "📷 Switch between Back Wide Camera (0) and Front Selfie Camera (1) [F]"))

        resPopup = makePopup([
            "4K UHD (3840×2160)",
            "QHD+ 3.2K (3264×1836)",
            "1080p Full HD (1920×1080 — Default)",
            "720p HD Fast (1280×720)"
        ], selected: 2, action: #selector(onResPopupChanged))
        mainStack.addArrangedSubview(makeFormRow("Resolution:", control: resPopup, tooltip: "📐 Select Hardware Capture Resolution (4K, QHD+, 1080p Default, or 720p) [1 / 2 / 3]"))

        fpsModePopup = makePopup([
            "Low-Light Auto (15–30 fps • Bright)",
            "Studio Locked (30 fps • Fast Shutter)",
            "Max Night Boost (5–30 fps • Max Light)"
        ], selected: 0, action: #selector(onFPSModeChanged))
        mainStack.addArrangedSubview(makeFormRow("Shutter / Light:", control: fpsModePopup, tooltip: "⏱ Select Shutter & Low-Light Integration Mode (Auto 15–30fps, Locked 30fps, or Max Night Boost)"))

        mainStack.addArrangedSubview(makeDivider())

        // --- SECTION 2: FLASH, FOCUS & WHITE BALANCE ---
        mainStack.addArrangedSubview(makeSectionHeader("FLASH, FOCUS & WHITE BALANCE"))

        flashSegment = NSSegmentedControl(labels: ["Off", "On (Torch)", "Auto Flash"], trackingMode: .selectOne, target: self, action: #selector(onFlashSegmentChanged))
        flashSegment.selectedSegment = 0
        mainStack.addArrangedSubview(makeFormRow("Flash Mode:", control: flashSegment, tooltip: "⚡ Select Rear LED Flash Mode: Off, Continuous Torch, or Auto Flash [T]"))

        let (fdRow, fdSlider, fdVal) = makeSliderRow(title: "Flash Dimmer:", min: 0.0, max: 100.0, val: 0.0, fmt: "OFF", action: #selector(onFlashDimmerSliderChanged), tooltip: "💡 Torch / Flash Dimmer (0% = OFF, 1%–100% = Smooth Optical + AE Torch Dimming)")
        flashDimmerSlider = fdSlider; flashDimmerLabel = fdVal
        mainStack.addArrangedSubview(fdRow)

        let (tzRow, tzSlider, tzVal) = makeSliderRow(title: "Zoom Level:", min: 1.0, max: 6.0, val: 1.0, fmt: "1.0×", action: #selector(onTopZoomSliderChanged), tooltip: "🔍 Hardware Sensor Crop & Zoom Level (1.0× – 6.0×) [+ / -]")
        topZoomSlider = tzSlider; topZoomLabel = tzVal
        mainStack.addArrangedSubview(tzRow)

        wbPopup = makePopup([
            "Auto White Balance",
            "Daylight (5500K)",
            "Cloudy (6500K)",
            "Warm / Incandescent (2800K)",
            "Fluorescent (4000K)",
            "Shade / Twilight"
        ], selected: 0, action: #selector(onWBChanged))
        mainStack.addArrangedSubview(makeFormRow("White Balance:", control: wbPopup, tooltip: "🌡 Select Camera2 White Balance Preset (Auto, Daylight, Cloudy, Warm, Fluorescent, Shade)"))

        afModePopup = makePopup([
            "Continuous Video AF",
            "Macro Close-Up AF",
            "Locked Focus (Manual)"
        ], selected: 0, action: #selector(onAFModeChanged))
        mainStack.addArrangedSubview(makeFormRow("Focus Mode:", control: afModePopup, tooltip: "🎯 Select Autofocus Mode (Continuous Video AF, Macro Close-Up, or Locked Manual Focus)"))

        let locksRow = NSStackView()
        locksRow.orientation = .horizontal
        locksRow.spacing = 14
        aeLockCheck = NSButton(checkboxWithTitle: "Lock Exposure (AE)", target: self, action: #selector(onAELockChanged))
        attachHoverTooltip(aeLockCheck, "🔒 Lock Hardware Auto-Exposure (AE) at current brightness")
        awbLockCheck = NSButton(checkboxWithTitle: "Lock WB (AWB)", target: self, action: #selector(onAWBLockChanged))
        attachHoverTooltip(awbLockCheck, "🔒 Lock Hardware Auto White Balance (AWB) at current color temperature")
        let refocusBtn = NSButton(title: "🎯 Refocus", target: self, action: #selector(onTriggerAF))
        refocusBtn.bezelStyle = .rounded
        attachHoverTooltip(refocusBtn, "🎯 Trigger a fresh hardware autofocus sweep right now [A]")
        locksRow.addArrangedSubview(aeLockCheck)
        locksRow.addArrangedSubview(awbLockCheck)
        locksRow.addArrangedSubview(refocusBtn)
        mainStack.addArrangedSubview(locksRow)

        mainStack.addArrangedSubview(makeDivider())

        // --- SECTION 3: HARDWARE SENSOR OPTICS ---
        mainStack.addArrangedSubview(makeSectionHeader("HARDWARE SENSOR OPTICS"))

        let (eRow, eSlider, eVal) = makeSliderRow(title: "Sensor EV:", min: -12.0, max: 12.0, val: 0.0, fmt: "+0 EV", action: #selector(onEVSliderChanged), tooltip: "☀️ Hardware Sensor Exposure Compensation (-12 EV to +12 EV) [[ / ]]")
        evSlider = eSlider; evLabel = eVal
        mainStack.addArrangedSubview(eRow)

        let (qRow, qSlider, qVal) = makeSliderRow(title: "ISP Quality:", min: 60.0, max: 98.0, val: 92.0, fmt: "92%", action: #selector(onQualitySliderChanged), tooltip: "🎞 Stream Encoding Quality (60% – 98%, Default 92%)")
        qualitySlider = qSlider; qualityLabel = qVal
        mainStack.addArrangedSubview(qRow)

        mainStack.addArrangedSubview(makeDivider())

        // --- SECTION 4: METAL GPU STUDIO ISP & LOW-LIGHT BOOST ---
        let ispHeaderRow = NSStackView()
        ispHeaderRow.orientation = .horizontal
        ispHeaderRow.spacing = 12
        ispHeaderRow.addArrangedSubview(makeSectionHeader("METAL STUDIO ISP & BRIGHTNESS"))
        ispCheck = NSButton(checkboxWithTitle: "Denoise + Sharpen", target: self, action: #selector(onISPCheckChanged))
        ispCheck.state = .on
        attachHoverTooltip(ispCheck, "✨ Enable/Disable Real-Time Metal GPU Bilateral Denoise & Luma Unsharp Mask")
        ispHeaderRow.addArrangedSubview(ispCheck)
        mainStack.addArrangedSubview(ispHeaderRow)

        let (bRow, bSlider, bVal) = makeSliderRow(title: "Brightness:", min: 0.6, max: 3.2, val: 1.12, fmt: "1.12×", action: #selector(onGPUGradeChanged), tooltip: "☀️ Metal Studio ISP Brightness Gain (0.60× – 3.20×)")
        brightnessSlider = bSlider; brightnessLabel = bVal
        mainStack.addArrangedSubview(bRow)

        let (sRow, sSlider, sVal) = makeSliderRow(title: "Shadow Lift:", min: 0.0, max: 0.90, val: 0.16, fmt: "+16%", action: #selector(onGPUGradeChanged), tooltip: "🌗 Metal Studio ISP Shadow Lift (Opens dark shadows without blowing highlights)")
        shadowSlider = sSlider; shadowLabel = sVal
        mainStack.addArrangedSubview(sRow)

        let (cRow, cSlider, cVal) = makeSliderRow(title: "Contrast:", min: 0.7, max: 1.45, val: 1.05, fmt: "1.05×", action: #selector(onGPUGradeChanged), tooltip: "◑ Metal Studio ISP S-Curve Contrast Boost (0.70× – 1.45×)")
        contrastSlider = cSlider; contrastLabel = cVal
        mainStack.addArrangedSubview(cRow)

        let (satRow, satSlider, satVal) = makeSliderRow(title: "Saturation:", min: 0.0, max: 1.8, val: 1.04, fmt: "1.04×", action: #selector(onGPUGradeChanged), tooltip: "🎨 Metal Studio ISP Color Saturation (0.00× Monochrome – 1.80× Vibrant)")
        saturationSlider = satSlider; saturationLabel = satVal
        mainStack.addArrangedSubview(satRow)

        let (dnRow, dnSlider, dnVal) = makeSliderRow(title: "Denoise:", min: 0.0, max: 0.95, val: 0.38, fmt: "0.38", action: #selector(onGPUGradeChanged), tooltip: "🧹 Metal GPU Bilateral + Temporal 3DNR Chroma/Luma Denoise Strength")
        denoiseSlider = dnSlider; denoiseLabel = dnVal
        mainStack.addArrangedSubview(dnRow)

        let (shpRow, shpSlider, shpVal) = makeSliderRow(title: "Sharpness:", min: 0.0, max: 1.2, val: 0.52, fmt: "0.52", action: #selector(onGPUGradeChanged), tooltip: "🔪 Metal GPU Luma-Adaptive Unsharp Mask Detail Sharpness")
        sharpnessSlider = shpSlider; sharpnessLabel = shpVal
        mainStack.addArrangedSubview(shpRow)

        mainStack.addArrangedSubview(makeDivider())

        // --- SECTION 5: ORIENTATION & PHONE POWER ---
        mainStack.addArrangedSubview(makeSectionHeader("ORIENTATION & DEVICE POWER"))

        let transformRow = NSStackView()
        transformRow.orientation = .horizontal
        transformRow.spacing = 8
        autoRotateCheck = NSButton(checkboxWithTitle: "📱 Auto-Rotate", target: self, action: #selector(onTransformChanged))
        autoRotateCheck.state = .on
        attachHoverTooltip(autoRotateCheck, "📱 Automatically rotate window between 16:9 Landscape & 9:16 Portrait with phone orientation")
        mirrorCheck = NSButton(checkboxWithTitle: "Mirror ⇄", target: self, action: #selector(onTransformChanged))
        attachHoverTooltip(mirrorCheck, "⇄ Mirror video horizontally (Left ↔ Right) [M]")
        flipVCheck = NSButton(checkboxWithTitle: "Flip ⇅", target: self, action: #selector(onTransformChanged))
        attachHoverTooltip(flipVCheck, "⇅ Flip video vertically (Top ↔ Bottom)")
        let rotBtn = NSButton(title: "↻ 90°", target: self, action: #selector(onRotate90))
        rotBtn.bezelStyle = .rounded
        attachHoverTooltip(rotBtn, "↻ Rotate video 90° clockwise [R]")
        let resetBtn = NSButton(title: "↺ Reset Defaults", target: self, action: #selector(onResetAllDefaults))
        resetBtn.bezelStyle = .rounded
        attachHoverTooltip(resetBtn, "↺ Reset all Camera2 & Studio ISP settings to defaults [D]")
        transformRow.addArrangedSubview(autoRotateCheck)
        transformRow.addArrangedSubview(mirrorCheck)
        transformRow.addArrangedSubview(flipVCheck)
        transformRow.addArrangedSubview(rotBtn)
        transformRow.addArrangedSubview(resetBtn)
        mainStack.addArrangedSubview(transformRow)

        let bottomRow = NSStackView()
        bottomRow.orientation = .horizontal
        bottomRow.spacing = 8
        screenOffCheck = NSButton(checkboxWithTitle: "🌙 Screen Off", target: self, action: #selector(onScreenOffChanged))
        screenOffCheck.state = .on
        attachHoverTooltip(screenOffCheck, "🌙 Dim phone OLED screen to save battery & eliminate heat while streaming")
        let snapBtn = NSButton(title: "📸 Snapshot [P]", target: self, action: #selector(onTakeSnapshot))
        snapBtn.bezelStyle = .rounded
        attachHoverTooltip(snapBtn, "📸 Save full-resolution JPEG snapshot to ~/Downloads [P]")
        recordBtn = NSButton(title: "⏺ Record MP4 [V]", target: self, action: #selector(onToggleRecording))
        recordBtn.bezelStyle = .rounded
        recordBtn.contentTintColor = .systemRed
        attachHoverTooltip(recordBtn, "🔴 Start / Stop H.264 MP4 Video Recording to ~/Downloads [V]")
        bottomRow.addArrangedSubview(screenOffCheck)
        bottomRow.addArrangedSubview(snapBtn)
        bottomRow.addArrangedSubview(recordBtn)
        mainStack.addArrangedSubview(bottomRow)

        bgView.addSubview(mainStack)
        settingsWindow.contentView = bgView
        // Start hidden so only the clean video + minimal gear pill is visible until ⚙️ is clicked!
        settingsWindow.orderOut(nil)
    }

    // MARK: - UI Builders for Settings Window
    private func makeSectionHeader(_ title: String) -> NSTextField {
        let lbl = NSTextField(labelWithString: title)
        lbl.font = NSFont.systemFont(ofSize: 10.5, weight: .bold)
        lbl.textColor = NSColor.systemCyan.withAlphaComponent(0.9)
        return lbl
    }

    private func makeDivider() -> NSBox {
        let box = NSBox()
        box.boxType = .separator
        box.translatesAutoresizingMaskIntoConstraints = false
        box.widthAnchor.constraint(equalToConstant: 356).isActive = true
        return box
    }

    private func makePopup(_ items: [String], selected: Int, action: Selector) -> NSPopUpButton {
        let popup = NSPopUpButton(frame: .zero, pullsDown: false)
        popup.addItems(withTitles: items)
        popup.selectItem(at: selected)
        popup.target = self
        popup.action = action
        popup.font = NSFont.systemFont(ofSize: 12)
        popup.translatesAutoresizingMaskIntoConstraints = false
        popup.widthAnchor.constraint(equalToConstant: 235).isActive = true
        return popup
    }

    private func makeFormRow(_ labelText: String, control: NSView, tooltip: String = "") -> NSStackView {
        let row = NSStackView()
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8
        let lbl = NSTextField(labelWithString: labelText)
        lbl.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        lbl.textColor = .secondaryLabelColor
        lbl.translatesAutoresizingMaskIntoConstraints = false
        lbl.widthAnchor.constraint(equalToConstant: 105).isActive = true
        if !tooltip.isEmpty {
            attachHoverTooltip(lbl, tooltip)
            attachHoverTooltip(control, tooltip)
        }
        row.addArrangedSubview(lbl)
        row.addArrangedSubview(control)
        return row
    }

    private func makeSliderRow(title: String, min: Double, max: Double, val: Double, fmt: String, action: Selector, tooltip: String = "") -> (NSStackView, NSSlider, NSTextField) {
        let row = NSStackView()
        row.orientation = .horizontal
        row.alignment = .centerY
        row.spacing = 8

        let lbl = NSTextField(labelWithString: title)
        lbl.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        lbl.textColor = .secondaryLabelColor
        lbl.translatesAutoresizingMaskIntoConstraints = false
        lbl.widthAnchor.constraint(equalToConstant: 95).isActive = true

        let slider = NSSlider(value: val, minValue: min, maxValue: max, target: self, action: action)
        slider.isContinuous = true
        slider.translatesAutoresizingMaskIntoConstraints = false
        slider.widthAnchor.constraint(equalToConstant: 185).isActive = true

        let valLbl = NSTextField(labelWithString: fmt)
        valLbl.font = NSFont.monospacedDigitSystemFont(ofSize: 11.5, weight: .semibold)
        valLbl.textColor = .white
        valLbl.alignment = .right
        valLbl.translatesAutoresizingMaskIntoConstraints = false
        valLbl.widthAnchor.constraint(equalToConstant: 54).isActive = true

        if !tooltip.isEmpty {
            attachHoverTooltip(lbl, tooltip)
            attachHoverTooltip(slider, tooltip)
            attachHoverTooltip(valLbl, tooltip)
        }

        row.addArrangedSubview(lbl)
        row.addArrangedSubview(slider)
        row.addArrangedSubview(valLbl)
        return (row, slider, valLbl)
    }

    // MARK: - Actions & Commands
    @objc private func onToggleSettingsWindow() {
        hoverTooltipWindow?.orderOut(nil)
        if settingsWindow.isVisible {
            settingsWindow.orderOut(nil)
        } else {
            let vFrame = videoWindow.frame
            let screenMaxX = NSScreen.main?.visibleFrame.maxX ?? 1600
            var x = vFrame.maxX + 12
            if x + settingsWindow.frame.width > screenMaxX {
                x = max(20, vFrame.minX - settingsWindow.frame.width - 12)
            }
            settingsWindow.setFrameOrigin(NSPoint(x: x, y: max(60, vFrame.maxY - settingsWindow.frame.height)))
            settingsWindow.makeKeyAndOrderFront(nil)
        }
        updateDockButtonStates()
    }

    @objc private func onPillWandButtonClicked() {
        showControlsAndResetTimer()
        if isWandModeActive {
            isWandModeActive = false
            onResetGrade()
            wandBannerText = "✨ AI Wand Off • Studio ISP"
            wandBannerUntil = CFAbsoluteTimeGetCurrent() + 2.2
            pillStatusLabel.stringValue = wandBannerText
            aiTelemetryLabel.stringValue = "✨ AI Magic Wand disabled — restored Studio Default ISP grade"
            updateDockButtonStates()
        } else {
            onMagicWandAutoTune()
        }
    }

    /// Pxl Cam Dual-Track Magic Wand Auto-Tuner:
    /// Calibrated for Android's native 64-point hardware sRGB + ADRC tone curve (`TONEMAP_MODE_FAST`).
    /// Combines Track A (Calibrated Camera2 Hardware Sensor Rules) + Track B (Continuous Feature-to-ISP Mapping)
    /// across three closed-loop passes (Pass 1 immediate + Pass 2 at 0.85s + Pass 3 at 1.65s after sensor AE settling).
    @objc private func onMagicWandAutoTune() {
        isWandModeActive = true
        currentPreset = -1
        updateDockButtonStates()
        guard let stats = videoView.analyzeCurrentFrame(detectFaces: true) else {
            wandBannerText = "✨ Waiting for live video..."
            wandBannerUntil = CFAbsoluteTimeGetCurrent() + 2.0
            pillStatusLabel.stringValue = wandBannerText
            return
        }

        // Unlock AE & AWB first so hardware can respond
        if aeLockCheck.state == .on {
            aeLockCheck.state = .off
            sendCommand("AELOCK:0")
        }
        if awbLockCheck.state == .on {
            awbLockCheck.state = .off
            sendCommand("AWBLOCK:0")
        }

        // --- TRACK A: Calibrated Hardware Sensor Rules (FPSMODE + EV + White Balance) ---
        let targetStudioLuma: Float = (stats.faceCount > 0) ? 0.43 : 0.41
        let effectiveRawLuma = max(0.55 * stats.rawMeanLuma + 0.45 * stats.p50Luma, 0.03)

        let recommendedFPSMode: Int
        let profileTag: String
        if stats.avgBrightness255 < 58.0 || effectiveRawLuma < 0.21 {
            recommendedFPSMode = 2 // Max Night Boost ([15, 20] -> 50ms-66.7ms shutter on Moto G Pure, 2x real light!)
            profileTag = "Night Boost"
        } else if stats.avgBrightness255 < 140.0 || effectiveRawLuma < 0.48 {
            recommendedFPSMode = 0 // Low-Light Auto ([15, 30])
            profileTag = (effectiveRawLuma < 0.32) ? "Low-Light Fix" : "Studio Balanced"
        } else {
            recommendedFPSMode = 1 // Studio Locked 30 fps (ample light)
            profileTag = (stats.brightClipFraction > 0.08) ? "Highlight Protect" : "Bright Studio"
        }

        if currentFPSMode != recommendedFPSMode {
            currentFPSMode = recommendedFPSMode
            fpsModePopup.selectItem(at: recommendedFPSMode)
            sendCommand("FPSMODE:\(recommendedFPSMode)")
        }

        // Compute hardware EV step adjustment from log2 exposure ratio.
        // With Android's 64-point hardware sRGB + ADRC tone curve active, only boost hardware EV when genuinely dark.
        let stopsNeeded = log2(targetStudioLuma / effectiveRawLuma)
        var evDelta = Int((stopsNeeded * 1.15).rounded())
        if effectiveRawLuma >= 0.31 && evDelta > 1 {
            evDelta = 1
        }
        if stats.brightClipFraction > 0.10 && evDelta > -1 {
            evDelta -= 1
        }
        let oldEV = currentEV
        let maxSafeHardwareEV = (effectiveRawLuma < 0.22) ? 3 : 2
        let newEV = min(max(oldEV + evDelta, -3), maxSafeHardwareEV)
        if newEV != currentEV {
            currentEV = newEV
            evSlider.intValue = Int32(newEV)
            evLabel.stringValue = "\(newEV >= 0 ? "+" : "")\(newEV) EV"
            sendCommand("EV:\(newEV)")
        }

        // Always keep Hardware White Balance on Auto (WB:0) to guarantee zero pink/magenta cast.
        if currentWBMode != 0 {
            currentWBMode = 0
            wbPopup.selectItem(at: 0)
        }
        sendCommand("WB:0")

        // Trigger fresh autofocus pass
        sendCommand("AF:TRIGGER")

        // --- TRACK B: Continuous Feature-to-ISP Mapping (Pass 1 Immediate Compensation) ---
        applyDualTrackISPGrade(from: stats, passNumber: 1, appliedEVDelta: newEV - oldEV, profileTag: profileTag)

        // --- PASS 2: Closed-Loop Fine-Tune after Android Sensor AE settles (0.85s later) ---
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.85) { [weak self] in
            guard let self = self, self.isWandModeActive else { return }
            guard let settledStats = self.videoView.analyzeCurrentFrame(detectFaces: false) else { return }
            let mergedStats = FrameSceneStats(
                rawMeanLuma: settledStats.rawMeanLuma,
                avgBrightness255: settledStats.avgBrightness255,
                exposureBias: settledStats.exposureBias,
                contrastStd255: settledStats.contrastStd255,
                laplacianSharpness: settledStats.laplacianSharpness,
                gaussianNoiseStd: settledStats.gaussianNoiseStd,
                hsvSaturation255: settledStats.hsvSaturation255,
                p10Luma: settledStats.p10Luma,
                p50Luma: settledStats.p50Luma,
                p90Luma: settledStats.p90Luma,
                darkPixelFraction: settledStats.darkPixelFraction,
                brightClipFraction: settledStats.brightClipFraction,
                redGreenRatio: settledStats.redGreenRatio,
                blueGreenRatio: settledStats.blueGreenRatio,
                sceneComplexity: settledStats.sceneComplexity,
                faceCount: stats.faceCount,
                faceCoverage: stats.faceCoverage
            )
            self.applyDualTrackISPGrade(from: mergedStats, passNumber: 2, appliedEVDelta: 0, profileTag: profileTag)
        }

        // --- PASS 3: Final Convergence Pass after long-shutter Night Boost fully stabilizes (1.65s later) ---
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.65) { [weak self] in
            guard let self = self, self.isWandModeActive else { return }
            self.updateDockButtonStates()
            guard let finalStats = self.videoView.analyzeCurrentFrame(detectFaces: false) else { return }
            let mergedStats = FrameSceneStats(
                rawMeanLuma: finalStats.rawMeanLuma,
                avgBrightness255: finalStats.avgBrightness255,
                exposureBias: finalStats.exposureBias,
                contrastStd255: finalStats.contrastStd255,
                laplacianSharpness: finalStats.laplacianSharpness,
                gaussianNoiseStd: finalStats.gaussianNoiseStd,
                hsvSaturation255: finalStats.hsvSaturation255,
                p10Luma: finalStats.p10Luma,
                p50Luma: finalStats.p50Luma,
                p90Luma: finalStats.p90Luma,
                darkPixelFraction: finalStats.darkPixelFraction,
                brightClipFraction: finalStats.brightClipFraction,
                redGreenRatio: finalStats.redGreenRatio,
                blueGreenRatio: finalStats.blueGreenRatio,
                sceneComplexity: finalStats.sceneComplexity,
                faceCount: stats.faceCount,
                faceCoverage: stats.faceCoverage
            )
            self.applyDualTrackISPGrade(from: mergedStats, passNumber: 3, appliedEVDelta: 0, profileTag: profileTag)
        }
    }

    private func applyDualTrackISPGrade(from stats: FrameSceneStats, passNumber: Int, appliedEVDelta: Int, profileTag: String) {
        let baseTargetLuma: Float = (stats.faceCount > 0) ? 0.31 : 0.28
        let targetLuma: Float = (stats.p10Luma < 0.06) ? min(baseTargetLuma, max(0.22, stats.p90Luma * 0.55)) : baseTargetLuma
        let sceneRawLuma = 0.50 * stats.rawMeanLuma + 0.30 * stats.p50Luma + 0.20 * stats.p90Luma
        let expectedHardwareFactor: Float = (passNumber == 1)
            ? pow(2.0, Float(appliedEVDelta) * 0.16)
            : 1.0
        let predictedRawLuma = min(max(sceneRawLuma * expectedHardwareFactor, 0.06), 0.92)

        // 1. Shadow Lift (calibrated for native sRGB + ADRC tone curve so shadows open up without washing out blacks)
        let shadowDeficit = max(0.0, 0.14 - stats.p10Luma)
        let midtoneDeficit = max(0.0, 0.24 - predictedRawLuma)
        var targetShadowLift = 0.08 + shadowDeficit * 0.65 + midtoneDeficit * 0.35 + stats.darkPixelFraction * 0.08
        if predictedRawLuma > 0.26 {
            targetShadowLift = min(targetShadowLift, 0.18)
        }
        targetShadowLift = min(max(targetShadowLift, 0.06), 0.26)

        // Exact match to studioISPFragment's zero-anchored shadowLift curve:
        // min(pow(y, 1 / (1 + lift * 1.45)), y * (1 + lift * 5.5))
        let gamma = 1.0 / (1.0 + targetShadowLift * 1.45)
        let powerY = pow(predictedRawLuma, gamma)
        let maxRatio = 1.0 + targetShadowLift * 5.5
        let postLiftLuma = min(powerY, predictedRawLuma * maxRatio)

        // 2. Brightness Gain (smooth ratio to hit studio target luma without blowing highlights)
        var targetBrightness = (targetLuma / max(postLiftLuma, 0.16)) * 1.02
        if stats.p90Luma > 0.42 {
            targetBrightness = min(targetBrightness, 0.78 / max(stats.p90Luma, 0.42))
        }
        targetBrightness = min(max(targetBrightness, 0.95), 1.32)

        // 3. Contrast Boost (crisp optical punch while keeping lifted shadows open)
        let dynRange = stats.p90Luma - stats.p10Luma
        var targetContrast: Float
        if predictedRawLuma < 0.20 {
            targetContrast = 1.03
        } else if stats.contrastStd255 < 44.0 || dynRange < 0.40 {
            targetContrast = 1.07 + (44.0 - min(stats.contrastStd255, 44.0)) * 0.0022
        } else if dynRange > 0.76 {
            targetContrast = 1.00
        } else {
            targetContrast = 1.06
        }
        targetContrast = min(max(targetContrast, 0.98), 1.15)

        // 4. Saturation Boost (neutral, true-to-life studio color)
        var targetSat: Float
        if predictedRawLuma < 0.20 {
            targetSat = 1.02
        } else if stats.hsvSaturation255 < 68.0 {
            targetSat = 1.06 + (68.0 - stats.hsvSaturation255) * 0.0014
        } else if stats.hsvSaturation255 > 135.0 {
            targetSat = 0.97
        } else {
            targetSat = 1.04
        }
        if stats.faceCount > 0 {
            targetSat = min(targetSat, 1.08)
        }
        targetSat = min(max(targetSat, 0.95), 1.14)

        // 5. Multi-Scale Bilateral + Recursive Temporal 3DNR Denoise
        let lowLightDenoiseBoost: Float = (predictedRawLuma < 0.24) ? (0.24 - predictedRawLuma) * 0.85 : 0.0
        let targetDenoise = min(max(0.28 + (stats.gaussianNoiseStd / 16.0) * 0.36 + lowLightDenoiseBoost, 0.24), 0.68)

        // 6. Luma-Adaptive Unsharp Mask Sharpness (crisp optical clarity beating scrcpy)
        var targetSharpen = 0.48 + (155.0 - min(stats.laplacianSharpness, 155.0)) / 240.0 - (stats.gaussianNoiseStd / 75.0)
        if stats.faceCount > 0 {
            targetSharpen *= 0.92
        }
        targetSharpen = min(max(targetSharpen, 0.38), 0.78)

        // Enable ISP and update sliders + Metal uniforms
        ispCheck.state = .on
        videoView.ispEnabled = true

        brightnessSlider.doubleValue = Double(targetBrightness)
        shadowSlider.doubleValue = Double(targetShadowLift)
        contrastSlider.doubleValue = Double(targetContrast)
        saturationSlider.doubleValue = Double(targetSat)
        denoiseSlider.doubleValue = Double(targetDenoise)
        sharpnessSlider.doubleValue = Double(targetSharpen)
        onGPUGradeChanged()

        let lumaPct = Int((stats.rawMeanLuma * 100.0).rounded())
        let faceBadge = (stats.faceCount > 0) ? " • 👤\(stats.faceCount)" : ""
        let evStr = "\(currentEV >= 0 ? "+" : "")\(currentEV)EV"
        let fullSummary = String(
            format: "✨ %@ • Luma %d%% • %@ • Gain %.2f× • Lift +%d%%%@",
            profileTag,
            lumaPct,
            evStr,
            targetBrightness,
            Int((targetShadowLift * 100.0).rounded()),
            faceBadge
        )
        let compactBanner = String(
            format: "✨ %@ • %@ • %.2f×",
            profileTag,
            evStr,
            targetBrightness
        )
        wandBannerText = compactBanner
        wandBannerUntil = CFAbsoluteTimeGetCurrent() + 3.6
        pillStatusLabel.stringValue = compactBanner
        pillStatusLabel.toolTip = fullSummary
        aiTelemetryLabel.stringValue = String(
            format: "%@ | Sharp %.0f | Noise %.1f | Sat %.0f",
            fullSummary,
            stats.laplacianSharpness,
            stats.gaussianNoiseStd,
            stats.hsvSaturation255
        )
        updateDockButtonStates()
    }

    @objc private func onQuickFlipLens() {
        currentLens = (currentLens == 0) ? 1 : 0
        lensPopup.selectItem(at: currentLens)
        sendCommand("LENS:\(currentLens)")
        updateDockButtonStates()
    }

    @objc private func onLensPopupChanged() {
        currentLens = lensPopup.indexOfSelectedItem
        sendCommand("LENS:\(currentLens)")
        updateDockButtonStates()
    }

    @objc private func onResPopupChanged() {
        currentResIndex = resPopup.indexOfSelectedItem
        switch currentResIndex {
        case 0: sendCommand("RES:3840x2160")
        case 1: sendCommand("RES:2560x1440") // Selects 3264x1836 on Back, 2592x1458 on Front
        case 2: sendCommand("RES:1920x1080")
        default: sendCommand("RES:1280x720")
        }
    }

    @objc private func onFPSModeChanged() {
        currentFPSMode = fpsModePopup.indexOfSelectedItem
        sendCommand("FPSMODE:\(currentFPSMode)")
    }

    @objc private func onQuickCycleFlash() {
        currentFlashMode = (currentFlashMode + 1) % 3
        if currentFlashMode != 0 && currentFlashStrength <= 0 {
            currentFlashStrength = 100
        }
        flashSegment.selectedSegment = currentFlashMode
        updateFlashIconAndSend()
    }

    @objc private func onFlashSegmentChanged() {
        currentFlashMode = flashSegment.selectedSegment
        if currentFlashMode != 0 && currentFlashStrength <= 0 {
            currentFlashStrength = 100
        }
        updateFlashIconAndSend()
    }

    @objc private func onFlashDimmerSliderChanged() {
        let pct = max(0, min(100, Int(flashDimmerSlider.intValue)))
        if pct <= 0 {
            currentFlashMode = 0
            flashSegment?.selectedSegment = 0
            flashDimmerLabel?.stringValue = "OFF"
            videoView.flashDimFactor = 1.0
            updateFlashPillIconOnly()
            _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
            sendCommand("TORCH_STRENGTH:0")
            sendCommand("EV:\(currentEV)")
        } else {
            currentFlashStrength = pct
            if currentLens == 0 && currentFlashMode == 0 {
                currentFlashMode = 1
                flashSegment?.selectedSegment = 1
            }
            flashDimmerLabel?.stringValue = "\(pct)%"
            videoView.flashDimFactor = 0.10 + 0.90 * powf(Float(pct) / 100.0, 1.30)
            updateFlashPillIconOnly()
            _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
            sendCommand("TORCH_STRENGTH:\(pct)")
            let dimEV = max(-12, currentEV - Int(round((1.0 - Float(pct) / 100.0) * 6.0)))
            sendCommand("EV:\(dimEV)")
        }
    }

    private func updateFlashPillIconOnly() {
        let symbol: String
        switch currentFlashMode {
        case 1:
            symbol = "bolt.fill"
        case 2:
            symbol = "bolt.badge.automatic.fill"
        default:
            symbol = "bolt.slash.fill"
        }
        if let img = NSImage(systemSymbolName: symbol, accessibilityDescription: "Flash") {
            let cfg = NSImage.SymbolConfiguration(pointSize: 13, weight: .semibold)
            pillFlashBtn.image = img.withSymbolConfiguration(cfg)
        }
        updateDockButtonStates()
    }

    private func updateFlashIconAndSend() {
        updateFlashPillIconOnly()
        if currentFlashMode == 0 {
            flashDimmerSlider?.intValue = 0
            flashDimmerLabel?.stringValue = "OFF"
            videoView.flashDimFactor = 1.0
            _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
            sendCommand("FLASH:0")
            sendCommand("EV:\(currentEV)")
        } else {
            let pct = max(1, min(100, currentFlashStrength))
            flashDimmerSlider?.intValue = Int32(pct)
            flashDimmerLabel?.stringValue = "\(pct)%"
            videoView.flashDimFactor = 0.10 + 0.90 * powf(Float(pct) / 100.0, 1.30)
            _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
            sendCommand("FLASH:\(currentFlashMode)")
            sendCommand("TORCH_STRENGTH:\(pct)")
            let dimEV = max(-12, currentEV - Int(round((1.0 - Float(pct) / 100.0) * 6.0)))
            sendCommand("EV:\(dimEV)")
        }
    }

    @objc private func onWBChanged() {
        currentWBMode = wbPopup.indexOfSelectedItem
        sendCommand("WB:\(currentWBMode)")
    }

    @objc private func onAFModeChanged() {
        currentAFMode = afModePopup.indexOfSelectedItem
        isAFActive = (currentAFMode != 0)
        sendCommand("AFMODE:\(currentAFMode)")
        updateDockButtonStates()
    }

    @objc private func onTriggerAF() {
        showControlsAndResetTimer()
        isAFActive.toggle()
        sendCommand("AF:TRIGGER")
        wandBannerText = isAFActive ? "🎯 Autofocus Sweep & Lock ON" : "🎯 Continuous Video AF"
        wandBannerUntil = CFAbsoluteTimeGetCurrent() + 2.0
        pillStatusLabel.stringValue = wandBannerText
        updateDockButtonStates()
    }

    @objc private func onAELockChanged() {
        sendCommand("AELOCK:\(aeLockCheck.state == .on ? 1 : 0)")
    }

    @objc private func onAWBLockChanged() {
        sendCommand("AWBLOCK:\(awbLockCheck.state == .on ? 1 : 0)")
    }

    private func syncAllZoomControls(_ z: Float, send: Bool = true) {
        let clamped = max(1.0, min(6.0, z))
        currentZoom = clamped
        let fmt = String(format: "%.1f×", clamped)
        topZoomSlider?.doubleValue = Double(clamped)
        topZoomLabel?.stringValue = fmt
        pillZoomSlider?.doubleValue = Double(clamped)
        pillZoomLabel?.stringValue = fmt
        if send {
            sendCommand(String(format: "ZOOM:%.2f", clamped))
        }
    }

    @objc private func onTopZoomSliderChanged() {
        syncAllZoomControls(Float(topZoomSlider.doubleValue))
    }

    @objc private func onPillZoomSliderChanged() {
        showControlsAndResetTimer()
        syncAllZoomControls(Float(pillZoomSlider.doubleValue))
    }

    @objc private func onEVSliderChanged() {
        currentEV = Int(evSlider.intValue)
        evLabel.stringValue = "\(currentEV >= 0 ? "+" : "")\(currentEV) EV"
        sendCommand("EV:\(currentEV)")
    }

    @objc private func onQualitySliderChanged() {
        let q = Int(qualitySlider.intValue)
        qualityLabel.stringValue = "\(q)%"
        sendCommand("QUALITY:\(q)")
    }

    @objc private func onISPCheckChanged() {
        videoView.ispEnabled = (ispCheck.state == .on)
        _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
    }

    @objc private func onGPUGradeChanged() {
        videoView.brightnessGain = Float(brightnessSlider.doubleValue)
        brightnessLabel.stringValue = String(format: "%.2f×", videoView.brightnessGain)

        videoView.shadowLift = Float(shadowSlider.doubleValue)
        shadowLabel.stringValue = String(format: "+%d%%", Int(videoView.shadowLift * 100))

        videoView.contrastBoost = Float(contrastSlider.doubleValue)
        contrastLabel.stringValue = String(format: "%.2f×", videoView.contrastBoost)

        videoView.saturationBoost = Float(saturationSlider.doubleValue)
        saturationLabel.stringValue = String(format: "%.2f×", videoView.saturationBoost)

        videoView.denoiseStrength = Float(denoiseSlider.doubleValue)
        denoiseLabel.stringValue = String(format: "%.2f", videoView.denoiseStrength)

        videoView.sharpenStrength = Float(sharpnessSlider.doubleValue)
        sharpnessLabel.stringValue = String(format: "%.2f", videoView.sharpenStrength)

        _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
    }

    @objc private func onTransformChanged() {
        videoView.mirrorHorizontal = (mirrorCheck.state == .on)
        videoView.flipVertical = (flipVCheck.state == .on)
        if let autoCheck = autoRotateCheck {
            autoRotateEnabled = (autoCheck.state == .on)
            applyEffectiveRotation()
        }
        _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
    }

    @objc private func onRotate90() {
        manualRotationQuadrants = (manualRotationQuadrants + 1) % 4
        applyEffectiveRotation()
        _ = videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
    }

    private func applyEffectiveRotation() {
        let base = autoRotateEnabled ? sensorRotationQuadrants : 0
        let effective = (base + manualRotationQuadrants) & 0x03
        videoView.rotationQuadrants = effective

        let wantPortrait = (effective % 2 == 1)
        if wantPortrait != isWindowPortrait, let win = videoWindow {
            isWindowPortrait = wantPortrait
            let oldFrame = win.frame
            let oldContent = win.contentRect(forFrameRect: oldFrame)
            let centerX = oldContent.midX
            let centerY = oldContent.midY

            let targetContentSize: NSSize
            if wantPortrait {
                // Portrait 9:16 (540x960)
                targetContentSize = NSSize(width: 540, height: 960)
                win.contentAspectRatio = NSSize(width: 9, height: 16)
            } else {
                // Landscape 16:9 (1152x648)
                targetContentSize = NSSize(width: 1152, height: 648)
                win.contentAspectRatio = NSSize(width: 16, height: 9)
            }

            var newOriginX = centerX - targetContentSize.width / 2.0
            var newOriginY = centerY - targetContentSize.height / 2.0
            if let screenFrame = (win.screen ?? NSScreen.main)?.visibleFrame {
                newOriginX = max(screenFrame.minX + 12, min(newOriginX, screenFrame.maxX - targetContentSize.width - 12))
                newOriginY = max(screenFrame.minY + 12, min(newOriginY, screenFrame.maxY - targetContentSize.height - 40))
            }

            let newContentRect = NSRect(x: newOriginX, y: newOriginY, width: targetContentSize.width, height: targetContentSize.height)
            let newFrameRect = win.frameRect(forContentRect: newContentRect)
            win.setFrame(newFrameRect, display: true, animate: true)
            repositionOverlayPill()
        }
    }

    @objc private func onResetGrade() {
        currentPreset = 1
        ispCheck.state = .on
        videoView.ispEnabled = true
        videoView.warmthShift = 0.0
        brightnessSlider.doubleValue = 1.12
        shadowSlider.doubleValue = 0.16
        contrastSlider.doubleValue = 1.05
        saturationSlider.doubleValue = 1.04
        denoiseSlider.doubleValue = 0.38
        sharpnessSlider.doubleValue = 0.52
        onGPUGradeChanged()
        updateDockButtonStates()
    }


    @objc private func onResetAllDefaults() {
        isWandModeActive = false
        isAFActive = false
        currentPreset = 1
        // 1. Reset Camera2 Hardware Controls (Shutter/FPS, Flash, WB, AF, Locks, Zoom, EV, Quality)
        if currentResIndex != 2 {
            currentResIndex = 2
            resPopup.selectItem(at: 2)
            sendCommand("RES:1920x1080")
        }

        currentFPSMode = 0
        fpsModePopup.selectItem(at: 0)
        sendCommand("FPSMODE:0")

        currentFlashMode = 0
        currentFlashStrength = 100
        flashSegment.selectedSegment = 0
        updateFlashIconAndSend()

        currentWBMode = 0
        wbPopup.selectItem(at: 0)
        sendCommand("WB:0")

        currentAFMode = 0
        afModePopup.selectItem(at: 0)
        sendCommand("AFMODE:0")

        aeLockCheck.state = .off
        sendCommand("AELOCK:0")

        awbLockCheck.state = .off
        sendCommand("AWBLOCK:0")

        syncAllZoomControls(1.0)

        currentEV = 0
        evSlider.intValue = 0
        evLabel.stringValue = "+0 EV"
        sendCommand("EV:0")

        qualitySlider.intValue = 92
        qualityLabel.stringValue = "92%"
        sendCommand("QUALITY:92")

        sendCommand("AF:TRIGGER")

        // 2. Reset Metal Studio ISP Sliders & Denoise/Sharpen
        onResetGrade()

        // 3. Reset Orientation & Transform Controls
        autoRotateEnabled = true
        autoRotateCheck.state = .on
        mirrorCheck.state = .off
        flipVCheck.state = .off
        videoView.mirrorHorizontal = false
        videoView.flipVertical = false
        manualRotationQuadrants = 0
        applyEffectiveRotation()

        // 4. Update HUD Banner & Telemetry
        let msg = "↺ Defaults • 1080p • 0EV • 1.0×"
        wandBannerText = msg
        wandBannerUntil = CFAbsoluteTimeGetCurrent() + 3.0
        pillStatusLabel.stringValue = msg
        aiTelemetryLabel.stringValue = "↺ All Camera2 Hardware & Metal Studio ISP settings restored to defaults"
        updateDockButtonStates()
        repositionOverlayPill()
    }

    @objc private func onScreenOffChanged() {
        let off = (screenOffCheck.state == .on)
        sendCommand("SCREEN:\(off ? 0 : 1)")
    }

    /// Generates a guaranteed collision-free file URL inside `~/Downloads/` using
    /// millisecond precision (`yyyy-MM-dd_HH-mm-ss-SSS`) plus an atomic `_1`, `_2`, ... suffix check.
    private func makeCollisionFreeDownloadsURL(prefix: String, width: Int, height: Int, ext: String) -> URL {
        let fm = FileManager.default
        let downloadsDir = fm.urls(for: .downloadsDirectory, in: .userDomainMask).first
            ?? fm.homeDirectoryForCurrentUser.appendingPathComponent("Downloads", isDirectory: true)
        try? fm.createDirectory(at: downloadsDir, withIntermediateDirectories: true)

        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss-SSS"
        let timestamp = formatter.string(from: Date())
        let baseName = "\(prefix)_\(width)x\(height)_\(timestamp)"

        var candidate = downloadsDir.appendingPathComponent("\(baseName).\(ext)")
        var suffix = 1
        while fm.fileExists(atPath: candidate.path) {
            candidate = downloadsDir.appendingPathComponent("\(baseName)_\(suffix).\(ext)")
            suffix += 1
        }
        return candidate
    }

    @objc private func onTakeSnapshot() {
        showControlsAndResetTimer()
        let snap: (data: Data, width: Int, height: Int)
        if let liveSnap = videoView.latestSnapshotJPEG() {
            snap = liveSnap
        } else if let rep = videoView.bitmapImageRepForCachingDisplay(in: videoView.bounds) {
            videoView.cacheDisplay(in: videoView.bounds, to: rep)
            guard let jpegData = rep.representation(using: .jpeg, properties: [.compressionFactor: 0.94]) else { return }
            snap = (data: jpegData, width: max(rep.pixelsWide, Int(videoView.bounds.width)), height: max(rep.pixelsHigh, Int(videoView.bounds.height)))
        } else {
            return
        }
        let fileURL = makeCollisionFreeDownloadsURL(
            prefix: "PxlCam",
            width: snap.width,
            height: snap.height,
            ext: "jpg"
        )
        do {
            try snap.data.write(to: fileURL, options: .atomic)
            NSSound.beep()
            let now = CFAbsoluteTimeGetCurrent()
            snapActiveUntil = now + 2.2
            wandBannerText = "📸 Saved JPEG → ~/Downloads"
            wandBannerUntil = now + 2.6
            pillStatusLabel.stringValue = wandBannerText
            pillStatusLabel.toolTip = "📸 Saved → ~/Downloads/\(fileURL.lastPathComponent)"
            aiTelemetryLabel.stringValue = "📸 Snapshot saved: ~/Downloads/\(fileURL.lastPathComponent)"
            updateDockButtonStates()
            repositionOverlayPill()
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.25) { [weak self] in
                self?.updateDockButtonStates()
            }
        } catch {
            wandBannerText = "⚠️ Snapshot failed"
            wandBannerUntil = CFAbsoluteTimeGetCurrent() + 3.0
        }
    }

    private func updateRecordingButtonsUI(recording: Bool) {
        let symbolName = recording ? "stop.circle.fill" : "record.circle"
        if let img = NSImage(systemSymbolName: symbolName, accessibilityDescription: "Record Video") {
            let cfg = NSImage.SymbolConfiguration(pointSize: 13, weight: .semibold)
            pillRecordBtn.image = img.withSymbolConfiguration(cfg)
        }
        recordBtn.title = recording ? "⏹ Stop REC [V]" : "⏺ Record MP4 [V]"
        recordBtn.contentTintColor = recording ? .systemOrange : .systemRed
        updateDockButtonStates()
    }

    @objc private func onToggleRecording() {
        if videoView.isRecording {
            videoView.stopRecording { [weak self] savedURL, duration, frames in
                guard let self = self else { return }
                self.updateRecordingButtonsUI(recording: false)
                NSSound.beep()
                if let url = savedURL {
                    let durStr = String(format: "%.1fs", duration)
                    self.wandBannerText = "🎬 Saved MP4 (\(durStr)) → Downloads"
                    self.wandBannerUntil = CFAbsoluteTimeGetCurrent() + 3.5
                    self.pillStatusLabel.stringValue = self.wandBannerText
                    self.pillStatusLabel.toolTip = "🎬 Saved (\(durStr)) → ~/Downloads/\(url.lastPathComponent)"
                    self.aiTelemetryLabel.stringValue = "🎬 MP4 saved (\(frames)f, \(durStr)): ~/Downloads/\(url.lastPathComponent)"
                } else {
                    self.wandBannerText = "⚠️ Recording failed"
                    self.wandBannerUntil = CFAbsoluteTimeGetCurrent() + 3.0
                }
                self.repositionOverlayPill()
            }
        } else {
            let rawW = max(2, videoView.frameWidth)
            let rawH = max(2, videoView.frameHeight)
            let outW = isWindowPortrait ? rawH : rawW
            let outH = isWindowPortrait ? rawW : rawH
            let targetURL = makeCollisionFreeDownloadsURL(
                prefix: "PxlCam_Rec",
                width: outW,
                height: outH,
                ext: "mp4"
            )
            let res = videoView.startRecording(to: targetURL)
            if res.started {
                updateRecordingButtonsUI(recording: true)
                NSSound.beep()
                wandBannerText = "🔴 REC MP4 • \(res.width)×\(res.height)"
                wandBannerUntil = CFAbsoluteTimeGetCurrent() + 2.2
                pillStatusLabel.stringValue = wandBannerText
                pillStatusLabel.toolTip = "🔴 Recording MP4 to ~/Downloads/\(targetURL.lastPathComponent)"
                aiTelemetryLabel.stringValue = "🔴 Recording MP4 to ~/Downloads/\(targetURL.lastPathComponent)"
                repositionOverlayPill()
            } else {
                wandBannerText = "⚠️ Recorder error"
                wandBannerUntil = CFAbsoluteTimeGetCurrent() + 3.0
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        if videoView != nil && videoView.isRecording {
            let sem = DispatchSemaphore(value: 0)
            videoView.stopRecording { _, _, _ in
                sem.signal()
            }
            _ = sem.wait(timeout: .now() + 2.0)
        }
    }

    private func handleKeyCommand(_ event: NSEvent) -> Bool {
        guard let chars = event.charactersIgnoringModifiers?.lowercased() else { return false }
        switch chars {
        case "w":
            onPillWandButtonClicked()
            return true
        case "d", "0":
            onResetAllDefaults()
            return true
        case ",":
            onToggleSettingsWindow()
            return true
        case "1":
            resPopup.selectItem(at: 0)
            onResPopupChanged()
            return true
        case "2":
            resPopup.selectItem(at: 1)
            onResPopupChanged()
            return true
        case "3":
            resPopup.selectItem(at: 2)
            onResPopupChanged()
            return true
        case "f", "c":
            onQuickFlipLens()
            return true
        case "t":
            onQuickCycleFlash()
            return true
        case "a":
            onTriggerAF()
            return true
        case "p", "s":
            onTakeSnapshot()
            return true
        case "v":
            onToggleRecording()
            return true
        case "m":
            mirrorCheck.state = (mirrorCheck.state == .on) ? .off : .on
            onTransformChanged()
            return true
        case "r":
            onRotate90()
            return true
        case "+", "=":
            syncAllZoomControls(min(currentZoom + 0.25, 6.0))
            return true
        case "-", "_":
            syncAllZoomControls(max(currentZoom - 0.25, 1.0))
            return true
        case "]":
            evSlider.intValue = min(evSlider.intValue + 1, 12)
            onEVSliderChanged()
            return true
        case "[":
            evSlider.intValue = max(evSlider.intValue - 1, -12)
            onEVSliderChanged()
            return true
        case "h":
            hudVisible.toggle()
            if hudVisible {
                videoWindow.addChildWindow(overlayPillWindow, ordered: .above)
                overlayPillWindow.orderFront(nil)
                if let topLeftBrandWindow = topLeftBrandWindow {
                    videoWindow.addChildWindow(topLeftBrandWindow, ordered: .above)
                    topLeftBrandWindow.orderFront(nil)
                }
                repositionOverlayPill()
            } else {
                videoWindow.removeChildWindow(overlayPillWindow)
                overlayPillWindow.orderOut(nil)
                if let topLeftBrandWindow = topLeftBrandWindow {
                    videoWindow.removeChildWindow(topLeftBrandWindow)
                    topLeftBrandWindow.orderOut(nil)
                }
            }
            return true
        default:
            return false
        }
    }

    private func sendCommand(_ cmd: String) {
        commandQueue.async { [weak self] in
            guard let self = self else { return }
            let sock = self.activeClientSocket
            guard sock >= 0 else { return }
            let line = cmd + "\n"
            if let data = line.data(using: .utf8) {
                data.withUnsafeBytes { rawBuf in
                    guard let base = rawBuf.baseAddress else { return }
                    _ = Darwin.send(sock, base, data.count, 0)
                }
            }
        }
    }

    // MARK: - Telemetry Timer
    private func startTelemetryTimer() {
        Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let now = CFAbsoluteTimeGetCurrent()
            let elapsed = now - self.lastFPSCheck
            if elapsed >= 0.5 {
                self.measuredFPS = Double(self.frameCounter) / elapsed
                self.frameCounter = 0
                self.lastFPSCheck = now
            }
            let rawW = self.videoView.frameWidth
            let rawH = self.videoView.frameHeight
            let dispW = self.isWindowPortrait ? rawH : rawW
            let dispH = self.isWindowPortrait ? rawW : rawH
            let lensName = (self.currentLens == 0) ? "BACK" : "FRONT"
            if self.isConnected {
                if self.videoView.isRecording {
                    self.pillStatusDot.textColor = NSColor.systemRed
                } else {
                    self.pillStatusDot.textColor = NSColor.systemGreen
                }
                if now < self.wandBannerUntil {
                    self.pillStatusLabel.stringValue = self.wandBannerText
                } else if self.videoView.isRecording {
                    let recSec = Int(self.videoView.recordingDurationSeconds.rounded(.down))
                    let mm = recSec / 60
                    let ss = recSec % 60
                    self.pillStatusLabel.stringValue = String(
                        format: "🔴 REC %02d:%02d • %dx%d",
                        mm, ss, dispW, dispH
                    )
                } else {
                    self.pillStatusLabel.stringValue = String(format: "%dx%d • %.0f fps • %@", dispW, dispH, self.measuredFPS, lensName)
                }
            } else {
                self.pillStatusDot.textColor = NSColor.systemOrange
                self.pillStatusLabel.stringValue = "Searching USB / Wi-Fi..."
            }
            self.updateDockButtonStates()
            if let videoWindow = self.videoWindow, let overlayPillWindow = self.overlayPillWindow {
                let contentRect = videoWindow.contentRect(forFrameRect: videoWindow.frame)
                let expectedWidth = min(668, max(480, contentRect.width - 20))
                let expectedX = round(contentRect.origin.x + (contentRect.width - expectedWidth) / 2.0)
                if abs(overlayPillWindow.frame.width - expectedWidth) > 0.5 || abs(overlayPillWindow.frame.origin.x - expectedX) > 0.5 {
                    self.repositionOverlayPill()
                }
            }
        }
    }

    // MARK: - Zero-Stall TCP Receiver Loop & Standalone Dock ADB Auto-Starter
    private func ensureAndroidServerIfNeeded() {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastAutoServerLaunchAttempt >= 6.0 else { return }
        lastAutoServerLaunchAttempt = now

        let helperCandidates = [
            "/Users/aaryn/Projects/pxl-cam/ensure_android_server.sh"
        ]
        guard let helperScript = helperCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else { return }

        DispatchQueue.global(qos: .utility).async {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/bin/bash")
            proc.arguments = [helperScript]
            proc.standardOutput = FileHandle.nullDevice
            proc.standardError = FileHandle.nullDevice
            try? proc.run()
        }
    }

    private func startStreamReceiverThread() {
        Thread.detachNewThread { [weak self] in
            guard let self = self else { return }
            var failedAttempts = 0
            while true {
                let connected = self.connectAndReceiveLoop()
                if connected {
                    failedAttempts = 0
                } else {
                    failedAttempts += 1
                    if failedAttempts >= 1 {
                        self.ensureAndroidServerIfNeeded()
                    }
                    if failedAttempts % 3 == 0 {
                        if let discovered = self.scanLANForPxlCamDevice(port: 9876) {
                            UserDefaults.standard.set(discovered, forKey: "PxlCamLastDeviceIP")
                        }
                    }
                }
                Thread.sleep(forTimeInterval: 0.25)
            }
        }
    }

    private func envWirelessHost() -> String? {
        let envPath = "/Users/aaryn/Projects/pxl-cam/.env"
        guard let content = try? String(contentsOfFile: envPath, encoding: .utf8) else { return nil }
        for line in content.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("ADB_WIRELESS_HOST=") {
                let val = String(trimmed.dropFirst("ADB_WIRELESS_HOST=".count))
                    .trimmingCharacters(in: CharacterSet(charactersIn: "\"' "))
                if !val.isEmpty {
                    return val
                }
            }
        }
        return nil
    }

    private func candidateDeviceHosts() -> [String] {
        var hosts: [String] = ["127.0.0.1"]
        if let saved = UserDefaults.standard.string(forKey: "PxlCamLastDeviceIP"), !saved.isEmpty {
            hosts.append(saved)
        }
        if let envHost = envWirelessHost(), !envHost.isEmpty {
            hosts.append(envHost)
        }
        hosts.append("192.168.1.82")
        var seen = Set<String>()
        return hosts.filter { seen.insert($0).inserted }
    }

    private func connectSocketWithTimeout(host: String, port: UInt16, timeoutMs: Int32) -> Int32 {
        let ipAddr = inet_addr(host)
        guard ipAddr != INADDR_NONE else { return -1 }

        let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard sock >= 0 else { return -1 }

        var noSigPipe: Int32 = 1
        setsockopt(sock, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))

        let flags = fcntl(sock, F_GETFL, 0)
        _ = fcntl(sock, F_SETFL, flags | O_NONBLOCK)

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = ipAddr

        let res = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }

        if res < 0 && errno != EINPROGRESS {
            Darwin.close(sock)
            return -1
        }

        if res != 0 {
            var pfd = pollfd(fd: sock, events: Int16(POLLOUT), revents: 0)
            let pollRes = Darwin.poll(&pfd, 1, timeoutMs)
            if pollRes <= 0 || (pfd.revents & Int16(POLLERR | POLLHUP)) != 0 {
                Darwin.close(sock)
                return -1
            }
            var soError: Int32 = 0
            var len = socklen_t(MemoryLayout<Int32>.size)
            if getsockopt(sock, SOL_SOCKET, SO_ERROR, &soError, &len) < 0 || soError != 0 {
                Darwin.close(sock)
                return -1
            }
        }

        _ = fcntl(sock, F_SETFL, flags)
        var one: Int32 = 1
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, socklen_t(MemoryLayout<Int32>.size))
        var tv = timeval(tv_sec: 2, tv_usec: 0)
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        return sock
    }

    private func scanLANForPxlCamDevice(port: UInt16 = 9876) -> String? {
        var prefixes: [String] = []
        var ifaddrPtr: UnsafeMutablePointer<ifaddrs>?
        if getifaddrs(&ifaddrPtr) == 0, let firstAddr = ifaddrPtr {
            defer { freeifaddrs(ifaddrPtr) }
            var ptr: UnsafeMutablePointer<ifaddrs>? = firstAddr
            while let cur = ptr {
                let flags = Int32(cur.pointee.ifa_flags)
                if (flags & IFF_UP) != 0 && (flags & IFF_LOOPBACK) == 0,
                   let addr = cur.pointee.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) {
                    var hostBuf = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                    if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &hostBuf, socklen_t(hostBuf.count), nil, 0, NI_NUMERICHOST) == 0 {
                        let ip = String(cString: hostBuf)
                        let parts = ip.split(separator: ".")
                        if parts.count == 4 && (parts[0] == "192" || parts[0] == "10" || parts[0] == "172") {
                            let prefix = "\(parts[0]).\(parts[1]).\(parts[2])."
                            if !prefixes.contains(prefix) {
                                prefixes.append(prefix)
                            }
                        }
                    }
                }
                ptr = cur.pointee.ifa_next
            }
        }
        if prefixes.isEmpty {
            prefixes = ["192.168.1."]
        }

        let lock = NSLock()
        var foundHost: String? = nil
        let sem = DispatchSemaphore(value: 32)

        for prefix in prefixes {
            DispatchQueue.concurrentPerform(iterations: 254) { idx in
                lock.lock()
                let alreadyFound = (foundHost != nil)
                lock.unlock()
                if alreadyFound { return }

                sem.wait()
                defer { sem.signal() }

                let candidate = "\(prefix)\(idx + 1)"
                let s = self.connectSocketWithTimeout(host: candidate, port: port, timeoutMs: 160)
                if s >= 0 {
                    Darwin.close(s)
                    lock.lock()
                    if foundHost == nil {
                        foundHost = candidate
                    }
                    lock.unlock()
                }
            }
            if foundHost != nil { break }
        }
        return foundHost
    }

    private func readExact(from sock: Int32, into buffer: UnsafeMutablePointer<UInt8>, length: Int) -> Bool {
        var totalRead = 0
        while totalRead < length {
            let n = Darwin.recv(sock, buffer.advanced(by: totalRead), length - totalRead, 0)
            if n <= 0 {
                return false
            }
            totalRead += n
        }
        return true
    }

    private func connectAndReceiveLoop() -> Bool {
        for host in candidateDeviceHosts() {
            let timeoutMs: Int32 = (host == "127.0.0.1") ? 280 : 500
            let sock = connectSocketWithTimeout(host: host, port: 9876, timeoutMs: timeoutMs)
            if sock < 0 { continue }

            if receiveStreamFromSocket(sock, host: host) {
                if host != "127.0.0.1" {
                    UserDefaults.standard.set(host, forKey: "PxlCamLastDeviceIP")
                }
                return true
            }
        }
        return false
    }

    private func receiveStreamFromSocket(_ sock: Int32, host: String) -> Bool {
        commandQueue.sync {
            self.activeClientSocket = sock
        }

        var receivedAnyFrame = false
        let headerBuf = UnsafeMutablePointer<UInt8>.allocate(capacity: 26)
        defer {
            headerBuf.deallocate()
            commandQueue.sync {
                self.activeClientSocket = -1
            }
            Darwin.close(sock)
            self.isConnected = false
        }

        while true {
            guard readExact(from: sock, into: headerBuf, length: 26) else {
                break
            }
            let magic = (UInt32(headerBuf[0]) << 24) | (UInt32(headerBuf[1]) << 16) | (UInt32(headerBuf[2]) << 8) | UInt32(headerBuf[3])
            if magic != 0x50584C43 { // 'PXLC'
                break
            }
            let codec = Int(headerBuf[4]) // Byte 4: 2 = Hardware Video ISP H.264, 1 = JPEG
            let packedCamAndRot = Int(headerBuf[5]) // Byte 5: bits 0..1 = camera, bits 2..3 = autoRotationQuadrants
            let phoneCam = packedCamAndRot & 0x03
            let rawPhoneRot = (packedCamAndRot >> 2) & 0x03
            let width = Int((UInt16(headerBuf[6]) << 8) | UInt16(headerBuf[7]))
            let height = Int((UInt16(headerBuf[8]) << 8) | UInt16(headerBuf[9]))
            let seqOrBitrate = Int((UInt32(headerBuf[10]) << 24) | (UInt32(headerBuf[11]) << 16) | (UInt32(headerBuf[12]) << 8) | UInt32(headerBuf[13]))
            let payloadLen = Int((UInt32(headerBuf[22]) << 24) | (UInt32(headerBuf[23]) << 16) | (UInt32(headerBuf[24]) << 8) | UInt32(headerBuf[25]))

            // PxlCamEngine.kt in com.pxlcam.app sends bitrate 12_000_000 in bytes 10..13 and static sensor mount angle in rawPhoneRot;
            // normalize static sensor mount angle to 0 (16:9 landscape) so it matches the 1920x1080 stream unless manually rotated.
            let phoneRot = (seqOrBitrate == 12_000_000) ? 0 : rawPhoneRot

            if payloadLen <= 0 || payloadLen > 16_000_000 {
                break
            }

            var frameData = Data(count: payloadLen)
            let ok = frameData.withUnsafeMutableBytes { rawBuf -> Bool in
                guard let base = rawBuf.baseAddress?.assumingMemoryBound(to: UInt8.self) else { return false }
                return self.readExact(from: sock, into: base, length: payloadLen)
            }
            if !ok {
                break
            }

            if phoneCam != self.currentLens {
                self.currentLens = phoneCam
                DispatchQueue.main.async { [weak self] in
                    guard let self = self else { return }
                    self.lensPopup?.selectItem(at: phoneCam)
                    self.updateDockButtonStates()
                }
            }

            if phoneRot != self.sensorRotationQuadrants {
                self.sensorRotationQuadrants = phoneRot
                DispatchQueue.main.async { [weak self] in
                    self?.applyEffectiveRotation()
                }
            }

            receivedAnyFrame = true
            self.isConnected = true
            self.lastFrameBytes = payloadLen
            self.frameCounter += 1
            if codec == 2 {
                self.videoView.updateWithH264Data(frameData, width: width, height: height)
            } else {
                self.videoView.updateWithJPEGData(frameData, width: width, height: height)
            }
        }
        return receivedAnyFrame
    }

    // MARK: - Local OBS Custom Dock & 1080p Direct Stream Server (127.0.0.1:9877)

    private func startLocalOBSBridgeServer() {
        DispatchQueue.global(qos: .userInteractive).async { [weak self] in
            let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
            guard sock >= 0 else { return }
            var yes: Int32 = 1
            setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))

            var addr = sockaddr_in()
            addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
            addr.sin_family = sa_family_t(AF_INET)
            addr.sin_port = UInt16(9877).bigEndian
            addr.sin_addr.s_addr = inet_addr("127.0.0.1")

            let bindRes = withUnsafePointer(to: &addr) {
                $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                    Darwin.bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
                }
            }
            guard bindRes == 0 else {
                Darwin.close(sock)
                return
            }
            guard Darwin.listen(sock, 32) == 0 else {
                Darwin.close(sock)
                return
            }

            while true {
                var clientAddr = sockaddr_in()
                var clientLen = socklen_t(MemoryLayout<sockaddr_in>.size)
                let client = withUnsafeMutablePointer(to: &clientAddr) {
                    $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                        Darwin.accept(sock, $0, &clientLen)
                    }
                }
                if client < 0 { continue }
                DispatchQueue.global(qos: .userInteractive).async { [weak self] in
                    self?.handleOBSBridgeClient(client)
                }
            }
        }
    }

    private func handleOBSBridgeClient(_ client: Int32) {
        var noSigPipe: Int32 = 1
        setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &noSigPipe, socklen_t(MemoryLayout<Int32>.size))

        var buf = [UInt8](repeating: 0, count: 4096)
        let n = Darwin.read(client, &buf, buf.count - 1)
        guard n > 0 else {
            Darwin.close(client)
            return
        }
        let req = String(decoding: buf[0..<n], as: UTF8.self)
        let firstLine = req.split(separator: "\r\n", maxSplits: 1).first.map(String.init) ?? ""
        let parts = firstLine.split(separator: " ")
        guard parts.count >= 2 else {
            Darwin.close(client)
            return
        }
        let fullPath = String(parts[1])
        let pathAndQuery = fullPath.split(separator: "?", maxSplits: 1, omittingEmptySubsequences: false)
        let path = String(pathAndQuery[0])
        let query = pathAndQuery.count > 1 ? String(pathAndQuery[1]) : ""

        if path == "/stream.mjpg" {
            let header =
                "HTTP/1.1 200 OK\r\n" +
                "Content-Type: multipart/x-mixed-replace; boundary=pxlcamframe\r\n" +
                "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n" +
                "Pragma: no-cache\r\n" +
                "Access-Control-Allow-Origin: *\r\n" +
                "Connection: close\r\n\r\n"
            _ = header.withCString { Darwin.send(client, $0, strlen($0), 0) }

            var lastGen: UInt64 = 0
            while true {
                let currentGen = self.videoView.processedFrameID
                if currentGen != lastGen, let jpeg = self.videoView.latestStreamJPEGData(quality: 0.92) {
                    lastGen = currentGen
                    let partHeader =
                        "--pxlcamframe\r\n" +
                        "Content-Type: image/jpeg\r\n" +
                        "Content-Length: \(jpeg.count)\r\n\r\n"
                    let hSent = partHeader.withCString { Darwin.send(client, $0, strlen($0), 0) }
                    if hSent <= 0 { break }
                    let bSent = jpeg.withUnsafeBytes { rawBuf -> Int in
                        guard let base = rawBuf.baseAddress else { return -1 }
                        return Darwin.send(client, base, jpeg.count, 0)
                    }
                    if bSent <= 0 { break }
                    let tailSent = "\r\n".withCString { Darwin.send(client, $0, 2, 0) }
                    if tailSent <= 0 { break }
                } else {
                    usleep(6_000)
                }
            }
            Darwin.close(client)
            return
        }

        if path == "/frame.jpg" {
            if let jpeg = self.videoView.latestStreamJPEGData(quality: 0.94) {
                sendHTTPResponse(client: client, status: "200 OK", contentType: "image/jpeg", body: jpeg)
            } else {
                sendHTTPResponse(client: client, status: "503 Service Unavailable", contentType: "text/plain", body: Data("No frame".utf8))
            }
            Darwin.close(client)
            return
        }

        if path == "/logo.png" {
            let candidates = [
                "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_logo_rounded.png",
                Bundle.main.resourcePath.map { "\($0)/AppIcon.icns" } ?? ""
            ]
            for p in candidates where !p.isEmpty {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: p)) {
                    sendHTTPResponse(client: client, status: "200 OK", contentType: "image/png", body: data)
                    Darwin.close(client)
                    return
                }
            }
            sendHTTPResponse(client: client, status: "404 Not Found", contentType: "text/plain", body: Data())
            Darwin.close(client)
            return
        }

        if path == "/wordmark.png" {
            let candidates = [
                "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_wordmark_bttf.png",
                Bundle.main.resourcePath.map { "\($0)/pxl_cam_wordmark_bttf.png" } ?? ""
            ]
            for p in candidates where !p.isEmpty {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: p)) {
                    sendHTTPResponse(client: client, status: "200 OK", contentType: "image/png", body: data)
                    Darwin.close(client)
                    return
                }
            }
            sendHTTPResponse(client: client, status: "404 Not Found", contentType: "text/plain", body: Data())
            Darwin.close(client)
            return
        }

        if path == "/overlay-brand.png" {
            let candidates = [
                "/Users/aaryn/Projects/pxl-cam/assets/pxl_cam_top_left_overlay.png",
                Bundle.main.resourcePath.map { "\($0)/pxl_cam_top_left_overlay.png" } ?? ""
            ]
            for p in candidates where !p.isEmpty {
                if let data = try? Data(contentsOf: URL(fileURLWithPath: p)) {
                    sendHTTPResponse(client: client, status: "200 OK", contentType: "image/png", body: data)
                    Darwin.close(client)
                    return
                }
            }
            sendHTTPResponse(client: client, status: "404 Not Found", contentType: "text/plain", body: Data())
            Darwin.close(client)
            return
        }

        if path == "/api/state" {
            let json = buildOBSBridgeStateJSON()
            sendHTTPResponse(client: client, status: "200 OK", contentType: "application/json", body: Data(json.utf8))
            Darwin.close(client)
            return
        }

        if path == "/api/control" {
            applyOBSBridgeControlQuery(query)
            let json = buildOBSBridgeStateJSON()
            sendHTTPResponse(client: client, status: "200 OK", contentType: "application/json", body: Data(json.utf8))
            Darwin.close(client)
            return
        }

        if path == "/player" {
            let html = Self.obsDirectPlayerHTML
            sendHTTPResponse(client: client, status: "200 OK", contentType: "text/html; charset=utf-8", body: Data(html.utf8))
            Darwin.close(client)
            return
        }

        let html = Self.obsCustomDockHTML
        sendHTTPResponse(client: client, status: "200 OK", contentType: "text/html; charset=utf-8", body: Data(html.utf8))
        Darwin.close(client)
    }

    private func sendHTTPResponse(client: Int32, status: String, contentType: String, body: Data) {
        let header =
            "HTTP/1.1 \(status)\r\n" +
            "Content-Type: \(contentType)\r\n" +
            "Content-Length: \(body.count)\r\n" +
            "Cache-Control: no-store, no-cache, must-revalidate, max-age=0\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Connection: close\r\n\r\n"
        _ = header.withCString { Darwin.send(client, $0, strlen($0), 0) }
        if !body.isEmpty {
            body.withUnsafeBytes { rawBuf in
                if let base = rawBuf.baseAddress {
                    _ = Darwin.send(client, base, body.count, 0)
                }
            }
        }
    }

    private var obsPresetIndex: Int {
        return currentPreset
    }

    private var isAtDefaultSettings: Bool {
        return !isWandModeActive &&
            currentPreset == 1 &&
            currentFlashMode == 0 &&
            abs(currentZoom - 1.0) < 0.02 &&
            currentEV == 0 &&
            currentWBMode == 0 &&
            manualRotationQuadrants == 0 &&
            !videoView.mirrorHorizontal &&
            !videoView.flipVertical &&
            abs(videoView.warmthShift) < 0.01
    }

    private func buildOBSBridgeStateJSON() -> String {
        let onMainBlock: () -> String = {
            let wandActive = self.isWandModeActive || (CFAbsoluteTimeGetCurrent() < self.wandBannerUntil && self.wandBannerText.hasPrefix("✨"))
            let effectiveFlashLevel = (self.currentFlashMode == 0) ? 0 : self.currentFlashStrength
            return String(
                format: """
                {"connected":%@,"fps":%.1f,"frameW":%d,"frameH":%d,"camera":%d,"flash":%@,"flashLevel":%d,"ev":%d,"zoom":%.2f,"wb":%d,"af":%d,"preset":%d,"wand":%@,"isDefault":%@,"rotation":%d,"mirrorH":%@,"mirrorV":%@,"denoise":%.3f,"sharpness":%.3f,"brightness":%.3f,"contrast":%.3f,"saturation":%.3f,"warmth":%.3f,"tint":0.000,"shadowLift":%.3f,"highlightRoll":0.150,"skinProtect":0.500,"blackPoint":0.008}
                """,
                self.isConnected ? "true" : "false",
                self.measuredFPS,
                self.videoView.frameWidth,
                self.videoView.frameHeight,
                self.currentLens,
                (self.currentFlashMode != 0) ? "true" : "false",
                effectiveFlashLevel,
                self.currentEV,
                self.currentZoom,
                self.currentWBMode,
                self.currentAFMode,
                self.obsPresetIndex,
                wandActive ? "true" : "false",
                self.isAtDefaultSettings ? "true" : "false",
                self.manualRotationQuadrants,
                self.videoView.mirrorHorizontal ? "true" : "false",
                self.videoView.flipVertical ? "true" : "false",
                self.videoView.denoiseStrength,
                self.videoView.sharpenStrength,
                self.videoView.brightnessGain,
                self.videoView.contrastBoost,
                self.videoView.saturationBoost,
                self.videoView.warmthShift,
                self.videoView.shadowLift
            )
        }
        if Thread.isMainThread {
            return onMainBlock()
        } else {
            return DispatchQueue.main.sync(execute: onMainBlock)
        }
    }

    private func applyOBSBridgeControlQuery(_ query: String) {
        var params: [String: String] = [:]
        for pair in query.split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            if kv.count == 2 {
                let k = String(kv[0]).removingPercentEncoding ?? String(kv[0])
                let v = String(kv[1]).removingPercentEncoding ?? String(kv[1])
                params[k] = v
            }
        }
        let applyBlock: () -> Void = { [weak self] in
            guard let self = self else { return }
            if let lensStr = params["lens"], let lens = Int(lensStr) {
                self.currentLens = (lens == 1) ? 1 : 0
                if self.currentLens == 1 && self.currentFlashMode != 0 {
                    self.currentFlashMode = 0
                    self.flashSegment?.selectedSegment = 0
                    self.updateFlashIconAndSend()
                }
                self.lensPopup?.selectItem(at: self.currentLens)
                self.sendCommand("LENS:\(self.currentLens)")
            }
            if let flashStr = params["flash"] {
                if flashStr == "toggle" {
                    self.currentFlashMode = (self.currentFlashMode == 0) ? 1 : 0
                } else {
                    let want = (flashStr == "1" || flashStr == "true")
                    self.currentFlashMode = (self.currentLens == 0 && want) ? 1 : 0
                }
                if self.currentFlashMode != 0 && self.currentFlashStrength <= 0 {
                    self.currentFlashStrength = 100
                }
                self.flashSegment?.selectedSegment = self.currentFlashMode
                self.updateFlashIconAndSend()
            }
            if let flStr = params["flashLevel"] ?? params["torchStrength"], let fl = Int(flStr) {
                let clamped = max(0, min(100, fl))
                self.flashDimmerSlider?.intValue = Int32(clamped)
                self.onFlashDimmerSliderChanged()
            }
            if let evStr = params["ev"], let ev = Int(evStr) {
                self.currentEV = max(-12, min(12, ev))
                self.evSlider?.intValue = Int32(self.currentEV)
                self.evLabel?.stringValue = "\(self.currentEV >= 0 ? "+" : "")\(self.currentEV) EV"
                self.sendCommand("EV:\(self.currentEV)")
            }
            if let zoomStr = params["zoom"], let z = Float(zoomStr) {
                self.syncAllZoomControls(max(1.0, min(6.0, z)))
            }
            if let wbStr = params["wb"], let wb = Int(wbStr), (0...5).contains(wb) {
                self.currentWBMode = wb
                self.wbPopup?.selectItem(at: wb)
                self.sendCommand("WB:\(wb)")
            }
            if let afStr = params["af"], let af = Int(afStr), (0...3).contains(af) {
                self.currentAFMode = af
                self.afModePopup?.selectItem(at: af)
                self.sendCommand("AFMODE:\(af)")
            }
            if let wandStr = params["wand"] {
                if wandStr == "toggle" {
                    if self.isWandModeActive {
                        self.isWandModeActive = false
                        self.onResetGrade()
                    } else {
                        self.onMagicWandAutoTune()
                    }
                } else if wandStr == "1" || wandStr == "trigger" {
                    self.onMagicWandAutoTune()
                } else if wandStr == "0" || wandStr == "false" {
                    self.isWandModeActive = false
                    self.onResetGrade()
                }
            }
            if let resetStr = params["reset"], resetStr == "1" {
                self.onResetAllDefaults()
            }
            if let presetStr = params["preset"], let p = Int(presetStr), (0...6).contains(p) {
                self.isWandModeActive = false
                self.currentPreset = p
                self.ispCheck?.state = .on
                self.videoView.ispEnabled = true
                switch p {
                case 0: // Natural (ISP minimal)
                    self.videoView.warmthShift = 0.0
                    self.brightnessSlider?.doubleValue = 1.00
                    self.shadowSlider?.doubleValue = 0.04
                    self.contrastSlider?.doubleValue = 1.00
                    self.saturationSlider?.doubleValue = 1.00
                    self.denoiseSlider?.doubleValue = 0.15
                    self.sharpnessSlider?.doubleValue = 0.25
                    self.onGPUGradeChanged()
                case 1: // Studio Default
                    self.onResetGrade()
                case 2: // Broadcast (Low-Light Boost)
                    self.videoView.warmthShift = 0.0
                    self.brightnessSlider?.doubleValue = 1.28
                    self.shadowSlider?.doubleValue = 0.28
                    self.contrastSlider?.doubleValue = 1.08
                    self.saturationSlider?.doubleValue = 1.05
                    self.denoiseSlider?.doubleValue = 0.52
                    self.sharpnessSlider?.doubleValue = 0.48
                    self.onGPUGradeChanged()
                case 3: // Cinema (Rich contrast + warm filmic tone)
                    self.videoView.warmthShift = 0.08
                    self.brightnessSlider?.doubleValue = 1.06
                    self.shadowSlider?.doubleValue = 0.10
                    self.contrastSlider?.doubleValue = 1.18
                    self.saturationSlider?.doubleValue = 1.08
                    self.denoiseSlider?.doubleValue = 0.34
                    self.sharpnessSlider?.doubleValue = 0.62
                    self.onGPUGradeChanged()
                case 4: // Clean (Extra smooth 3DNR + neutral balance)
                    self.videoView.warmthShift = -0.04
                    self.brightnessSlider?.doubleValue = 1.14
                    self.shadowSlider?.doubleValue = 0.20
                    self.contrastSlider?.doubleValue = 1.03
                    self.saturationSlider?.doubleValue = 1.01
                    self.denoiseSlider?.doubleValue = 0.68
                    self.sharpnessSlider?.doubleValue = 0.44
                    self.onGPUGradeChanged()
                case 5: // Warm (Golden indoor warmth + soft shadows)
                    self.videoView.warmthShift = 0.24
                    self.brightnessSlider?.doubleValue = 1.14
                    self.shadowSlider?.doubleValue = 0.18
                    self.contrastSlider?.doubleValue = 1.05
                    self.saturationSlider?.doubleValue = 1.10
                    self.denoiseSlider?.doubleValue = 0.40
                    self.sharpnessSlider?.doubleValue = 0.50
                    self.onGPUGradeChanged()
                case 6: // Crisp (High-clarity detail + punchy micro-contrast)
                    self.videoView.warmthShift = 0.0
                    self.brightnessSlider?.doubleValue = 1.10
                    self.shadowSlider?.doubleValue = 0.14
                    self.contrastSlider?.doubleValue = 1.14
                    self.saturationSlider?.doubleValue = 1.06
                    self.denoiseSlider?.doubleValue = 0.22
                    self.sharpnessSlider?.doubleValue = 0.88
                    self.onGPUGradeChanged()
                default:
                    break
                }
            }
            if let rotStr = params["rotate"] {
                if rotStr == "step" {
                    self.onRotate90()
                } else if let r = Int(rotStr) {
                    self.manualRotationQuadrants = ((r % 4) + 4) % 4
                    self.applyEffectiveRotation()
                    _ = self.videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
                }
            }
            if let mh = params["mirrorH"], mh == "toggle" {
                self.videoView.mirrorHorizontal.toggle()
                self.mirrorCheck?.state = self.videoView.mirrorHorizontal ? .on : .off
                _ = self.videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
            }
            if let mv = params["mirrorV"], mv == "toggle" {
                self.videoView.flipVertical.toggle()
                self.flipVCheck?.state = self.videoView.flipVertical ? .on : .off
                _ = self.videoView.renderStudioProcessedPixelBuffer(updateHistory: true)
            }

            var ispUpdated = false
            if let s = params["denoise"], let v = Float(s) {
                self.denoiseSlider?.doubleValue = Double(max(0, min(1, v)))
                ispUpdated = true
            }
            if let s = params["sharpness"], let v = Float(s) {
                self.sharpnessSlider?.doubleValue = Double(max(0, min(1.5, v)))
                ispUpdated = true
            }
            if let s = params["brightness"], let v = Float(s) {
                self.brightnessSlider?.doubleValue = Double(max(0.5, min(1.8, v)))
                ispUpdated = true
            }
            if let s = params["shadowLift"], let v = Float(s) {
                self.shadowSlider?.doubleValue = Double(max(-0.25, min(0.8, v)))
                ispUpdated = true
            }
            if let s = params["contrast"], let v = Float(s) {
                self.contrastSlider?.doubleValue = Double(max(0.5, min(1.6, v)))
                ispUpdated = true
            }
            if let s = params["saturation"], let v = Float(s) {
                self.saturationSlider?.doubleValue = Double(max(0, min(1.8, v)))
                ispUpdated = true
            }
            if let s = params["warmth"], let v = Float(s) {
                self.videoView.warmthShift = max(-0.5, min(0.5, v))
                ispUpdated = true
            }

            if ispUpdated {
                self.onGPUGradeChanged()
            }
            self.updateDockButtonStates()
        }
        if Thread.isMainThread {
            applyBlock()
        } else {
            DispatchQueue.main.sync(execute: applyBlock)
        }
    }

    private static let obsDirectPlayerHTML: String = """
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>Pxl Cam 1080p Direct</title>
    <style>
      html, body {
        margin: 0; padding: 0; width: 100%; height: 100%;
        background: #000000; overflow: hidden;
      }
      img {
        width: 100vw; height: 100vh;
        object-fit: contain;
        image-rendering: high-quality;
        display: block;
      }
    </style>
    </head>
    <body>
      <img id="feed" src="/stream.mjpg" alt="Pxl Cam 1080p Feed" />
      <script>
        const img = document.getElementById('feed');
        img.addEventListener('error', () => {
          setTimeout(() => { img.src = '/stream.mjpg?t=' + Date.now(); }, 600);
        });
      </script>
    </body>
    </html>
    """

    private static let obsCustomDockHTML: String = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Pxl Cam Studio Dock</title>
    <style>
      :root {
        --bg: #0B0C10;
        --card: #13151C;
        --border: rgba(255, 132, 0, 0.24);
        --orange: #FF8400;
        --orange-dim: rgba(255, 132, 0, 0.16);
        --text: #F4F5F8;
        --muted: #8E95A5;
        --green: #19E673;
      }
      * { box-sizing: border-box; user-select: none; -webkit-user-select: none; }
      body {
        margin: 0; padding: 10px;
        background: var(--bg); color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
        font-size: 11px;
      }
      .header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 8px 10px; background: var(--card);
        border: 1px solid var(--border); border-radius: 10px;
        margin-bottom: 8px;
      }
      .brand { display: flex; align-items: center; gap: 8px; }
      .brand img.fox-logo { width: 32px; height: 32px; border-radius: 7px; border: 1px solid var(--orange); }
      .brand img.bttf-wordmark { height: 34px; width: auto; display: block; }
      .brand-title { font-weight: 800; font-size: 12px; letter-spacing: 0.4px; color: var(--orange); }
      .brand-sub { font-size: 10px; color: var(--muted); font-family: ui-monospace, SFMono-Regular, monospace; }
      .status-pill {
        display: flex; align-items: center; gap: 5px;
        padding: 3px 8px; border-radius: 999px;
        background: rgba(25, 230, 115, 0.12); border: 1px solid rgba(25, 230, 115, 0.4);
        font-family: ui-monospace, SFMono-Regular, monospace; font-size: 10px; font-weight: 700;
      }
      .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); }
      .preview-box {
        position: relative; width: 100%; aspect-ratio: 16 / 9;
        background: #050608; border-radius: 8px; overflow: hidden;
        border: 1px solid rgba(255, 132, 0, 0.28); margin-bottom: 8px;
      }
      .preview-box img#miniPreview { width: 100%; height: 100%; object-fit: contain; display: block; }
      .preview-topleft-brand {
        position: absolute; top: 6px; left: 8px;
        height: 44px; width: auto;
        pointer-events: none;
        filter: drop-shadow(0 2px 6px rgba(0, 0, 0, 0.85));
      }
      .preview-badge {
        position: absolute; bottom: 6px; left: 6px;
        background: rgba(0, 0, 0, 0.76); color: var(--orange);
        padding: 2px 6px; border-radius: 4px; font-size: 9.5px;
        font-family: ui-monospace, SFMono-Regular, monospace;
        border: 1px solid rgba(255, 132, 0, 0.35);
      }
      .section {
        background: var(--card); border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 9px; padding: 8px; margin-bottom: 8px;
      }
      .section-title {
        font-size: 9.5px; font-weight: 800; color: var(--orange);
        letter-spacing: 0.8px; text-transform: uppercase; margin-bottom: 6px;
      }
      .row { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 6px; }
      .row:last-child { margin-bottom: 0; }
      button {
        flex: 1; min-width: 44px; padding: 5px 7px;
        display: inline-flex; align-items: center; justify-content: space-between; gap: 5px;
        background: rgba(255, 255, 255, 0.045); color: #B8C0D0;
        border: 1px solid rgba(255, 255, 255, 0.13); border-radius: 6px;
        font-size: 10.5px; font-weight: 600; cursor: pointer;
        transition: all 0.12s ease;
      }
      button:hover { border-color: var(--orange); background: rgba(255, 132, 0, 0.12); color: #FFFFFF; }
      button.active {
        background: linear-gradient(135deg, rgba(255, 132, 0, 0.28), rgba(25, 230, 115, 0.14));
        color: #FFFFFF;
        border-color: var(--orange);
        box-shadow: 0 0 8px rgba(255, 132, 0, 0.28);
        font-weight: 700;
      }
      button.wand {
        background: rgba(255, 180, 0, 0.10);
        border-color: rgba(255, 180, 0, 0.35); color: #FFE2B0; font-weight: 700;
      }
      button.wand.active {
        background: linear-gradient(135deg, rgba(255, 132, 0, 0.38), rgba(255, 204, 0, 0.26));
        border-color: #FFB800; color: #FFFFFF;
        box-shadow: 0 0 10px rgba(255, 184, 0, 0.35);
      }
      .state-pill {
        font-family: ui-monospace, SFMono-Regular, monospace;
        font-size: 8.5px; font-weight: 800; letter-spacing: 0.3px;
        padding: 1.5px 4.5px; border-radius: 4px;
        background: rgba(255, 255, 255, 0.08);
        color: #7E8696;
        border: 1px solid rgba(255, 255, 255, 0.10);
        line-height: 1.1;
        flex-shrink: 0;
      }
      button.active .state-pill {
        background: rgba(25, 230, 115, 0.22);
        color: #19E673;
        border-color: rgba(25, 230, 115, 0.55);
      }
      .slider-row {
        display: grid; grid-template-columns: 68px 1fr 42px;
        align-items: center; gap: 6px; margin-bottom: 5px;
      }
      .slider-row:last-child { margin-bottom: 0; }
      .slider-label { font-size: 10.5px; color: #CFD4E0; font-weight: 500; }
      .slider-val {
        font-size: 10px; color: var(--orange); text-align: right;
        font-family: ui-monospace, SFMono-Regular, monospace; font-weight: 700;
      }
      input[type=range] {
        -webkit-appearance: none; width: 100%; height: 4px;
        background: rgba(255, 255, 255, 0.14); border-radius: 2px; outline: none;
      }
      input[type=range]::-webkit-slider-thumb {
        -webkit-appearance: none; width: 12px; height: 12px;
        border-radius: 50%; background: var(--orange); cursor: pointer;
        border: 1.5px solid #000;
      }
    </style>
    </head>
    <body>
      <div class="header">
        <div class="brand">
          <img class="fox-logo" src="/logo.png" alt="Pxl Cam" />
          <img class="bttf-wordmark" src="/wordmark.png" alt="PXL CAM" />
          <div>
            <div class="brand-sub" id="resText">1920x1080 • METAL ISP</div>
          </div>
        </div>
        <div class="status-pill">
          <div class="dot" id="connDot"></div>
          <span id="fpsText">30.0 FPS</span>
        </div>
      </div>

      <div class="preview-box">
        <img id="miniPreview" src="/stream.mjpg" alt="Live Monitor" />
        <img class="preview-topleft-brand" src="/overlay-brand.png" alt="Pxl Cam Overlay" />
        <div class="preview-badge">MINIMIZE-SAFE 1080P DIRECT</div>
      </div>

      <div class="section">
        <div class="section-title">Quick Actions & Hardware</div>
        <div class="row">
          <button class="wand" id="btnWand" title="AI Magic Wand: Toggle Auto-Tune Exposure, White Balance & Metal Studio ISP [W]" onclick="sendCtrl('wand=toggle')"><span>✨ Auto Studio Wand</span><span class="state-pill" id="pillWand">OFF</span></button>
          <button id="btnReset" title="Reset all Camera2 Hardware & Metal Studio ISP settings to defaults [D]" onclick="sendCtrl('reset=1')"><span>↺ Defaults</span><span class="state-pill" id="pillReset">ON</span></button>
        </div>
        <div class="row">
          <button id="lens0" title="Switch to Back Wide Camera (0) [F]" onclick="sendCtrl('lens=0')"><span>Back Lens</span><span class="state-pill" id="pillLens0">ON</span></button>
          <button id="lens1" title="Switch to Front Selfie Camera (1) [F]" onclick="sendCtrl('lens=1')"><span>Front Lens</span><span class="state-pill" id="pillLens1">OFF</span></button>
          <button id="btnFlash" title="Toggle Rear LED Torch Flash On / Off [T]" onclick="sendCtrl('flash=toggle')"><span>⚡ Flash</span><span class="state-pill" id="pillFlash">OFF</span></button>
        </div>
        <div class="slider-row" style="margin-top:4px; margin-bottom:5px;" title="Torch / Flash Dimmer (0% = OFF, 1%–100% = Smooth Optical + AE Torch Dimming)">
          <span class="slider-label">⚡ Dimmer</span>
          <input type="range" id="slFlashLevel" min="0" max="100" step="1" value="0" oninput="onSlider('flashLevel', this.value)" />
          <span class="slider-val" id="valFlashLevel">OFF</span>
        </div>
        <div class="slider-row" style="margin-bottom:6px;" title="Hardware Sensor Zoom Level (1.0× – 6.0×) [+ / -]">
          <span class="slider-label">🔍 Zoom</span>
          <input type="range" id="slQuickZoom" min="1.0" max="6.0" step="0.05" value="1.0" oninput="onSlider('zoom', this.value)" />
          <span class="slider-val" id="valQuickZoom">1.0×</span>
        </div>
        <div class="row">
          <button id="btnRotate" title="Rotate video 90° clockwise [R]" onclick="sendCtrl('rotate=step')"><span>↻ Rotate</span><span class="state-pill" id="pillRotate">OFF 0°</span></button>
          <button id="btnMirrorH" title="Mirror video horizontally (Left ↔ Right) [M]" onclick="sendCtrl('mirrorH=toggle')"><span>⇄ Flip H</span><span class="state-pill" id="pillMirrorH">OFF</span></button>
          <button id="btnMirrorV" title="Flip video vertically (Top ↔ Bottom)" onclick="sendCtrl('mirrorV=toggle')"><span>⇅ Flip V</span><span class="state-pill" id="pillMirrorV">OFF</span></button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Studio Look Presets</div>
        <div class="row">
          <button id="preset0" title="Natural Look (Minimal ISP grading)" onclick="sendCtrl('preset=0')"><span>Natural</span><span class="state-pill" id="pillPreset0">OFF</span></button>
          <button id="preset1" title="Studio Default (Balanced brightness, shadow lift & 3DNR)" onclick="sendCtrl('preset=1')"><span>Studio</span><span class="state-pill" id="pillPreset1">ON</span></button>
          <button id="preset2" title="Low-Light Boost (Lifted shadows + stronger 3DNR)" onclick="sendCtrl('preset=2')"><span>Broadcast</span><span class="state-pill" id="pillPreset2">OFF</span></button>
          <button id="preset3" title="Cinema Look (Rich contrast + warm filmic tone)" onclick="sendCtrl('preset=3')"><span>Cinema</span><span class="state-pill" id="pillPreset3">OFF</span></button>
        </div>
        <div class="row">
          <button id="preset4" title="Clean Studio Look (Extra smooth 3DNR)" onclick="sendCtrl('preset=4')"><span>Clean</span><span class="state-pill" id="pillPreset4">OFF</span></button>
          <button id="preset5" title="Warm Indoor Look (Golden indoor warmth)" onclick="sendCtrl('preset=5')"><span>Warm</span><span class="state-pill" id="pillPreset5">OFF</span></button>
          <button id="preset6" title="Crisp Detail Look (High-clarity detail)" onclick="sendCtrl('preset=6')"><span>Crisp</span><span class="state-pill" id="pillPreset6">OFF</span></button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Hardware Sensor (Camera2)</div>
        <div class="slider-row" title="Hardware Sensor Exposure Compensation (-8 EV to +8 EV) [[ / ]]">
          <span class="slider-label">HW EV</span>
          <input type="range" id="slEv" min="-8" max="8" step="1" value="0" oninput="onSlider('ev', this.value)" />
          <span class="slider-val" id="valEv">0</span>
        </div>
        <div class="row" style="margin-top:6px;">
          <button id="wb0" title="Auto White Balance (AWB)" onclick="sendCtrl('wb=0')"><span>AWB</span><span class="state-pill" id="pillWb0">ON</span></button>
          <button id="wb1" title="Daylight White Balance (5500K)" onclick="sendCtrl('wb=1')"><span>Day</span><span class="state-pill" id="pillWb1">OFF</span></button>
          <button id="wb2" title="Cloudy Daylight White Balance (6500K)" onclick="sendCtrl('wb=2')"><span>Cloud</span><span class="state-pill" id="pillWb2">OFF</span></button>
          <button id="wb3" title="Tungsten / Incandescent White Balance (2800K)" onclick="sendCtrl('wb=3')"><span>Tung</span><span class="state-pill" id="pillWb3">OFF</span></button>
          <button id="wb4" title="Fluorescent White Balance (4000K)" onclick="sendCtrl('wb=4')"><span>Fluo</span><span class="state-pill" id="pillWb4">OFF</span></button>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Metal Studio ISP</div>
        <div class="slider-row" title="Metal Studio ISP Exposure / Brightness Gain (70% – 160%)">
          <span class="slider-label">Exposure</span>
          <input type="range" id="slBrightness" min="0.70" max="1.60" step="0.01" value="1.12" oninput="onSlider('brightness', this.value)" />
          <span class="slider-val" id="valBrightness">112%</span>
        </div>
        <div class="slider-row" title="Metal Studio ISP Shadow Lift (-25% to +65%)">
          <span class="slider-label">Shadows</span>
          <input type="range" id="slShadowLift" min="-0.25" max="0.65" step="0.01" value="0.16" oninput="onSlider('shadowLift', this.value)" />
          <span class="slider-val" id="valShadowLift">16%</span>
        </div>
        <div class="slider-row" title="Metal Studio ISP S-Curve Contrast (75% – 145%)">
          <span class="slider-label">Contrast</span>
          <input type="range" id="slContrast" min="0.75" max="1.45" step="0.01" value="1.05" oninput="onSlider('contrast', this.value)" />
          <span class="slider-val" id="valContrast">105%</span>
        </div>
        <div class="slider-row" title="Metal GPU Bilateral + Temporal 3DNR Denoise (0% – 100%)">
          <span class="slider-label">Denoise</span>
          <input type="range" id="slDenoise" min="0.0" max="1.0" step="0.01" value="0.38" oninput="onSlider('denoise', this.value)" />
          <span class="slider-val" id="valDenoise">38%</span>
        </div>
        <div class="slider-row" title="Metal GPU Luma-Adaptive Unsharp Mask Sharpness (0% – 120%)">
          <span class="slider-label">Sharpness</span>
          <input type="range" id="slSharpness" min="0.0" max="1.2" step="0.01" value="0.52" oninput="onSlider('sharpness', this.value)" />
          <span class="slider-val" id="valSharpness">52%</span>
        </div>
        <div class="slider-row" title="Metal Studio ISP Color Saturation (50% – 150%)">
          <span class="slider-label">Saturation</span>
          <input type="range" id="slSaturation" min="0.5" max="1.5" step="0.01" value="1.04" oninput="onSlider('saturation', this.value)" />
          <span class="slider-val" id="valSaturation">104%</span>
        </div>
        <div class="slider-row" title="Metal Studio ISP Color Temperature Warmth (-0.5 Cool to +0.5 Warm)">
          <span class="slider-label">Warmth</span>
          <input type="range" id="slWarmth" min="-0.5" max="0.5" step="0.01" value="0.0" oninput="onSlider('warmth', this.value)" />
          <span class="slider-val" id="valWarmth">+0</span>
        </div>
      </div>

      <script>
        let dragging = false;
        document.querySelectorAll('input[type=range]').forEach(el => {
          el.addEventListener('mousedown', () => dragging = true);
          el.addEventListener('touchstart', () => dragging = true, {passive: true});
          el.addEventListener('mouseup', () => dragging = false);
          el.addEventListener('touchend', () => dragging = false);
        });
        window.addEventListener('mouseup', () => dragging = false);
        window.addEventListener('touchend', () => dragging = false);

        async function sendCtrl(q) {
          try {
            const res = await fetch('/api/control?' + q);
            const st = await res.json();
            renderState(st);
          } catch (e) {}
        }

        function onSlider(k, v) {
          sendCtrl(k + '=' + encodeURIComponent(v));
        }

        function setButtonState(btnId, pillId, isOn, customOnText, customOffText) {
          const el = document.getElementById(btnId);
          const pill = document.getElementById(pillId);
          if (el) {
            if (isOn) el.classList.add('active');
            else el.classList.remove('active');
          }
          if (pill) {
            pill.textContent = isOn ? (customOnText || 'ON') : (customOffText || 'OFF');
          }
        }

        function renderState(st) {
          document.getElementById('fpsText').textContent = (st.fps || 0).toFixed(1) + ' FPS';
          document.getElementById('resText').textContent = (st.frameW || 1920) + 'x' + (st.frameH || 1080) + ' • METAL ISP';
          document.getElementById('connDot').style.background = st.connected ? '#19E673' : '#FF453A';

          const fl = (st.flash && st.flashLevel > 0) ? st.flashLevel : 0;
          setButtonState('btnWand', 'pillWand', !!st.wand, 'ON', 'OFF');
          setButtonState('btnReset', 'pillReset', !!st.isDefault, 'ON', 'OFF');
          setButtonState('lens0', 'pillLens0', st.camera === 0, 'ON', 'OFF');
          setButtonState('lens1', 'pillLens1', st.camera === 1, 'ON', 'OFF');
          setButtonState('btnFlash', 'pillFlash', !!st.flash, fl > 0 ? ('ON ' + fl + '%') : 'ON', 'OFF');

          const rotDeg = ((st.rotation || 0) % 4) * 90;
          setButtonState('btnRotate', 'pillRotate', rotDeg !== 0, 'ON ' + rotDeg + '°', 'OFF 0°');
          setButtonState('btnMirrorH', 'pillMirrorH', !!st.mirrorH, 'ON', 'OFF');
          setButtonState('btnMirrorV', 'pillMirrorV', !!st.mirrorV, 'ON', 'OFF');

          for (let i = 0; i < 7; i++) {
            setButtonState('preset' + i, 'pillPreset' + i, st.preset === i, 'ON', 'OFF');
          }
          for (let i = 0; i < 5; i++) {
            setButtonState('wb' + i, 'pillWb' + i, st.wb === i, 'ON', 'OFF');
          }

          if (!dragging) {
            document.getElementById('slFlashLevel').value = fl;
            document.getElementById('slQuickZoom').value = st.zoom;
            document.getElementById('slEv').value = st.ev;
            document.getElementById('slBrightness').value = st.brightness;
            document.getElementById('slShadowLift').value = st.shadowLift;
            document.getElementById('slContrast').value = st.contrast;
            document.getElementById('slDenoise').value = st.denoise;
            document.getElementById('slSharpness').value = st.sharpness;
            document.getElementById('slSaturation').value = st.saturation;
            document.getElementById('slWarmth').value = st.warmth;
          }
          document.getElementById('valFlashLevel').textContent = fl > 0 ? fl + '%' : 'OFF';
          document.getElementById('valQuickZoom').textContent = Number(st.zoom).toFixed(1) + '×';
          document.getElementById('valEv').textContent = st.ev > 0 ? '+' + st.ev : '' + st.ev;
          document.getElementById('valBrightness').textContent = Math.round(st.brightness * 100) + '%';
          document.getElementById('valShadowLift').textContent = Math.round(st.shadowLift * 100) + '%';
          document.getElementById('valContrast').textContent = Math.round(st.contrast * 100) + '%';
          document.getElementById('valDenoise').textContent = Math.round(st.denoise * 100) + '%';
          document.getElementById('valSharpness').textContent = Math.round(st.sharpness * 100) + '%';
          document.getElementById('valSaturation').textContent = Math.round(st.saturation * 100) + '%';
          const w = Math.round(st.warmth * 100);
          document.getElementById('valWarmth').textContent = (w >= 0 ? '+' : '') + w;
        }

        async function poll() {
          try {
            const res = await fetch('/api/state');
            const st = await res.json();
            renderState(st);
          } catch (e) {}
        }
        setInterval(poll, 650);
        poll();
      </script>
    </body>
    </html>
    """
}

let app = NSApplication.shared
let delegate = PxlCamAppDelegate()
app.delegate = delegate
app.run()
