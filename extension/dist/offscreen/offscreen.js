let nsfwSessionPromise = null;
let detectorSessionPromise = null;

const NSFW_MODEL_VERSION = "nsfw_image_detection_int8_observe_only";
const NSFW_MODEL_PATH = "models/external/nsfw_image_detection_onnx/onnx/model_int8.onnx";

const DETECTOR_MODEL_VERSION = "yolov8n_coco_640_observe_only";
const DETECTOR_MODEL_PATH = "models/detectors/yolov8n_coco_640.onnx";

const DETECTOR_INPUT_SIZE = 640;
const DETECTOR_CONF_THRESHOLD = 0.25;
const DETECTOR_IOU_THRESHOLD = 0.45;

const COCO_NAMES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
  "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
  "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
  "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
  "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
  "scissors", "teddy bear", "hair drier", "toothbrush"
];

const RESTRICTED_COCO_MAP = {
  "bottle": {
    restricted_category: "possible_alcohol_container",
    evidence_strength: "weak",
    notes: "Generic bottle is weak evidence. It can be water, soda, alcohol, perfume, shampoo, or another container."
  },
  "wine glass": {
    restricted_category: "possible_alcohol_glass",
    evidence_strength: "medium",
    notes: "Wine glass is relevant but not proof of alcohol content."
  },
  "knife": {
    restricted_category: "weapon_like_object",
    evidence_strength: "medium",
    notes: "COCO knife can be kitchen context. Do not block from this alone."
  }
};


const POLICY_CONFIG = {
  mode: "observe_only",
  nsfwWarnThreshold: 0.50,
  nsfwReplaceThreshold: 0.85,
  restrictedObjectWarnThreshold: 0.50,
  unknownHandling: "unknown_observe_only",
  notes: "Policy recommendations are logged only. DOM replacement is not enabled in Milestone 5.1."
};

function getEvidenceStrengthRank(value) {
  if (value === "strong") {
    return 3;
  }

  if (value === "medium") {
    return 2;
  }

  if (value === "weak") {
    return 1;
  }

  return 0;
}

function selectStrongestRestrictedHit(hits) {
  if (!Array.isArray(hits) || hits.length === 0) {
    return null;
  }

  return [...hits].sort((a, b) => {
    const strengthDiff =
      getEvidenceStrengthRank(b.evidenceStrength) - getEvidenceStrengthRank(a.evidenceStrength);

    if (strengthDiff !== 0) {
      return strengthDiff;
    }

    return (b.confidence || 0) - (a.confidence || 0);
  })[0];
}

function buildGenerationMetadata(action, category, severity) {
  const eligibleForGeneratedReplacement = action === "replace" || action === "warn";

  return {
    eligible: eligibleForGeneratedReplacement,
    replacementType: eligibleForGeneratedReplacement ? "safe_generated_placeholder" : "none",
    category,
    severity,
    safePromptFamily: eligibleForGeneratedReplacement ? "family_friendly_abstract_placeholder" : "none",
    unsafePixelsUsed: false,
    note: "Future generation receives safe metadata only, not the original image pixels."
  };
}

function evaluatePolicy(result) {
  const nsfwScore =
    result && result.nsfw && typeof result.nsfw.nsfwScore === "number"
      ? result.nsfw.nsfwScore
      : null;

  const detectorHits =
    result && result.detector && Array.isArray(result.detector.restrictedHits)
      ? result.detector.restrictedHits
      : [];

  const strongestRestrictedHit = selectStrongestRestrictedHit(detectorHits);

  if (result.imageAccess !== "readable") {
    return {
      mode: POLICY_CONFIG.mode,
      recommendedAction: "unknown",
      category: "unknown_image_access",
      severity: "unknown",
      reason: "Image pixels were not readable, so SafeLens cannot evaluate the image.",
      evidence: {
        imageAccess: result.imageAccess,
        imageAccessReason: result.imageAccessReason
      },
      generationMetadata: buildGenerationMetadata("unknown", "unknown_image_access", "unknown")
    };
  }

  if (!result.nsfw.ok && !result.detector.ok) {
    return {
      mode: POLICY_CONFIG.mode,
      recommendedAction: "unknown",
      category: "model_failure",
      severity: "unknown",
      reason: "Both visual safety branches failed.",
      evidence: {
        nsfwReason: result.nsfw.reason,
        detectorReason: result.detector.reason
      },
      generationMetadata: buildGenerationMetadata("unknown", "model_failure", "unknown")
    };
  }

  if (nsfwScore !== null && nsfwScore >= POLICY_CONFIG.nsfwReplaceThreshold) {
    return {
      mode: POLICY_CONFIG.mode,
      recommendedAction: "replace",
      category: "explicit_nsfw",
      severity: "high",
      reason: "NSFW score is above the replace threshold.",
      evidence: {
        nsfwScore,
        threshold: POLICY_CONFIG.nsfwReplaceThreshold,
        modelVersion: result.nsfw.modelVersion
      },
      generationMetadata: buildGenerationMetadata("replace", "explicit_nsfw", "high")
    };
  }

  if (nsfwScore !== null && nsfwScore >= POLICY_CONFIG.nsfwWarnThreshold) {
    return {
      mode: POLICY_CONFIG.mode,
      recommendedAction: "warn",
      category: "possible_nsfw",
      severity: "medium",
      reason: "NSFW score is above the warning threshold but below the replace threshold.",
      evidence: {
        nsfwScore,
        threshold: POLICY_CONFIG.nsfwWarnThreshold,
        modelVersion: result.nsfw.modelVersion
      },
      generationMetadata: buildGenerationMetadata("warn", "possible_nsfw", "medium")
    };
  }

  if (
    strongestRestrictedHit &&
    strongestRestrictedHit.confidence >= POLICY_CONFIG.restrictedObjectWarnThreshold
  ) {
    return {
      mode: POLICY_CONFIG.mode,
      recommendedAction: "warn",
      category: strongestRestrictedHit.mappedRestrictedCategory,
      severity: strongestRestrictedHit.evidenceStrength === "weak" ? "low" : "medium",
      reason: "A COCO object mapped to a restricted evidence category was detected.",
      evidence: {
        label: strongestRestrictedHit.label,
        confidence: strongestRestrictedHit.confidence,
        mappedRestrictedCategory: strongestRestrictedHit.mappedRestrictedCategory,
        evidenceStrength: strongestRestrictedHit.evidenceStrength,
        note: strongestRestrictedHit.notes
      },
      generationMetadata: buildGenerationMetadata(
        "warn",
        strongestRestrictedHit.mappedRestrictedCategory,
        strongestRestrictedHit.evidenceStrength === "weak" ? "low" : "medium"
      )
    };
  }

  return {
    mode: POLICY_CONFIG.mode,
    recommendedAction: "allow",
    category: "safe_or_no_policy_trigger",
    severity: "none",
    reason: "No configured policy threshold was triggered.",
    evidence: {
      nsfwScore,
      restrictedHitCount: detectorHits.length,
      nsfwOk: result.nsfw.ok,
      detectorOk: result.detector.ok
    },
    generationMetadata: buildGenerationMetadata("allow", "safe_or_no_policy_trigger", "none")
  };
}


function configureOrt() {
  if (typeof ort === "undefined") {
    throw new Error("ONNX Runtime Web global 'ort' is not available in offscreen document.");
  }

  ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/onnxruntime-web/");
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
}

function getNsfwSession() {
  if (!nsfwSessionPromise) {
    configureOrt();

    nsfwSessionPromise = ort.InferenceSession.create(
      chrome.runtime.getURL(NSFW_MODEL_PATH),
      { executionProviders: ["wasm"] }
    );
  }

  return nsfwSessionPromise;
}

function getDetectorSession() {
  if (!detectorSessionPromise) {
    configureOrt();

    detectorSessionPromise = ort.InferenceSession.create(
      chrome.runtime.getURL(DETECTOR_MODEL_PATH),
      { executionProviders: ["wasm"] }
    );
  }

  return detectorSessionPromise;
}

function softmax2(a, b) {
  const maxValue = Math.max(a, b);
  const expA = Math.exp(a - maxValue);
  const expB = Math.exp(b - maxValue);
  const sum = expA + expB;

  return [expA / sum, expB / sum];
}

async function fetchImageBlob(src) {
  if (!src || typeof src !== "string") {
    return { ok: false, reason: "empty_image_src" };
  }

  if (src.startsWith("file:")) {
    return { ok: false, reason: "file_url_not_accessed_by_offscreen_document" };
  }

  try {
    const response = await fetch(src, {
      method: "GET",
      mode: "cors",
      credentials: "omit",
      cache: "force-cache"
    });

    if (!response.ok) {
      return { ok: false, reason: "http_status_" + response.status };
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.toLowerCase().startsWith("image/")) {
      return { ok: false, reason: "not_image_content_type_" + contentType };
    }

    const blob = await response.blob();

    return {
      ok: true,
      blob,
      reason: "fetch_blob_ok_" + Math.round(blob.size / 1024) + "kb"
    };
  } catch (error) {
    return {
      ok: false,
      reason: error instanceof Error ? error.name + ": " + error.message : "fetch_failed"
    };
  }
}

async function decodeNsfwTensor(blob) {
  const bitmap = await createImageBitmap(blob);
  const size = 224;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, size, size);

  const imageData = ctx.getImageData(0, 0, size, size);
  const data = imageData.data;
  const tensorData = new Float32Array(1 * 3 * size * size);

  let sumR = 0;
  let sumG = 0;
  let sumB = 0;

  for (let i = 0; i < size * size; i += 1) {
    const rgbaIndex = i * 4;

    const rRaw = data[rgbaIndex];
    const gRaw = data[rgbaIndex + 1];
    const bRaw = data[rgbaIndex + 2];

    sumR += rRaw;
    sumG += gRaw;
    sumB += bRaw;

    const r = (rRaw / 255 - 0.5) / 0.5;
    const g = (gRaw / 255 - 0.5) / 0.5;
    const b = (bRaw / 255 - 0.5) / 0.5;

    tensorData[i] = r;
    tensorData[size * size + i] = g;
    tensorData[2 * size * size + i] = b;
  }

  bitmap.close();

  return {
    ok: true,
    tensorData,
    inputWidth: size,
    inputHeight: size,
    meanR: Math.round(sumR / (size * size)),
    meanG: Math.round(sumG / (size * size)),
    meanB: Math.round(sumB / (size * size)),
    reason: "nsfw_decoded_resized_224_normalized"
  };
}

async function decodeDetectorTensor(blob) {
  const bitmap = await createImageBitmap(blob);
  const originalWidth = bitmap.width;
  const originalHeight = bitmap.height;

  const scale = Math.min(DETECTOR_INPUT_SIZE / originalWidth, DETECTOR_INPUT_SIZE / originalHeight);
  const resizedWidth = Math.round(originalWidth * scale);
  const resizedHeight = Math.round(originalHeight * scale);

  const padX = Math.round((DETECTOR_INPUT_SIZE - resizedWidth) / 2);
  const padY = Math.round((DETECTOR_INPUT_SIZE - resizedHeight) / 2);

  const canvas = document.createElement("canvas");
  canvas.width = DETECTOR_INPUT_SIZE;
  canvas.height = DETECTOR_INPUT_SIZE;

  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114, 114, 114)";
  ctx.fillRect(0, 0, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE);
  ctx.drawImage(bitmap, padX, padY, resizedWidth, resizedHeight);

  const imageData = ctx.getImageData(0, 0, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE);
  const data = imageData.data;
  const tensorData = new Float32Array(1 * 3 * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE);

  for (let i = 0; i < DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE; i += 1) {
    const rgbaIndex = i * 4;

    tensorData[i] = data[rgbaIndex] / 255;
    tensorData[DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE + i] = data[rgbaIndex + 1] / 255;
    tensorData[2 * DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE + i] = data[rgbaIndex + 2] / 255;
  }

  bitmap.close();

  return {
    ok: true,
    tensorData,
    originalWidth,
    originalHeight,
    resizedWidth,
    resizedHeight,
    scale,
    padX,
    padY,
    reason: "detector_letterbox_640_normalized"
  };
}

async function runNsfwInference(tensorData) {
  const startedAt = performance.now();
  const session = await getNsfwSession();

  const inputName = session.inputNames[0] || "pixel_values";
  const outputName = session.outputNames[0] || "logits";

  const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, 224, 224]);
  const outputs = await session.run({ [inputName]: inputTensor });

  const logits = Array.from(outputs[outputName].data);
  const probabilities = softmax2(logits[0], logits[1]);

  return {
    ok: true,
    modelVersion: NSFW_MODEL_VERSION,
    logits: logits.map((x) => Number(x.toFixed(6))),
    probabilities: probabilities.map((x) => Number(x.toFixed(6))),
    normalScore: Number(probabilities[0].toFixed(6)),
    nsfwScore: Number(probabilities[1].toFixed(6)),
    inferenceLatencyMs: Math.round(performance.now() - startedAt)
  };
}

function xywhToXyxy(cx, cy, width, height) {
  return [
    cx - width / 2,
    cy - height / 2,
    cx + width / 2,
    cy + height / 2
  ];
}

function unletterboxBox(box, meta) {
  let [x1, y1, x2, y2] = box;

  x1 = (x1 - meta.padX) / meta.scale;
  y1 = (y1 - meta.padY) / meta.scale;
  x2 = (x2 - meta.padX) / meta.scale;
  y2 = (y2 - meta.padY) / meta.scale;

  x1 = Math.max(0, Math.min(meta.originalWidth, x1));
  y1 = Math.max(0, Math.min(meta.originalHeight, y1));
  x2 = Math.max(0, Math.min(meta.originalWidth, x2));
  y2 = Math.max(0, Math.min(meta.originalHeight, y2));

  return [
    Number(x1.toFixed(2)),
    Number(y1.toFixed(2)),
    Number(x2.toFixed(2)),
    Number(y2.toFixed(2))
  ];
}

function boxIou(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);

  const interWidth = Math.max(0, x2 - x1);
  const interHeight = Math.max(0, y2 - y1);
  const intersection = interWidth * interHeight;

  const areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]);
  const areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);

  const union = areaA + areaB - intersection;
  return union <= 0 ? 0 : intersection / union;
}

function nms(detections, iouThreshold) {
  const sorted = [...detections].sort((a, b) => b.confidence - a.confidence);
  const selected = [];

  while (sorted.length > 0) {
    const current = sorted.shift();
    selected.push(current);

    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      if (current.classId === sorted[i].classId && boxIou(current.boxXyxy, sorted[i].boxXyxy) > iouThreshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return selected;
}

function parseYoloOutput(rawData, meta) {
  const numChannels = 84;
  const numCandidates = 8400;
  const candidates = [];

  for (let i = 0; i < numCandidates; i += 1) {
    const cx = rawData[i];
    const cy = rawData[numCandidates + i];
    const width = rawData[2 * numCandidates + i];
    const height = rawData[3 * numCandidates + i];

    let bestClassId = 0;
    let bestScore = -Infinity;

    for (let classId = 0; classId < 80; classId += 1) {
      const score = rawData[(4 + classId) * numCandidates + i];

      if (score > bestScore) {
        bestScore = score;
        bestClassId = classId;
      }
    }

    if (bestScore < DETECTOR_CONF_THRESHOLD) {
      continue;
    }

    const box = unletterboxBox(xywhToXyxy(cx, cy, width, height), meta);
    const boxWidth = box[2] - box[0];
    const boxHeight = box[3] - box[1];

    if (boxWidth <= 1 || boxHeight <= 1) {
      continue;
    }

    const label = COCO_NAMES[bestClassId] || "class_" + bestClassId;
    const restrictedMapping = RESTRICTED_COCO_MAP[label] || null;

    candidates.push({
      label,
      classId: bestClassId,
      confidence: bestScore,
      boxXyxy: box,
      restrictedMapping
    });
  }

  const selected = nms(candidates, DETECTOR_IOU_THRESHOLD)
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 20);

  return selected.map((item) => {
    const restrictedMapping = item.restrictedMapping;

    return {
      label: item.label,
      classId: item.classId,
      confidence: Number(item.confidence.toFixed(4)),
      boxXyxy: item.boxXyxy,
      mappedRestrictedCategory: restrictedMapping ? restrictedMapping.restricted_category : null,
      evidenceStrength: restrictedMapping ? restrictedMapping.evidence_strength : null,
      notes: restrictedMapping ? restrictedMapping.notes : null
    };
  });
}

async function runDetectorInference(decoded) {
  const startedAt = performance.now();
  const session = await getDetectorSession();

  const inputName = session.inputNames[0] || "images";
  const outputName = session.outputNames[0] || "output0";

  const inputTensor = new ort.Tensor(
    "float32",
    decoded.tensorData,
    [1, 3, DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE]
  );

  const outputs = await session.run({ [inputName]: inputTensor });
  const outputTensor = outputs[outputName];

  const detections = parseYoloOutput(outputTensor.data, decoded);
  const restrictedHits = detections.filter((item) => item.mappedRestrictedCategory !== null);

  return {
    ok: true,
    modelVersion: DETECTOR_MODEL_VERSION,
    inputSize: DETECTOR_INPUT_SIZE,
    confThreshold: DETECTOR_CONF_THRESHOLD,
    iouThreshold: DETECTOR_IOU_THRESHOLD,
    rawOutputShape: Array.from(outputTensor.dims),
    detections,
    detectionCount: detections.length,
    restrictedHits,
    restrictedHitCount: restrictedHits.length,
    inferenceLatencyMs: Math.round(performance.now() - startedAt)
  };
}

async function inspectImageAndRunInference(src) {
  const startedAt = performance.now();

  const result = {
    imageAccess: "unknown",
    imageAccessReason: "not_started",
    decoded: false,
    tensorReady: false,
    policyMode: "observe_only",
    policyDecision: "allow_observe_only",
    nsfw: {
      ok: false,
      modelVersion: NSFW_MODEL_VERSION,
      reason: "not_started"
    },
    detector: {
      ok: false,
      modelVersion: DETECTOR_MODEL_VERSION,
      reason: "not_started"
    }
  };

  const fetched = await fetchImageBlob(src);

  if (!fetched.ok) {
    return {
      ...result,
      imageAccess: "unreadable",
      imageAccessReason: fetched.reason,
      nsfw: {
        ...result.nsfw,
        reason: "image_unreadable"
      },
      detector: {
        ...result.detector,
        reason: "image_unreadable"
      },
      latencyMs: Math.round(performance.now() - startedAt)
    };
  }

  result.imageAccess = "readable";
  result.imageAccessReason = fetched.reason;

  try {
    const nsfwDecoded = await decodeNsfwTensor(fetched.blob);
    const nsfwInference = await runNsfwInference(nsfwDecoded.tensorData);

    result.decoded = true;
    result.tensorReady = true;
    result.decodeReason = nsfwDecoded.reason;
    result.inputWidth = nsfwDecoded.inputWidth;
    result.inputHeight = nsfwDecoded.inputHeight;
    result.meanR = nsfwDecoded.meanR;
    result.meanG = nsfwDecoded.meanG;
    result.meanB = nsfwDecoded.meanB;
    result.nsfw = {
      ok: true,
      reason: "real_nsfw_inference_ok_observe_only",
      ...nsfwInference
    };
  } catch (error) {
    result.nsfw = {
      ok: false,
      modelVersion: NSFW_MODEL_VERSION,
      reason: error instanceof Error ? error.name + ": " + error.message : "real_nsfw_inference_failed"
    };
  }

  try {
    const detectorDecoded = await decodeDetectorTensor(fetched.blob);
    const detectorInference = await runDetectorInference(detectorDecoded);

    result.detector = {
      ok: true,
      reason: "yolo_detector_inference_ok_observe_only",
      ...detectorInference
    };
  } catch (error) {
    result.detector = {
      ok: false,
      modelVersion: DETECTOR_MODEL_VERSION,
      reason: error instanceof Error ? error.name + ": " + error.message : "yolo_detector_inference_failed"
    };
  }

  const policyRecommendation = evaluatePolicy(result);

  return {
    ...result,
    policyRecommendation,
    policyRecommendedAction: policyRecommendation.recommendedAction,
    policyCategory: policyRecommendation.category,
    policySeverity: policyRecommendation.severity,
    policyReason: policyRecommendation.reason,
    policyEvidence: policyRecommendation.evidence,
    generationMetadata: policyRecommendation.generationMetadata,
    inferenceOk: result.nsfw.ok,
    inferenceReason: result.nsfw.reason,
    modelVersion: result.nsfw.modelVersion,
    probabilities: result.nsfw.probabilities,
    normalScore: result.nsfw.normalScore,
    nsfwScore: result.nsfw.nsfwScore,
    detectorOk: result.detector.ok,
    detectorReason: result.detector.reason,
    detectorModelVersion: result.detector.modelVersion,
    detectionCount: result.detector.detectionCount || 0,
    restrictedHitCount: result.detector.restrictedHitCount || 0,
    detections: result.detector.detections || [],
    restrictedHits: result.detector.restrictedHits || [],
    latencyMs: Math.round(performance.now() - startedAt)
  };
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.target !== "offscreen" || message.type !== "SAFELENS_OFFSCREEN_INFER") {
    return false;
  }

  const src = message.payload && message.payload.src;

  inspectImageAndRunInference(src)
    .then((inspection) => {
      sendResponse({
        type: "SAFELENS_CLASSIFY_IMAGE_RESULT",
        payload: {
          prediction: {
            status: "combined_visual_safety_observe_only",
            reason: "Milestone 5.1 runs NSFW classification, YOLO object detection, and observe-only policy recommendation.",
            ...inspection
          },
          decision: {
            action: "allow",
            reason: "Observe-only mode: policy recommendation is logged, but DOM replacement is not enabled yet."
          }
        }
      });
    })
    .catch((error) => {
      sendResponse({
        type: "SAFELENS_CLASSIFY_IMAGE_RESULT",
        payload: {
          prediction: {
            status: "combined_visual_safety_failed",
            reason: error instanceof Error ? error.name + ": " + error.message : "combined_inference_failed",
            policyMode: "observe_only"
          },
          decision: {
            action: "allow",
            reason: "Fail-open observe-only mode."
          }
        }
      });
    });

  return true;
});

console.log("[SafeLens] Combined NSFW + YOLO detector observe-only offscreen document loaded.");
