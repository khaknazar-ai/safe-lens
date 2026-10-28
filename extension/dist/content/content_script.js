const STORAGE_KEY = "safelensStats";
const processedImages = new WeakSet();

const GENERATION_SERVICE_BASE_URL = "http://127.0.0.1:8787";
const GENERATION_SERVICE_TIMEOUT_MS = 180000;

const RESTRICTED_DETECTOR_SERVICE_BASE_URL = "http://127.0.0.1:8790";
const RESTRICTED_DETECTOR_TIMEOUT_MS = 60000;
const RESTRICTED_DETECTOR_MODEL_ID = "cigarette-vape-detection-xrnvm/1";
const RESTRICTED_DETECTOR_TARGET_CLASSES = ["cigarette", "vape"];
const RESTRICTED_DETECTOR_HIT_CONFIDENCE = 0.35;
const INPAINTING_CLIENT_LOCK_KEY = "safelensInpaintingClientLockV2";
const INPAINTING_CLIENT_LOCK_TTL_MS = 30000;

// Final project policy default:
// high risk  -> full-image safe replacement
// medium risk -> localized object cleanup / inpainting when explicit demo or future user setting allows it
// low risk -> observe / warn only
const ENABLE_POLICY_PLACEHOLDER_REPLACEMENT = false;
const ENABLE_POLICY_INPAINTING_REPLACEMENT = false;
const ENABLE_LOCAL_DEMO_BACKENDS = false;

const HIGH_RISK_NSFW_THRESHOLD = 0.5;
const MEDIUM_RISK_OBJECT_CONFIDENCE_THRESHOLD = 0.5;
const LOW_RISK_OBJECT_CONFIDENCE_THRESHOLD = 0.25;

let statsUpdateQueue = Promise.resolve();

const DEFAULT_STATS = {
  scanned: 0,
  restored: 0,
  replaced: 0,
  readable: 0,
  decoded: 0,
  tensorReady: 0,
  unreadable: 0,
  errors: 0,

  inferenceOk: 0,
  inferenceFailed: 0,

  detectorOk: 0,
  detectorFailed: 0,

  restrictedDetectorOk: 0,
  restrictedDetectorFailed: 0,

  generationServiceOk: 0,
  generationServiceFailed: 0,
  generationServiceBusy: 0,
  generatedReplacementApplied: 0,
  generatedReplacementSkipped: 0,

  lastStatus: "unknown",
  lastAccess: "unknown",
  lastAccessReason: "none",
  lastDecodeReason: "none",
  lastInferenceReason: "none",
  lastModelVersion: "none",
  lastProbabilities: null,

  lastDetectorReason: "none",
  lastDetectorModelVersion: "none",
  lastDetectionCount: 0,
  lastDetections: [],
  lastRestrictedHitCount: 0,
  lastRestrictedHits: [],

  lastPolicyRecommendedAction: "none",
  lastPolicyCategory: "none",
  lastPolicySeverity: "none",
  lastPolicyReason: "none",
  lastPolicyEvidence: null,
  lastGenerationMetadata: null,

  lastRiskLevel: "none",
  lastRiskSource: "none",
  lastRiskAction: "none",
  lastRiskReason: "none",
  lastRiskTargetLabel: "none",
  lastRiskTargetConfidence: 0,

  lastRestrictedDetectorStatus: "none",
  lastRestrictedDetectorModel: "none",
  lastRestrictedDetectorReason: "none",
  lastRestrictedDetectorLatencyMs: 0,
  lastRestrictedDetectorPredictionCount: 0,
  lastRestrictedDetectorTargetHitCount: 0,
  lastRestrictedDetectorHits: [],
  lastRestrictedDetectorError: "none",
  lastRestrictedCleanupDecisionReason: "none",

  lastGenerationServiceStatus: "none",
  lastGenerationBackend: "none",
  lastGeneratedImageUrl: "none",
  lastReplacementMode: "none",
  lastReplacementReason: "none",
  lastGenerationLatencyMs: 0,
  lastGenerationEndpoint: "none",
  lastInpaintingReportPath: "none",
  lastInpaintingOutputUrl: "none"
};

function mergeStats(stats) {
  return {
    ...DEFAULT_STATS,
    ...(stats || {})
  };
}

function getStoredStats() {
  return new Promise((resolve) => {
    chrome.storage.local.get(STORAGE_KEY, (data) => {
      resolve(mergeStats(data[STORAGE_KEY]));
    });
  });
}

function setStoredStats(stats) {
  return new Promise((resolve) => {
    chrome.storage.local.set({ [STORAGE_KEY]: stats }, resolve);
  });
}

function updateStats(mutator) {
  statsUpdateQueue = statsUpdateQueue
    .then(async () => {
      const current = await getStoredStats();
      const next = mutator(current) || current;
      await setStoredStats(next);
    })
    .catch((error) => {
      console.warn("[SafeLens] Stats update failed:", error);
    });

  return statsUpdateQueue;
}

function getStorageValue(key) {
  return new Promise((resolve) => {
    chrome.storage.local.get(key, (data) => {
      resolve(data[key]);
    });
  });
}

function setStorageValue(key, value) {
  return new Promise((resolve) => {
    chrome.storage.local.set({ [key]: value }, resolve);
  });
}

function removeStorageValue(key) {
  return new Promise((resolve) => {
    chrome.storage.local.remove(key, resolve);
  });
}

function createLockToken() {
  return String(Date.now()) + "_" + Math.random().toString(16).slice(2);
}

async function tryAcquireInpaintingClientLock(jobKey) {
  const now = Date.now();
  const existing = await getStorageValue(INPAINTING_CLIENT_LOCK_KEY);

  if (existing && existing.expiresAt && existing.expiresAt > now) {
    return {
      acquired: false,
      reason: "client_lock_active",
      existing
    };
  }

  const token = createLockToken();
  const lock = {
    token,
    jobKey,
    startedAt: now,
    expiresAt: now + INPAINTING_CLIENT_LOCK_TTL_MS
  };

  await setStorageValue(INPAINTING_CLIENT_LOCK_KEY, lock);

  return {
    acquired: true,
    token
  };
}

async function releaseInpaintingClientLock(token) {
  const existing = await getStorageValue(INPAINTING_CLIENT_LOCK_KEY);

  if (existing && existing.token === token) {
    await removeStorageValue(INPAINTING_CLIENT_LOCK_KEY);
  }
}

function getImageSrc(image) {
  return image.currentSrc || image.src || "";
}

function shouldProcessImage(image) {
  const src = getImageSrc(image);

  if (!src) {
    return false;
  }

  if (processedImages.has(image)) {
    return false;
  }

  return true;
}

function createTemporaryCard(image) {
  const rect = image.getBoundingClientRect();
  const card = document.createElement("div");

  card.textContent = "SafeLens checking image locally...";
  card.setAttribute("data-safelens-card", "true");

  card.style.boxSizing = "border-box";
  card.style.width = Math.max(160, Math.round(rect.width || image.width || 240)) + "px";
  card.style.minHeight = Math.max(80, Math.round(rect.height || image.height || 120)) + "px";
  card.style.display = "flex";
  card.style.alignItems = "center";
  card.style.justifyContent = "center";
  card.style.padding = "12px";
  card.style.border = "1px solid #cbd5e1";
  card.style.borderRadius = "12px";
  card.style.background = "#f8fafc";
  card.style.color = "#334155";
  card.style.font = "13px system-ui, sans-serif";
  card.style.textAlign = "center";

  image.dataset.safelensOriginalDisplay = image.style.display || "";
  image.style.display = "none";

  image.insertAdjacentElement("afterend", card);
  return card;
}

function removeCard(card) {
  if (card && card.parentNode) {
    card.parentNode.removeChild(card);
  }
}

function restoreImage(image, card) {
  image.style.display = image.dataset.safelensOriginalDisplay || "";
  removeCard(card);
}

function summarizeDetections(detections) {
  if (!Array.isArray(detections) || detections.length === 0) {
    return [];
  }

  return detections.slice(0, 5).map((item) => ({
    label: item.label,
    confidence: item.confidence,
    box: item.boxXyxy
  }));
}

function summarizeRestrictedHits(hits) {
  if (!Array.isArray(hits) || hits.length === 0) {
    return [];
  }

  return hits.slice(0, 5).map((item) => ({
    label: item.label,
    confidence: item.confidence,
    category: item.mappedRestrictedCategory,
    evidence: item.evidenceStrength,
    box: item.boxXyxy
  }));
}

function clampDimension(value, fallback) {
  const number = Number(value || fallback || 512);

  if (!Number.isFinite(number)) {
    return fallback;
  }

  return Math.max(128, Math.min(1024, Math.round(number)));
}

function getNsfwScore(prediction) {
  const probabilities = prediction?.probabilities;

  if (!Array.isArray(probabilities) || probabilities.length < 2) {
    return 0;
  }

  const score = Number(probabilities[1]);

  if (!Number.isFinite(score)) {
    return 0;
  }

  return score;
}

function findStrongestRestrictedHit(prediction) {
  const hits = Array.isArray(prediction?.restrictedHits)
    ? prediction.restrictedHits
    : [];

  if (hits.length === 0) {
    return null;
  }

  return hits.reduce((best, current) => {
    const bestScore = Number(best?.confidence || 0);
    const currentScore = Number(current?.confidence || 0);
    return currentScore > bestScore ? current : best;
  }, hits[0]);
}

function hasExplicitHighRiskMarker(image) {
  const source = String(getImageSrc(image) || "").toLowerCase();

  return (
    image.dataset.safelensHighRiskFallbackTest === "true" ||
    source.includes("nsfw_private_test") ||
    source.includes("private_nsfw") ||
    source.includes("restricted_visual_test")
  );
}

function evaluateRiskPolicy(image, prediction) {
  const nsfwScore = getNsfwScore(prediction);
  const strongestHit = findStrongestRestrictedHit(prediction);
  const category = String(prediction?.policyCategory || "").toLowerCase();

  if (
    hasExplicitHighRiskMarker(image) ||
    category.includes("explicit") ||
    category.includes("nsfw") ||
    category.includes("sexual") ||
    nsfwScore >= HIGH_RISK_NSFW_THRESHOLD
  ) {
    return {
      riskLevel: "high",
      riskSource: hasExplicitHighRiskMarker(image) ? "explicit_high_risk_marker" : "image_level_nsfw",
      recommendedReplacementMode: "safe_placeholder",
      recommendedAction: "full_image_replacement",
      reason: "high_risk_image_level_content_not_eligible_for_inpainting",
      targetLabel: "none",
      targetConfidence: nsfwScore
    };
  }

  if (strongestHit) {
    const confidence = Number(strongestHit.confidence || 0);
    const hitCategory = String(strongestHit.mappedRestrictedCategory || "");

    if (confidence >= MEDIUM_RISK_OBJECT_CONFIDENCE_THRESHOLD) {
      return {
        riskLevel: "medium",
        riskSource: "localized_restricted_object",
        recommendedReplacementMode: "local_inpainting_research",
        recommendedAction: strongestHit.default_action || "localized_object_cleanup",
        reason: "medium_risk_localized_object_with_bbox",
        targetLabel: strongestHit.label || "object",
        targetConfidence: confidence,
        targetCategory: hitCategory
      };
    }

    if (confidence >= LOW_RISK_OBJECT_CONFIDENCE_THRESHOLD) {
      return {
        riskLevel: "low",
        riskSource: "weak_localized_restricted_object",
        recommendedReplacementMode: "observe_only",
        recommendedAction: "warn_or_observe",
        reason: "low_risk_or_weak_evidence_no_automatic_replacement",
        targetLabel: strongestHit.label || "object",
        targetConfidence: confidence,
        targetCategory: hitCategory
      };
    }
  }

  return {
    riskLevel: "none",
    riskSource: "no_policy_trigger",
    recommendedReplacementMode: "none",
    recommendedAction: "allow",
    reason: "no_risk_threshold_triggered",
    targetLabel: "none",
    targetConfidence: 0
  };
}


function fuseRestrictedDetectorRiskDecision(baseRiskDecision, restrictedDetectorResult) {
  if (
    restrictedDetectorResult &&
    restrictedDetectorResult.ok &&
    restrictedDetectorResult.targetHitCount > 0 &&
    Array.isArray(restrictedDetectorResult.targetHits) &&
    restrictedDetectorResult.targetHits.length > 0
  ) {
    const strongestHit = restrictedDetectorResult.targetHits.reduce((best, current) => {
      const bestScore = Number(best?.confidence || 0);
      const currentScore = Number(current?.confidence || 0);
      return currentScore > bestScore ? current : best;
    }, restrictedDetectorResult.targetHits[0]);

    return {
      riskLevel: "medium",
      riskSource: "dedicated_restricted_detector",
      recommendedReplacementMode: "observe_only",
      recommendedAction: strongestHit.default_action || "localized_object_cleanup",
      reason: "dedicated_restricted_detector_category_hit_with_bbox",
      targetLabel: strongestHit.label || "restricted_object",
      targetConfidence: Number(strongestHit.confidence || 0),
      targetCategory: strongestHit.category || "restricted_object",
      targetBox: strongestHit.box || strongestHit.box_xyxy || null
    };
  }

  return baseRiskDecision;
}

function shouldRequestSafePlaceholder(image, prediction, riskDecision) {
  if (ENABLE_LOCAL_DEMO_BACKENDS && image.dataset.safelensForcePlaceholder === "true") {
    return {
      shouldReplace: true,
      reason: "demo_force_placeholder_attribute"
    };
  }

  if (
    riskDecision.riskLevel === "high" &&
    (
      ENABLE_POLICY_PLACEHOLDER_REPLACEMENT ||
      (ENABLE_LOCAL_DEMO_BACKENDS &&
        (image.dataset.safelensHighRiskFallbackTest === "true" ||
          image.dataset.safelensForceInpaint === "true"))
    )
  ) {
    return {
      shouldReplace: true,
      reason: "high_risk_full_image_replacement"
    };
  }

  return {
    shouldReplace: false,
    reason: "replacement_not_requested"
  };
}

function shouldRequestLocalInpainting(image, riskDecision) {
  if (
    ENABLE_LOCAL_DEMO_BACKENDS &&
    image.dataset.safelensForceInpaint === "true" &&
    riskDecision.riskLevel === "medium"
  ) {
    return {
      shouldReplace: true,
      reason: "medium_risk_local_object_cleanup"
    };
  }

  if (
    ENABLE_POLICY_INPAINTING_REPLACEMENT &&
    riskDecision.riskLevel === "medium"
  ) {
    return {
      shouldReplace: true,
      reason: "policy_medium_risk_local_object_cleanup"
    };
  }

  if (
    ENABLE_LOCAL_DEMO_BACKENDS &&
    image.dataset.safelensForceInpaint === "true" &&
    riskDecision.riskLevel === "high"
  ) {
    return {
      shouldReplace: false,
      reason: "inpainting_blocked_high_risk"
    };
  }

  return {
    shouldReplace: false,
    reason: "local_inpainting_not_requested"
  };
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    return await fetch(url, {
      ...options,
      signal: controller.signal
    });
  } finally {
    clearTimeout(timeoutId);
  }
}


function getRestrictedDetectorLocalPath(image) {
  return image.dataset.safelensRestrictedDetectorLocalPath || "";
}

function shouldRequestRestrictedDetector(image) {
  const localPath = getRestrictedDetectorLocalPath(image);

  if (ENABLE_LOCAL_DEMO_BACKENDS && localPath) {
    return {
      shouldRun: true,
      reason: "restricted_detector_local_path_present",
      imagePath: localPath
    };
  }

  return {
    shouldRun: false,
    reason: "restricted_detector_not_requested",
    imagePath: ""
  };
}


function getRestrictedDetectorModelId(image) {
  return image.dataset.safelensRestrictedDetectorModelId || RESTRICTED_DETECTOR_MODEL_ID;
}

function getRestrictedDetectorTargetClasses(image) {
  const raw = image.dataset.safelensRestrictedDetectorTargetClasses || "";

  if (!raw.trim()) {
    return RESTRICTED_DETECTOR_TARGET_CLASSES;
  }

  try {
    const parsed = JSON.parse(raw);

    if (Array.isArray(parsed)) {
      return parsed.map((item) => String(item).trim()).filter(Boolean);
    }
  } catch (_error) {
    // Fall back to comma-separated parsing.
  }

  return raw
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function getRestrictedDetectorHitConfidence(image) {
  const raw = image.dataset.safelensRestrictedDetectorHitConfidence;

  if (!raw) {
    return RESTRICTED_DETECTOR_HIT_CONFIDENCE;
  }

  const value = Number(raw);

  if (!Number.isFinite(value)) {
    return RESTRICTED_DETECTOR_HIT_CONFIDENCE;
  }

  return value;
}

function normalizeRestrictedServiceHits(serviceResult) {
  if (Array.isArray(serviceResult.category_hits) && serviceResult.category_hits.length > 0) {
    return serviceResult.category_hits;
  }

  if (Array.isArray(serviceResult.target_hits)) {
    return serviceResult.target_hits;
  }

  return [];
}

function countRestrictedServiceHits(serviceResult) {
  if (typeof serviceResult.category_hit_count === "number") {
    return serviceResult.category_hit_count;
  }

  if (typeof serviceResult.target_hit_count === "number") {
    return serviceResult.target_hit_count;
  }

  return 0;
}

function summarizeRestrictedDetectorServiceHits(hits) {
  if (!Array.isArray(hits) || hits.length === 0) {
    return [];
  }

  return hits.slice(0, 5).map((item) => ({
    label: item.label,
    confidence: item.confidence,
    box: item.box_xyxy || item.boxXyxy || item.box
  }));
}

async function requestRestrictedDetectorService(image, decision) {
  const startedAt = performance.now();

  const response = await fetchWithTimeout(
    RESTRICTED_DETECTOR_SERVICE_BASE_URL + "/v1/restricted/detect-local-file",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        image_path: decision.imagePath,
        model_id: getRestrictedDetectorModelId(image),
        target_classes: getRestrictedDetectorTargetClasses(image),
        hit_confidence: getRestrictedDetectorHitConfidence(image)
      })
    },
    RESTRICTED_DETECTOR_TIMEOUT_MS
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error("restricted_detector_service_http_" + response.status + ": " + text.slice(0, 240));
  }

  const data = await response.json();

  return {
    ...data,
    client_latency_ms: Math.round(performance.now() - startedAt)
  };
}



function shouldRequestRestrictedObjectCleanup(image, riskDecision, restrictedDetectorResult) {
  if (image.dataset.safelensForceRestrictedCleanup !== "true") {
    return {
      shouldReplace: false,
      reason: "restricted_object_cleanup_not_requested"
    };
  }

  if (!riskDecision || riskDecision.riskLevel !== "medium") {
    return {
      shouldReplace: false,
      reason: "restricted_object_cleanup_requires_medium_risk"
    };
  }

  if (riskDecision.riskSource !== "dedicated_restricted_detector") {
    return {
      shouldReplace: false,
      reason: "restricted_object_cleanup_requires_dedicated_detector"
    };
  }

  const detectorBox =
    Array.isArray(riskDecision.targetBox) && riskDecision.targetBox.length >= 4
      ? riskDecision.targetBox
      : (
          restrictedDetectorResult &&
          Array.isArray(restrictedDetectorResult.targetHits) &&
          restrictedDetectorResult.targetHits.length > 0 &&
          Array.isArray(restrictedDetectorResult.targetHits[0].box) &&
          restrictedDetectorResult.targetHits[0].box.length >= 4
            ? restrictedDetectorResult.targetHits[0].box
            : null
        );

  if (!detectorBox) {
    return {
      shouldReplace: false,
      reason: "restricted_object_cleanup_missing_bbox"
    };
  }

  riskDecision.targetBox = detectorBox;

  const localPath = getRestrictedDetectorLocalPath(image);

  if (!localPath) {
    return {
      shouldReplace: false,
      reason: "restricted_object_cleanup_missing_local_path"
    };
  }

  return {
    shouldReplace: true,
    reason: "bbox_guided_restricted_object_cleanup",
    imagePath: localPath
  };
}

async function requestRestrictedObjectCleanup(image, riskDecision) {
  const startedAt = performance.now();
  const localPath = getRestrictedDetectorLocalPath(image);
  const targetLabel = riskDecision.targetLabel || "restricted_object";
  const outputStem =
    image.dataset.safelensRestrictedCleanupOutputStem ||
    ("browser_restricted_cleanup_" + targetLabel);

  const reportPath =
    image.dataset.safelensRestrictedCleanupReportPath ||
    ("reports/generation_service/" + outputStem + "_report.json");

  const response = await fetchWithTimeout(
    GENERATION_SERVICE_BASE_URL + "/v1/inpaint-local-bbox",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        input_image: localPath,
        box_xyxy: riskDecision.targetBox,
        target_label: targetLabel,
        output_stem: outputStem,
        report_path: reportPath,
        box_expand_ratio: Number(image.dataset.safelensRestrictedCleanupBoxExpandRatio || 0.18),
        inpaint_radius: Number(image.dataset.safelensRestrictedCleanupInpaintRadius || 5.0),
        method: image.dataset.safelensRestrictedCleanupMethod || "telea"
      })
    },
    GENERATION_SERVICE_TIMEOUT_MS
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error("bbox_cleanup_service_http_" + response.status + ": " + text.slice(0, 300));
  }

  const data = await response.json();

  if (!data.image_url && !data.output_url) {
    throw new Error("bbox_cleanup_missing_output_url");
  }

  return {
    status: data.status || "completed",
    backend: data.backend || "bbox_guided_local_cleanup",
    imageUrl: data.image_url || data.output_url,
    outputUrl: data.output_url || data.image_url,
    reportPath: data.report_path || reportPath,
    latencyMs: data.latency_ms || Math.round(performance.now() - startedAt),
    payload: {
      input_image: localPath,
      box_xyxy: riskDecision.targetBox,
      target_label: targetLabel
    },
    raw: data
  };
}


function buildPlaceholderRequest(image, prediction, replacementReason) {
  const width = clampDimension(image.naturalWidth || image.width || image.getBoundingClientRect().width, 512);
  const height = clampDimension(image.naturalHeight || image.height || image.getBoundingClientRect().height, 320);

  const highRisk = replacementReason === "high_risk_full_image_replacement";

  return {
    recommended_action: highRisk ? "replace" : (prediction.policyRecommendedAction === "none" ? "warn" : (prediction.policyRecommendedAction || "warn")),
    category: highRisk
      ? "high_risk_visual_content"
      : (
          prediction.policyCategory && prediction.policyCategory !== "none"
            ? prediction.policyCategory
            : "debug_forced_safe_placeholder"
        ),
    severity: highRisk
      ? "high"
      : (
          prediction.policySeverity && prediction.policySeverity !== "none"
            ? prediction.policySeverity
            : "low"
        ),
    replacement_type: "safe_generated_placeholder",
    safe_prompt_family: "family_friendly_abstract_placeholder",
    width,
    height,
    style: "soft_abstract",
    seed: 123,
    unsafe_pixels_used: false,
    debug_reason: replacementReason
  };
}

async function requestSafePlaceholder(image, prediction, replacementReason) {
  const startedAt = performance.now();
  const payload = buildPlaceholderRequest(image, prediction, replacementReason);

  const response = await fetchWithTimeout(
    GENERATION_SERVICE_BASE_URL + "/v1/generate-placeholder",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    },
    GENERATION_SERVICE_TIMEOUT_MS
  );

  if (!response.ok) {
    const text = await response.text();
    throw new Error("generation_service_http_" + response.status + ": " + text.slice(0, 200));
  }

  const data = await response.json();

  if (!data.image_url) {
    throw new Error("generation_service_missing_image_url");
  }

  return {
    status: data.status || "completed",
    backend: data.backend || "safe_placeholder",
    imageUrl: data.image_url,
    latencyMs: Math.round(performance.now() - startedAt),
    payload
  };
}

function buildLocalInpaintingRequest(image, riskDecision) {
  const targetLabel =
    image.dataset.safelensInpaintTargetLabel ||
    riskDecision.targetLabel ||
    "object";

  return {
    input_image: image.dataset.safelensInpaintInputImage || "uploads/input.png",
    target_label: targetLabel,
    prompt: image.dataset.safelensInpaintPrompt || "clean realistic background, natural lighting, object removed",
    negative_prompt: image.dataset.safelensInpaintNegativePrompt || "object, text, logo, watermark, distorted, low quality",
    steps: Number(image.dataset.safelensInpaintSteps || 15),
    max_size: Number(image.dataset.safelensInpaintMaxSize || 512),
    box_expand_ratio: Number(image.dataset.safelensInpaintBoxExpandRatio || 0.12),
    mask_blur_radius: Number(image.dataset.safelensInpaintMaskBlurRadius || 3.0),
    execute: true,
    output_stem: image.dataset.safelensInpaintOutputStem || "extension_medium_risk_cleanup",
    report_path: image.dataset.safelensInpaintReportPath || "reports/generation_service/extension_medium_risk_cleanup_report.json"
  };
}

async function requestLocalInpaintingResearch(image, riskDecision) {
  const startedAt = performance.now();
  const payload = buildLocalInpaintingRequest(image, riskDecision);

  const lock = await tryAcquireInpaintingClientLock(payload.output_stem || "local_inpainting_research");

  if (!lock.acquired) {
    throw new Error("client_inpainting_busy:" + (lock.reason || "lock_active"));
  }

  try {
    const response = await fetchWithTimeout(
      GENERATION_SERVICE_BASE_URL + "/v1/inpaint-local-research",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      },
      GENERATION_SERVICE_TIMEOUT_MS
    );

    if (!response.ok) {
      const text = await response.text();
      throw new Error("generation_service_inpaint_http_" + response.status + ": " + text.slice(0, 300));
    }

    const data = await response.json();

    if (!data.output_url) {
      throw new Error("generation_service_missing_inpainting_output_url");
    }

    return {
      status: data.status || "completed",
      backend: data.backend || "local_inpainting_research",
      imageUrl: data.output_url,
      outputUrl: data.output_url,
      reportPath: data.report_path || "none",
      latencyMs: Math.round(performance.now() - startedAt),
      payload,
      raw: data
    };
  } finally {
    await releaseInpaintingClientLock(lock.token);
  }
}

function applyGeneratedReplacement(image, imageUrl) {
  if (!image.dataset.safelensOriginalSrc) {
    image.dataset.safelensOriginalSrc = image.getAttribute("src") || "";
  }

  if (!image.dataset.safelensOriginalSrcset) {
    image.dataset.safelensOriginalSrcset = image.getAttribute("srcset") || "";
  }

  image.removeAttribute("srcset");
  image.src = imageUrl;
  image.dataset.safelensReplaced = "true";
  image.dataset.safelensReplacementUrl = imageUrl;
  image.title = "SafeLens visual replacement";
  image.style.display = image.dataset.safelensOriginalDisplay || "";
}

async function processImage(image) {
  if (!shouldProcessImage(image)) {
    return;
  }

  processedImages.add(image);

  const src = getImageSrc(image);
  const card = createTemporaryCard(image);

  await updateStats((stats) => ({
    ...stats,
    scanned: stats.scanned + 1,
    lastStatus: "checking"
  }));

  try {
    const response = await chrome.runtime.sendMessage({
      type: "SAFELENS_CLASSIFY_IMAGE",
      payload: { src }
    });

    const prediction = response?.payload?.prediction || {};

    const restrictedDetectorDecision = shouldRequestRestrictedDetector(image);

    let restrictedDetectorResult = {
      requested: restrictedDetectorDecision.shouldRun,
      ok: false,
      failed: false,
      status: restrictedDetectorDecision.shouldRun ? "requested" : "not_requested",
      model: "none",
      reason: restrictedDetectorDecision.reason,
      latencyMs: 0,
      predictionCount: 0,
      targetHitCount: 0,
      targetHits: [],
      error: "none"
    };

    if (restrictedDetectorDecision.shouldRun) {
      try {
        const serviceResult = await requestRestrictedDetectorService(image, restrictedDetectorDecision);

        restrictedDetectorResult = {
          requested: true,
          ok: serviceResult.status === "completed",
          failed: serviceResult.status !== "completed",
          status: serviceResult.status || "completed",
          model: serviceResult.model_id || getRestrictedDetectorModelId(image),
          reason: restrictedDetectorDecision.reason,
          latencyMs: serviceResult.client_latency_ms || 0,
          predictionCount: serviceResult.prediction_count || 0,
          targetHitCount: countRestrictedServiceHits(serviceResult),
          targetHits: summarizeRestrictedDetectorServiceHits(normalizeRestrictedServiceHits(serviceResult)),
          error: "none"
        };
      } catch (error) {
        restrictedDetectorResult = {
          requested: true,
          ok: false,
          failed: true,
          status: "failed",
          model: getRestrictedDetectorModelId(image),
          reason: restrictedDetectorDecision.reason,
          latencyMs: 0,
          predictionCount: 0,
          targetHitCount: 0,
          targetHits: [],
          error: error instanceof Error ? error.message : "restricted_detector_service_failed"
        };
      }
    }

    let riskDecision = evaluateRiskPolicy(image, prediction);
    riskDecision = fuseRestrictedDetectorRiskDecision(riskDecision, restrictedDetectorResult);
    const restrictedCleanupDecision = shouldRequestRestrictedObjectCleanup(image, riskDecision, restrictedDetectorResult);
    const inpaintingDecision = shouldRequestLocalInpainting(image, riskDecision);
    const placeholderDecision = shouldRequestSafePlaceholder(image, prediction, riskDecision);

    const activeReplacement = restrictedCleanupDecision.shouldReplace
      ? {
          shouldReplace: true,
          mode: "bbox_guided_local_cleanup",
          endpoint: "/v1/inpaint-local-bbox",
          reason: restrictedCleanupDecision.reason
        }
      : (
          inpaintingDecision.shouldReplace
            ? {
                shouldReplace: true,
                mode: "local_inpainting_research",
                endpoint: "/v1/inpaint-local-research",
                reason: inpaintingDecision.reason
              }
            : {
                shouldReplace: placeholderDecision.shouldReplace,
                mode: placeholderDecision.shouldReplace ? "safe_placeholder" : "none",
                endpoint: placeholderDecision.shouldReplace ? "/v1/generate-placeholder" : "none",
                reason: placeholderDecision.reason
              }
        );

    let replacementResult = {
      requested: activeReplacement.shouldReplace,
      applied: false,
      failed: false,
      busy: false,
      status: "not_requested",
      backend: "none",
      imageUrl: "none",
      latencyMs: 0,
      mode: activeReplacement.shouldReplace ? activeReplacement.mode : "none",
      endpoint: activeReplacement.shouldReplace ? activeReplacement.endpoint : "none",
      reportPath: "none",
      reason: activeReplacement.reason,
      error: null
    };

    if (activeReplacement.shouldReplace) {
      try {
        card.textContent = activeReplacement.mode === "local_inpainting_research"
          ? "SafeLens running local object cleanup..."
          : "SafeLens generating local safe replacement...";

        const generated = activeReplacement.mode === "bbox_guided_local_cleanup"
          ? await requestRestrictedObjectCleanup(image, riskDecision)
          : (
              activeReplacement.mode === "local_inpainting_research"
                ? await requestLocalInpaintingResearch(image, riskDecision)
                : await requestSafePlaceholder(image, prediction, activeReplacement.reason)
            );

        applyGeneratedReplacement(image, generated.imageUrl);
        removeCard(card);

        replacementResult = {
          requested: true,
          applied: true,
          failed: false,
          busy: false,
          status: generated.status,
          backend: generated.backend,
          imageUrl: generated.imageUrl,
          latencyMs: generated.latencyMs,
          mode: activeReplacement.mode,
          endpoint: activeReplacement.endpoint,
          reportPath: generated.reportPath || "none",
          reason: activeReplacement.reason,
          error: null
        };
      } catch (error) {
        const errorMessage = error instanceof Error
          ? error.name + ": " + error.message
          : "generation_service_failed";

        const isBusy =
          errorMessage.includes("client_inpainting_busy") ||
          errorMessage.includes("generation_service_inpaint_http_409");

        replacementResult = {
          requested: true,
          applied: false,
          failed: !isBusy,
          busy: isBusy,
          status: isBusy ? "busy" : "failed",
          backend: activeReplacement.mode === "local_inpainting_research" ? "local_inpainting_research" : "safe_placeholder",
          imageUrl: "none",
          latencyMs: 0,
          mode: activeReplacement.mode,
          endpoint: activeReplacement.endpoint,
          reportPath: "none",
          reason: activeReplacement.reason,
          error: errorMessage
        };

        restoreImage(image, card);
      }
    } else {
      restoreImage(image, card);
    }

    await updateStats((stats) => {
      const next = { ...stats };

      next.lastStatus = prediction.status || "unknown";
      next.lastAccess = prediction.imageAccess || "unknown";
      next.lastAccessReason = prediction.imageAccessReason || "none";
      next.lastDecodeReason = prediction.decodeReason || "none";
      next.lastLatencyMs = prediction.latencyMs || 0;

      if (prediction.imageAccess === "readable") {
        next.readable += 1;
      }

      if (prediction.imageAccess === "unreadable") {
        next.unreadable += 1;
      }

      if (prediction.decoded) {
        next.decoded += 1;
      }

      if (prediction.tensorReady) {
        next.tensorReady += 1;
      }

      if (prediction.inferenceOk) {
        next.inferenceOk += 1;
      } else {
        next.inferenceFailed += 1;
      }

      if (prediction.detectorOk) {
        next.detectorOk += 1;
      } else {
        next.detectorFailed += 1;
      }

      next.lastInferenceReason = prediction.inferenceReason || "none";
      next.lastModelVersion = prediction.modelVersion || "none";
      next.lastProbabilities = prediction.probabilities || null;

      next.lastDetectorReason = prediction.detectorReason || "none";
      next.lastDetectorModelVersion = prediction.detectorModelVersion || "none";
      next.lastDetectionCount = prediction.detectionCount || 0;
      next.lastDetections = summarizeDetections(prediction.detections);
      next.lastRestrictedHitCount = prediction.restrictedHitCount || 0;
      next.lastRestrictedHits = summarizeRestrictedHits(prediction.restrictedHits);

      next.lastPolicyRecommendedAction = prediction.policyRecommendedAction || "none";
      next.lastPolicyCategory = prediction.policyCategory || "none";
      next.lastPolicySeverity = prediction.policySeverity || "none";
      next.lastPolicyReason = prediction.policyReason || "none";
      next.lastPolicyEvidence = prediction.policyEvidence || null;
      next.lastGenerationMetadata = prediction.generationMetadata || null;

      next.lastRiskLevel = riskDecision.riskLevel || "none";
      next.lastRiskSource = riskDecision.riskSource || "none";
      next.lastRiskAction = riskDecision.recommendedAction || "none";
      next.lastRiskReason = riskDecision.reason || "none";
      next.lastRiskTargetLabel = riskDecision.targetLabel || "none";
      next.lastRiskTargetConfidence = riskDecision.targetConfidence || 0;

      if (restrictedDetectorResult.ok) {
        next.restrictedDetectorOk += 1;
      } else if (restrictedDetectorResult.failed) {
        next.restrictedDetectorFailed += 1;
      }

      next.lastRestrictedDetectorStatus = restrictedDetectorResult.status || "none";
      next.lastRestrictedDetectorModel = restrictedDetectorResult.model || "none";
      next.lastRestrictedDetectorReason = restrictedDetectorResult.reason || "none";
      next.lastRestrictedDetectorLatencyMs = restrictedDetectorResult.latencyMs || 0;
      next.lastRestrictedDetectorPredictionCount = restrictedDetectorResult.predictionCount || 0;
      next.lastRestrictedDetectorTargetHitCount = restrictedDetectorResult.targetHitCount || 0;
      next.lastRestrictedDetectorHits = restrictedDetectorResult.targetHits || [];
      next.lastRestrictedDetectorError = restrictedDetectorResult.error || "none";
      next.lastRestrictedCleanupDecisionReason = restrictedCleanupDecision.reason || "none";

      next.lastReplacementMode = replacementResult.mode || "none";
      next.lastReplacementReason = replacementResult.error || replacementResult.reason;
      next.lastGenerationServiceStatus = replacementResult.status;
      next.lastGenerationBackend = replacementResult.backend;
      next.lastGeneratedImageUrl = replacementResult.imageUrl;
      next.lastGenerationLatencyMs = replacementResult.latencyMs;
      next.lastGenerationEndpoint = replacementResult.endpoint || "none";
      next.lastInpaintingReportPath = replacementResult.reportPath || "none";
      next.lastInpaintingOutputUrl =
        replacementResult.mode === "local_inpainting_research" ||
        replacementResult.mode === "bbox_guided_local_cleanup"
          ? replacementResult.imageUrl
          : "none";

      if (replacementResult.applied) {
        next.replaced += 1;
        next.generatedReplacementApplied += 1;
        next.generationServiceOk += 1;
      } else {
        next.restored += 1;

        if (replacementResult.busy) {
          next.generatedReplacementSkipped += 1;
          next.generationServiceBusy += 1;
        } else if (replacementResult.failed) {
          next.generatedReplacementSkipped += 1;
          next.generationServiceFailed += 1;
        }
      }

      return next;
    });
  } catch (error) {
    await updateStats((stats) => ({
      ...stats,
      errors: stats.errors + 1,
      restored: stats.restored + 1,
      lastStatus: error instanceof Error ? error.name + ": " + error.message : "content_script_error"
    }));

    restoreImage(image, card);
  }
}

function scanImages() {
  const images = Array.from(document.images);
  images.forEach((image) => {
    processImage(image);
  });
}

function startObserver() {
  const observer = new MutationObserver(() => {
    scanImages();
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true
  });
}

console.log("[SafeLens] Risk-based replacement policy content script loaded.");
scanImages();
startObserver();
