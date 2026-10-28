function setText(id, value) {
  const element = document.getElementById(id);

  if (element) {
    element.textContent = String(value);
  }
}

function formatProbabilities(value) {
  if (!Array.isArray(value)) {
    return "none";
  }

  return "[" + value.map((x) => Number(x).toFixed(6)).join(", ") + "]";
}

function formatDetections(value) {
  if (!Array.isArray(value) || value.length === 0) {
    return "none";
  }

  return value
    .map((item) => `${item.label}:${item.confidence}`)
    .join(" | ");
}

function formatRestrictedHits(value) {
  if (!Array.isArray(value) || value.length === 0) {
    return "none";
  }

  return value
    .map((item) => `${item.label}:${item.category}:${item.confidence}`)
    .join(" | ");
}



function formatRestrictedDetectorHits(value) {
  if (!Array.isArray(value) || value.length === 0) {
    return "none";
  }

  return value
    .map((item) => {
      const label = item.label ?? "unknown";
      const confidence = Number(item.confidence ?? 0).toFixed(3);
      return `${label}:${confidence}`;
    })
    .join(" | ");
}

function formatReplacementDisplay(value) {
  const raw = String(value ?? "none");

  const mapping = {
    "local_inpainting_research": "local object cleanup",
    "bbox_guided_local_cleanup": "bbox-guided local cleanup",
    "safe_placeholder": "safe visual replacement",
    "demo_force_local_inpainting_research": "visual filter preview request",
    "demo_force_placeholder_attribute": "safe replacement preview request",
    "/v1/inpaint-local-research": "local object cleanup service",
    "/v1/inpaint-local-bbox": "bbox cleanup service",
    "/v1/generate-placeholder": "safe replacement service"
  };

  return mapping[raw] || raw;
}

function renderStats(stats) {
  const s = stats || {};

  setText("modelStatus", "combined_observe_only");

  setText("lastPolicyRecommendedAction", s.lastPolicyRecommendedAction ?? "none");
  setText("lastPolicyCategory", s.lastPolicyCategory ?? "none");
  setText("lastPolicySeverity", s.lastPolicySeverity ?? "none");
  setText("lastPolicyReason", s.lastPolicyReason ?? "none");

  setText("lastRiskLevel", s.lastRiskLevel ?? "none");
  setText("lastRiskSource", s.lastRiskSource ?? "none");
  setText("lastRiskAction", s.lastRiskAction ?? "none");
  setText("lastRiskReason", s.lastRiskReason ?? "none");
  setText("lastRiskTargetLabel", s.lastRiskTargetLabel ?? "none");
  setText("lastRiskTargetConfidence", Number(s.lastRiskTargetConfidence ?? 0).toFixed(4));

  setText("generationServiceOk", s.generationServiceOk ?? 0);
  setText("generationServiceFailed", s.generationServiceFailed ?? 0);
  setText("generationServiceBusy", s.generationServiceBusy ?? 0);
  setText("generatedReplacementApplied", s.generatedReplacementApplied ?? 0);
  setText("generatedReplacementSkipped", s.generatedReplacementSkipped ?? 0);
  setText("lastGenerationServiceStatus", s.lastGenerationServiceStatus ?? "none");
  setText("lastGenerationBackend", formatReplacementDisplay(s.lastGenerationBackend ?? "none"));

  setText("restrictedDetectorOk", s.restrictedDetectorOk ?? 0);
  setText("restrictedDetectorFailed", s.restrictedDetectorFailed ?? 0);
  setText("lastRestrictedDetectorStatus", s.lastRestrictedDetectorStatus ?? "none");
  setText("lastRestrictedDetectorModel", s.lastRestrictedDetectorModel ?? "none");
  setText("lastRestrictedDetectorReason", s.lastRestrictedDetectorReason ?? "none");
  setText("lastRestrictedDetectorLatencyMs", `${s.lastRestrictedDetectorLatencyMs ?? 0} ms`);
  setText("lastRestrictedDetectorPredictionCount", s.lastRestrictedDetectorPredictionCount ?? 0);
  setText("lastRestrictedDetectorTargetHitCount", s.lastRestrictedDetectorTargetHitCount ?? 0);
  setText("lastRestrictedDetectorHits", formatRestrictedDetectorHits(s.lastRestrictedDetectorHits));
  setText("lastRestrictedDetectorError", s.lastRestrictedDetectorError ?? "none");

  setText("lastReplacementMode", formatReplacementDisplay(s.lastReplacementMode ?? "none"));
  setText("lastReplacementReason", formatReplacementDisplay(s.lastReplacementReason ?? "none"));
  setText("lastGenerationLatencyMs", `${s.lastGenerationLatencyMs ?? 0} ms`);
  setText("lastGeneratedImageUrl", s.lastGeneratedImageUrl ?? "none");
  setText("lastGenerationEndpoint", formatReplacementDisplay(s.lastGenerationEndpoint ?? "none"));
  setText("lastInpaintingReportPath", s.lastInpaintingReportPath ?? "none");
  setText("lastInpaintingOutputUrl", s.lastInpaintingOutputUrl ?? "none");

  setText("scanned", s.scanned ?? 0);
  setText("readable", s.readable ?? 0);
  setText("decoded", s.decoded ?? 0);
  setText("tensorReady", s.tensorReady ?? 0);
  setText("restored", s.restored ?? 0);
  setText("replaced", s.replaced ?? 0);
  setText("errors", s.errors ?? 0);
  setText("lastLatencyMs", `${s.lastLatencyMs ?? 0} ms`);

  setText("inferenceOk", s.inferenceOk ?? 0);
  setText("inferenceFailed", s.inferenceFailed ?? 0);
  setText("lastModelVersion", s.lastModelVersion ?? "none");
  setText("lastInferenceReason", s.lastInferenceReason ?? "none");
  setText("lastProbabilities", formatProbabilities(s.lastProbabilities));

  setText("detectorOk", s.detectorOk ?? 0);
  setText("detectorFailed", s.detectorFailed ?? 0);
  setText("lastDetectorModelVersion", s.lastDetectorModelVersion ?? "none");
  setText("lastDetectorReason", s.lastDetectorReason ?? "none");
  setText("lastDetectionCount", s.lastDetectionCount ?? 0);
  setText("lastDetections", formatDetections(s.lastDetections));
  setText("lastRestrictedHitCount", s.lastRestrictedHitCount ?? 0);
  setText("lastRestrictedHits", formatRestrictedHits(s.lastRestrictedHits));
}

function loadStats() {
  chrome.storage.local.get("safelensStats", (data) => {
    renderStats(data.safelensStats);
  });
}

function resetStats() {
  chrome.storage.local.remove([STORAGE_KEY, INPAINTING_CLIENT_LOCK_KEY], () => {
    renderStats(null);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  loadStats();

  const refreshButton = document.getElementById("refreshButton");
  if (refreshButton) {
    refreshButton.addEventListener("click", loadStats);
  }

  const resetButton = document.getElementById("resetButton");
  if (resetButton) {
    resetButton.addEventListener("click", resetStats);
  }
});
