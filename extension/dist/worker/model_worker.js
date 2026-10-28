async function ensureOffscreenDocument() {
  const offscreenUrl = chrome.runtime.getURL("offscreen/offscreen.html");

  if (chrome.runtime.getContexts) {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [offscreenUrl]
    });

    if (contexts.length > 0) {
      return;
    }
  }

  await chrome.offscreen.createDocument({
    url: "offscreen/offscreen.html",
    reasons: ["BLOBS"],
    justification: "SafeLens runs local ONNX visual safety inference on page images in observe-only mode."
  });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (!message || message.type !== "SAFELENS_CLASSIFY_IMAGE") {
    return false;
  }

  ensureOffscreenDocument()
    .then(() => {
      return chrome.runtime.sendMessage({
        target: "offscreen",
        type: "SAFELENS_OFFSCREEN_INFER",
        payload: message.payload || {}
      });
    })
    .then((response) => {
      sendResponse(response);
    })
    .catch((error) => {
      sendResponse({
        type: "SAFELENS_CLASSIFY_IMAGE_RESULT",
        payload: {
          prediction: {
            status: "worker_router_failed",
            reason: error instanceof Error ? error.name + ": " + error.message : "worker_router_failed",
            policyMode: "observe_only"
          },
          decision: {
            action: "allow",
            reason: "Fail-open observe-only mode because worker/offscreen routing failed."
          }
        }
      });
    });

  return true;
});

console.log("[SafeLens] Service worker router loaded.");
