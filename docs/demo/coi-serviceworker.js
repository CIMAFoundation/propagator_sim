// Cross-origin isolation for the demo on hosts that cannot set response
// headers (GitHub Pages). SharedArrayBuffer, which the multi-threaded wasm
// bundle needs, is only available to cross-origin-isolated pages, i.e. pages
// served with Cross-Origin-Opener-Policy and Cross-Origin-Embedder-Policy.
// This service worker adds both headers to every same-scope response it
// serves. `credentialless` (rather than `require-corp`) keeps cross-origin
// subresources of the surrounding docs pages loading; browsers that do not
// know it simply stay non-isolated and the page falls back to the
// single-threaded bundle.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => event.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.cache === "only-if-cached" && request.mode !== "same-origin") {
    return;
  }
  event.respondWith(
    fetch(request).then((response) => {
      if (response.status === 0) {
        // opaque cross-origin response: cannot and need not be rewritten
        return response;
      }
      const headers = new Headers(response.headers);
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");
      return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      });
    })
  );
});
