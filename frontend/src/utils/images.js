/**
 * Build a PNG File from raw base64 string (without data URI prefix).
 * @param {string} base64NoPrefix - base64-encoded PNG bytes
 */
export function pngBase64ToFile(base64NoPrefix, filename = "heatmap.png") {
  const binary = atob(base64NoPrefix);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new File([bytes], filename, { type: "image/png" });
}
