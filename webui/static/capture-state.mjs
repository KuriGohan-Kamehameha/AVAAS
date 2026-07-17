const MAX_CAPTURE_TOKEN_CHARS = 2048;
const ALLOWED_SOURCES = new Set(['browser-mic', 'upload']);

function requireText(value, field, maximum = 2048) {
  if (typeof value !== 'string' || value.length < 1 || value.length > maximum) {
    throw new TypeError(`invalid ${field}`);
  }
  return value;
}

function requireInteger(value, field) {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError(`invalid ${field}`);
  return value;
}

function requireBlob(value, field) {
  if (!(value instanceof Blob) || value.size < 1) throw new TypeError(`invalid ${field}`);
  return value;
}

export function bindCapture(prompt, authorization, source = 'browser-mic') {
  if (!prompt || typeof prompt !== 'object' || !authorization || typeof authorization !== 'object') {
    throw new TypeError('capture binding requires prompt and authorization objects');
  }
  const promptId = requireText(prompt.id, 'prompt id', 128);
  const corpusVersion = requireText(prompt.corpus_version, 'corpus version', 64);
  if (authorization.prompt_id !== promptId || authorization.corpus_version !== corpusVersion) {
    throw new TypeError('capture authorization does not match prompt snapshot');
  }
  if (!ALLOWED_SOURCES.has(source)) throw new TypeError('invalid capture source');
  const captureToken = requireText(
    authorization.capture_token,
    'capture token',
    MAX_CAPTURE_TOKEN_CHARS,
  );
  const issuedAt = requireInteger(authorization.issued_at, 'capture issued time');
  const expiresAt = requireInteger(authorization.expires_at, 'capture expiry time');
  if (expiresAt <= issuedAt) throw new TypeError('invalid capture lifetime');
  return {
    promptId,
    corpusVersion,
    captureToken,
    generation: requireInteger(authorization.generation, 'capture generation'),
    issuedAt,
    expiresAt,
    source,
    chunks: [],
    blob: null,
    mimeType: '',
    recorder: null,
  };
}

export function appendCaptureChunk(capture, chunk) {
  if (!capture || !Array.isArray(capture.chunks)) throw new TypeError('invalid capture');
  capture.chunks.push(requireBlob(chunk, 'capture chunk'));
}

export function finalizeCapture(capture, mimeType = 'audio/webm') {
  if (!capture || !Array.isArray(capture.chunks) || capture.chunks.length < 1) {
    throw new TypeError('capture has no chunks');
  }
  capture.mimeType = requireText(mimeType || 'audio/webm', 'capture MIME type', 128);
  capture.blob = new Blob(capture.chunks, {type: capture.mimeType});
  capture.chunks.length = 0;
  return requireBlob(capture.blob, 'capture blob');
}

export function attachUpload(capture, file) {
  capture.blob = requireBlob(file, 'upload');
  capture.mimeType = requireText(file.type || 'application/octet-stream', 'upload MIME type', 128);
  return capture;
}

export function buildCaptureForm(capture, denoise) {
  if (!capture || !ALLOWED_SOURCES.has(capture.source)) throw new TypeError('invalid capture');
  const blob = requireBlob(capture.blob, 'capture blob');
  const type = capture.mimeType || blob.type || 'audio/webm';
  const extension = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : type.includes('wav') ? 'wav' : 'webm';
  const form = new FormData();
  form.append('file', blob, `take.${extension}`);
  form.append('prompt_id', requireText(capture.promptId, 'prompt id', 128));
  form.append('capture_token', requireText(capture.captureToken, 'capture token', MAX_CAPTURE_TOKEN_CHARS));
  form.append('denoise', denoise ? 'true' : 'false');
  form.append('source', capture.source);
  return form;
}

export function createChunkCollector() {
  const chunks = [];
  return {
    push(chunk) {
      chunks.push(requireBlob(chunk, 'collector chunk'));
    },
    toBlob(mimeType = 'audio/webm') {
      if (chunks.length < 1) throw new TypeError('collector has no chunks');
      const blob = new Blob(chunks, {type: requireText(mimeType, 'collector MIME type', 128)});
      chunks.length = 0;
      return requireBlob(blob, 'collector blob');
    },
    get size() {
      return chunks.length;
    },
  };
}
