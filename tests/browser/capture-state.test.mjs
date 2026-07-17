import assert from 'node:assert/strict';
import test from 'node:test';

import {
  appendCaptureChunk,
  attachUpload,
  bindCapture,
  buildCaptureForm,
  createChunkCollector,
  finalizeCapture,
} from '../../webui/static/capture-state.mjs';

const prompt = {id: 'sec13_001__satraj', corpus_version: '2026.07.16'};
const authorization = {
  capture_token: 'signed-token',
  prompt_id: prompt.id,
  corpus_version: prompt.corpus_version,
  generation: 4,
  issued_at: 1000,
  expires_at: 1300,
};

test('navigation cannot relabel an in-flight microphone capture', () => {
  const capture = bindCapture(prompt, authorization);
  appendCaptureChunk(capture, new Blob(['voice'], {type: 'audio/webm'}));
  finalizeCapture(capture, 'audio/webm');

  const currentPromptAfterNavigation = {id: 'sec13_001__piranesi', corpus_version: '2026.07.16'};
  assert.notEqual(currentPromptAfterNavigation.id, capture.promptId);
  const form = buildCaptureForm(capture, true);
  assert.equal(form.get('prompt_id'), prompt.id);
  assert.equal(form.get('capture_token'), authorization.capture_token);
  assert.equal(form.get('source'), 'browser-mic');
});

test('an authorization for another prompt is rejected before recording', () => {
  assert.throws(
    () => bindCapture({...prompt, id: 'sec13_001__piranesi'}, authorization),
    /does not match/,
  );
});

test('uploads retain their original prompt binding', () => {
  const capture = bindCapture(prompt, authorization, 'upload');
  attachUpload(capture, new Blob(['wave'], {type: 'audio/wav'}));
  const form = buildCaptureForm(capture, false);
  assert.equal(form.get('prompt_id'), prompt.id);
  assert.equal(form.get('denoise'), 'false');
  assert.equal(form.get('source'), 'upload');
});

test('room-tone chunks cannot leak into voice chunks', async () => {
  const capture = bindCapture(prompt, authorization);
  appendCaptureChunk(capture, new Blob(['voice']));
  const roomTone = createChunkCollector();
  roomTone.push(new Blob(['room']));
  const roomBlob = roomTone.toBlob();
  const voiceBlob = finalizeCapture(capture);
  assert.equal(await roomBlob.text(), 'room');
  assert.equal(await voiceBlob.text(), 'voice');
  assert.equal(roomTone.size, 0);
});
