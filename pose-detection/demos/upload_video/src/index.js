/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import * as mpPose from '@mediapipe/pose';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

tfjsWasm.setWasmPaths(
    `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${
        tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import {setupStats} from './stats_panel';
import {Context} from './camera';
import {setupDatGui} from './option_panel';
import {STATE} from './params';
import {setBackendAndEnvFlags} from './util';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let currentTime = 0.0, lastTime = 0;
let globalPoses;
let rightAnkleX, leftAnkleX, lastRightAnkleX, lastLeftAnkleX, leftAnkleSpeed, rightAnkleSpeed;
let leftFiveSpeeds = [], rightFiveSpeeds = [];

let rafId;
const statusElement = document.getElementById('status');
const frameElement = document.getElementById('frameNumber');

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: {width: 500, height: 500},
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
            STATE.model, {runtime, modelType: STATE.modelConfig.type});
      }
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
          posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
          posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, {modelType});
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
        1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

function printPose() {
  console.log('keypoints');
  console.log(globalPoses[0].keypoints);
  console.log('left_ankle');
  console.log(globalPoses[0].keypoints[15]);
  console.log('right_ankle');
  console.log(globalPoses[0].keypoints[16]);
  console.log('score: ' + globalPoses[0].score);
  console.log('currentTime: ' + currentTime);
  trackSpeeds();
}

function trackSpeeds() {
  leftAnkleX = globalPoses[0].keypoints[15].x;
  rightAnkleX = globalPoses[0].keypoints[16].x;
  leftAnkleSpeed = (leftAnkleX - lastLeftAnkleX)/(currentTime - lastTime);
  rightAnkleSpeed = (rightAnkleX - lastRightAnkleX)/(currentTime - lastTime);
  leftFiveSpeeds.push(leftAnkleSpeed);
  if (leftFiveSpeeds.length > 10) {
    leftFiveSpeeds.shift();
  }
  rightFiveSpeeds.push(rightAnkleSpeed);
  if (rightFiveSpeeds.length > 10) {
    rightFiveSpeeds.shift();
  }
  console.log('leftSpeed: ' + leftAnkleSpeed);
  console.log('rightSpeed: ' + rightAnkleSpeed);
  lastLeftAnkleX = leftAnkleX;
  lastRightAnkleX= rightAnkleX;
}


function allSpeedsSlow() {
  console.log('rightFiveSpeeds');
  console.log(rightFiveSpeeds);
  console.log('leftFiveSpeeds');
  console.log(leftFiveSpeeds);
  let allSlowRight = true;
  let allSlowLeft = true;
  for (let i = 0; i < rightFiveSpeeds.length; i++) {
    if (Math.abs(rightFiveSpeeds[i]) > 200) {
      allSlowRight = false;
    }
  }
  for (let i = 0; i < leftFiveSpeeds.length; i++) {
    if (Math.abs(leftFiveSpeeds[i]) > 200) {
      allSlowLeft = false;
    }
  }
  return allSlowRight && allSlowLeft;
}
async function renderResult() {
  // FPS only counts the time it takes to finish estimatePoses.
  beginEstimatePosesStats();

  const poses = await detector.estimatePoses(
      camera.video,
      {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});

  globalPoses = poses;
  endEstimatePosesStats();

  camera.drawCtx();

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);
    trackSpeeds();
    if (allSpeedsSlow()) {
      console.log('cut at frame: ' + currentTime);
    };
  }
}

async function checkUpdate() {
  await checkGuiUpdate();

  requestAnimationFrame(checkUpdate);
};

async function updateVideo(event) {
  // Clear reference to any previous uploaded video.
  URL.revokeObjectURL(camera.video.currentSrc);
  const file = event.target.files[0];
  camera.source.src = URL.createObjectURL(file);

  // Wait for video to be loaded.
  camera.video.load();
  await new Promise((resolve) => {
    camera.video.onloadeddata = () => {
      resolve(video);
    };
  });

  camera.video.volume = 0;

  const videoWidth = camera.video.videoWidth;
  const videoHeight = camera.video.videoHeight;
  // Must set below two lines, otherwise video element doesn't show.
  camera.video.width = videoWidth;
  camera.video.height = videoHeight;
  camera.canvas.width = videoWidth;
  camera.canvas.height = videoHeight;

  statusElement.innerHTML = 'Video is loaded.';
}

async function pause() {
  lastTime = currentTime;
  currentTime = video.currentTime;
  video.pause();
  camera.video.pause();
  // camera.mediaRecorder.pause();
}

async function nextFrame() {
  lastTime = currentTime;
  currentTime += 1/24;
  video.currentTime = currentTime;

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  frameElement.innerHTML = 'Frame ' + (currentTime * 24);
  await renderResult();
  // printPose();
}

async function previousFrame() {
  lastTime = currentTime;
  currentTime = currentTime - 1/24;
  if (currentTime < 0) {
    currentTime = 0;
  }
  video.currentTime = currentTime;

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  frameElement.innerHTML = 'Frame ' + (currentTime * 24);
  await renderResult();
  printPose();
}

async function runFrame() {
  if (video.paused) {
    // video has finished.
    // camera.mediaRecorder.stop();
    // camera.clearCtx();
    // camera.video.style.visibility = 'visible';
    // return;
  }
  lastTime = currentTime;
  currentTime = video.currentTime;
  frameElement.innerHTML = 'Frame ' + (currentTime * 24);
  await renderResult();
  rafId = requestAnimationFrame(runFrame);
}

async function warmup() {
  statusElement.innerHTML = 'Warming up model.';

  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
        tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    await detector.estimatePoses(
        warmUpTensor,
        {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }
  camera.video.style.visibility = 'hidden';
}

async function run() {
  statusElement.innerHTML = 'Warming up model.';

  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
        tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    await detector.estimatePoses(
        warmUpTensor,
        {maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false});
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }

  camera.video.style.visibility = 'hidden';
  video.pause();
  video.currentTime = currentTime || 0;
  video.play();
  // camera.mediaRecorder.start();

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  await runFrame();
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);
  stats = setupStats();
  detector = await createDetector();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);

  const runButton = document.getElementById('runButton');
  runButton.onclick = run;

  const pauseButton = document.getElementById('pauseButton');
  pauseButton.onclick = pause;

  const warmupButton = document.getElementById('warmupButton');
  warmupButton.onclick = warmup;

  const nextFrameButton = document.getElementById('nextFrameButton');
  nextFrameButton.onclick = nextFrame;

  const previousFrameButton = document.getElementById('previousFrameButton');
  previousFrameButton.onclick = previousFrame;

  const printPoseButton = document.getElementById('printPoseButton');
  printPoseButton.onclick = printPose;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;

  checkUpdate();
};

app();
