// Checkpoint 1 frontend: stage list (all "Not started") and live backend health check.
// Nothing here is simulated: the status indicator reflects the real /api/health response.
"use strict";

const STAGES = [
  "Source encoding (Huffman)",
  "Binary representation",
  "Compression (LZ77)",
  "Finite field GF(2^8)",
  "AES-128 encryption",
  "RSA key exchange",
  "Packetization",
  "CRC-32 error detection",
  "Routing (Dijkstra)",
  "Channel transmission",
  "Error correction",
  "AES decryption",
  "Decoding and comparison",
];

function renderStages() {
  const list = document.getElementById("stage-list");
  list.replaceChildren(
    ...STAGES.map((name) => {
      const li = document.createElement("li");
      li.className = "stage-item";
      const label = document.createElement("span");
      label.textContent = name;
      const state = document.createElement("span");
      state.className = "stage-state";
      state.textContent = "Not started";
      li.append(label, state);
      return li;
    })
  );
}

function setStatus(kind, text) {
  const box = document.getElementById("backend-status");
  box.className = `status status-${kind}`;
  document.getElementById("backend-status-text").textContent = text;
  document.getElementById("retry-btn").hidden = kind !== "bad";
}

async function checkBackend() {
  setStatus("pending", "Checking backend…");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    setStatus("ok", "Backend connected");
    document.getElementById("fact-version").textContent = data.version;
  } catch (err) {
    setStatus("bad", "Backend unreachable");
    document.getElementById("fact-version").textContent = "unknown";
  }
}

document.getElementById("retry-btn").addEventListener("click", checkBackend);
renderStages();
checkBackend();
