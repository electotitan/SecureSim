// ============================================================================
// app.js — Signal Path frontend logic.
// Vanilla JS + fetch() against the FastAPI backend; D3 (CDN) for the tree
// and graph visualizations only, per the project's "rendering only" rule.
// ============================================================================

const STAGE_SHORT_LABELS = [
  "Huffman", "Binary", "LZ77", "AES/RSA", "Packets",
  "CRC-32", "Routing", "Channel", "ECC", "Decrypt", "Decode",
];

let currentTrace = [];
let currentResult = null;
let currentStageIndex = 0;

// ----------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------
function $(id) { return document.getElementById(id); }

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function kvGrid(pairs) {
  return `<div class="kv-grid">${pairs.map(([k, v, cls]) => `
    <div class="kv">
      <div class="k">${escapeHtml(k)}</div>
      <div class="v${cls ? " " + cls : ""}">${v}</div>
    </div>`).join("")}</div>`;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data && data.detail ? data.detail : res.statusText;
    throw new Error(detail);
  }
  return data;
}

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

// ============================================================================
// Controls wiring
// ============================================================================
function initControls() {
  const noiseSlider = $("noiseSlider");
  const noiseValue = $("noiseValue");
  noiseSlider.addEventListener("input", () => { noiseValue.textContent = Number(noiseSlider.value).toFixed(3); });

  $("runBtn").addEventListener("click", runPipeline);
  $("prevBtn").addEventListener("click", () => showStage(currentStageIndex - 1));
  $("nextBtn").addEventListener("click", () => showStage(currentStageIndex + 1));

  loadTopologyIntoRouteSelects();
}

async function loadTopologyIntoRouteSelects() {
  try {
    const topo = await getJSON("/api/routing/topology");
    const src = $("routeSource"), tgt = $("routeTarget");
    src.innerHTML = topo.nodes.map((n) => `<option value="${n}">${n}</option>`).join("");
    tgt.innerHTML = topo.nodes.map((n) => `<option value="${n}">${n}</option>`).join("");
    src.value = topo.nodes[0];
    tgt.value = topo.nodes[topo.nodes.length - 1];
  } catch (e) {
    // topology endpoint unreachable -- leave selects empty, orchestrator falls back to defaults
  }
}

// ============================================================================
// Run the full pipeline
// ============================================================================
async function runPipeline() {
  const runBtn = $("runBtn");
  const status = $("runStatus");
  runBtn.disabled = true;
  status.textContent = "Transmitting…";
  $("resultBanner").hidden = true;

  const payload = {
    message: $("messageInput").value || "Hello, world.",
    noise_probability: Number($("noiseSlider").value),
    error_correction_scheme: $("eccSelect").value,
    channel_type: $("channelSelect").value,
    topology_source: $("routeSource").value || "A",
    topology_target: $("routeTarget").value || "F",
  };

  try {
    const result = await postJSON("/api/pipeline/run", payload);
    currentResult = result;
    currentTrace = result.trace;
    status.textContent = "";
    renderSignalPath();
    renderResultBanner(result);
    $("stepControls").hidden = false;
    showStage(0);
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    runBtn.disabled = false;
  }
}

function renderResultBanner(result) {
  const banner = $("resultBanner");
  banner.hidden = false;
  banner.className = "result-banner " + (result.success ? "success" : "failure");
  if (result.success) {
    banner.innerHTML = `<strong>Message recovered successfully</strong>
      The receiver's decoded text matches the original, byte for byte, after all eleven stages.`;
  } else {
    let diffView = "";
    if (result.recovered_message !== null && result.recovered_message !== undefined) {
      diffView = `<div class="diff-view">original:  ${escapeHtml(result.original_message)}
recovered: ${escapeHtml(result.recovered_message)}
mismatched positions: ${result.diff_positions.length ? result.diff_positions.join(", ") : "(recovered text differs in length)"}</div>`;
    }
    banner.innerHTML = `<strong>Transmission failed — errors exceeded correction capacity</strong>
      ${escapeHtml(result.failure_reason || "The message could not be recovered.")}${diffView}`;
  }
}

// ============================================================================
// Signal path nav (11-stage circuit trace)
// ============================================================================
function renderSignalPath() {
  const nav = $("signalPath");
  nav.innerHTML = "";
  currentTrace.forEach((stage, i) => {
    if (i > 0) {
      const connector = document.createElement("div");
      connector.className = "stage-connector";
      nav.appendChild(connector);
    }
    const btn = document.createElement("button");
    btn.className = "stage-node";
    btn.innerHTML = `<span class="dot"></span><span class="label">${STAGE_SHORT_LABELS[i] || stage.stage_name}</span>`;
    btn.addEventListener("click", () => showStage(i));
    btn.dataset.index = i;
    nav.appendChild(btn);
  });
}

function updateSignalPathActive(index) {
  document.querySelectorAll(".stage-node").forEach((node) => {
    const i = Number(node.dataset.index);
    node.classList.toggle("active", i === index);
    node.classList.toggle("done", i < index);
  });
}

// ============================================================================
// Step-through: show one stage's detail
// ============================================================================
function showStage(index) {
  if (!currentTrace.length) return;
  currentStageIndex = Math.max(0, Math.min(currentTrace.length - 1, index));
  updateSignalPathActive(currentStageIndex);
  $("stepIndicator").textContent = `Stage ${currentStageIndex + 1} / ${currentTrace.length}`;
  $("prevBtn").disabled = currentStageIndex === 0;
  $("nextBtn").disabled = currentStageIndex === currentTrace.length - 1;
  renderStageDetail(currentTrace[currentStageIndex]);
}

function renderStageDetail(stage) {
  const container = $("stageDetail");
  const num = stage.stage_name.split(".")[0];
  let body = "";

  if (stage.stage_name.includes("Source Encoding")) body = renderHuffmanStage(stage);
  else if (stage.stage_name.includes("Binary Representation")) body = renderBinaryStage(stage);
  else if (stage.stage_name.includes("Compression")) body = renderCompressionStage(stage);
  else if (stage.stage_name.includes("Encryption")) body = renderEncryptionStage(stage);
  else if (stage.stage_name.includes("Packetization")) body = renderPacketStage(stage);
  else if (stage.stage_name.includes("Error Detection")) body = renderCrcStage(stage);
  else if (stage.stage_name.includes("Routing")) body = renderRoutingStage(stage);
  else if (stage.stage_name.includes("Channel")) body = renderChannelStage(stage);
  else if (stage.stage_name.includes("Error Correction")) body = renderEccStage(stage);
  else if (stage.stage_name.includes("Decryption")) body = renderDecryptionStage(stage);
  else if (stage.stage_name.includes("Decoding")) body = renderDecodingStage(stage);
  else body = `<pre>${escapeHtml(JSON.stringify(stage.output_snapshot, null, 2))}</pre>`;

  container.innerHTML = `
    <div class="stage-head">
      <h2>${escapeHtml(stage.stage_name.replace(/^\d+\.\s*/, ""))}</h2>
      <span class="stage-tag">STAGE ${num} / 11</span>
    </div>
    <p class="stage-math">${escapeHtml(stage.math_detail)}</p>
    ${body}
  `;

  if (stage.stage_name.includes("Routing")) {
    const holder = document.getElementById("routingGraphHolder");
    if (holder) drawGraph(holder, stage.input_snapshot.topology, stage.output_snapshot.path);
  }
}

// ---- individual stage renderers -------------------------------------------

function renderHuffmanStage(stage) {
  const o = stage.output_snapshot;
  const rows = kvGrid([
    ["Shannon entropy H(X)", o.entropy_bits_per_symbol.toFixed(4) + " bits/symbol", "accent"],
    ["Achieved avg length L", o.avg_code_length_bits_per_symbol.toFixed(4) + " bits/symbol"],
    ["Distinct symbols", o.num_distinct_symbols],
    ["Fixed-width baseline", o.original_bits_fixed_width + " bits"],
    ["Huffman-encoded size", o.encoded_bits + " bits"],
  ]);
  return `${rows}<div class="hexblock">${escapeHtml(o.bitstring_preview)}</div>`;
}

function renderBinaryStage(stage) {
  const o = stage.output_snapshot;
  const rows = kvGrid([
    ["Pad bits added", o.pad_bits_added],
    ["Byte-aligned length", o.byte_aligned_length_bits + " bits"],
    ["Bytes", o.num_bytes],
  ]);
  const grid = (o.byte_grid_preview || []).map((b) =>
    `<div class="byte-cell"><span class="bits">${b.bits}</span><span class="ascii">${b.hex}${b.ascii ? " '" + escapeHtml(b.ascii) + "'" : ""}</span></div>`
  ).join("");
  return `${rows}<div class="byte-grid">${grid}</div>`;
}

function renderCompressionStage(stage) {
  const o = stage.output_snapshot;
  const inputBytes = stage.input_snapshot.input_size_bytes;
  const outBytes = o.serialized_size_bytes;
  const maxH = 120;
  const scale = Math.max(inputBytes, outBytes, 1);
  const rows = kvGrid([
    ["LZ77 tokens", o.num_tokens],
    ["Compression ratio", o.compression_ratio ? o.compression_ratio + "x" : "—"],
  ]);
  return `${rows}
  <div class="bar-chart">
    <div class="bar-col">
      <div class="bar-value">${inputBytes}B</div>
      <div class="bar dim" style="height:${Math.max(3, (inputBytes / scale) * maxH)}px"></div>
      <div class="bar-label">before</div>
    </div>
    <div class="bar-col">
      <div class="bar-value">${outBytes}B</div>
      <div class="bar" style="height:${Math.max(3, (outBytes / scale) * maxH)}px"></div>
      <div class="bar-label">after (tokens)</div>
    </div>
  </div>`;
}

function renderEncryptionStage(stage) {
  const o = stage.output_snapshot;
  const trunc = (v) => (String(v).length > 24 ? String(v).slice(0, 24) + "…" : v);
  const rows = kvGrid([
    ["AES-128 key", o.aes_key_hex, "violet"],
    ["IV", o.iv_hex, "violet"],
    ["Ciphertext size", o.ciphertext_size_bytes + " bytes"],
    ["RSA modulus n", trunc(o.rsa_n), "amber"],
    ["RSA public e", o.rsa_e, "amber"],
    ["RSA private d", trunc(o.rsa_d), "amber"],
    ["Wrapped session key", trunc(o.wrapped_aes_key_int), "amber"],
  ]);
  return `${rows}<div class="hexblock">ciphertext: ${escapeHtml(o.ciphertext_hex_preview)}</div>`;
}

function renderPacketStage(stage) {
  const o = stage.output_snapshot;
  const rows = kvGrid([["Total packets", o.num_packets], ["Payload size", o.payload_size + " bytes/packet"]]);
  const trs = (o.packet_preview || []).map((p) =>
    `<tr><td>${p.seq_num}</td><td>${p.total_packets}</td><td>${p.payload_size}</td><td>${p.header_checksum}</td></tr>`
  ).join("");
  return `${rows}
  <table class="packet-table">
    <thead><tr><th>seq</th><th>total</th><th>payload bytes</th><th>header checksum</th></tr></thead>
    <tbody>${trs}</tbody>
  </table>
  ${o.num_packets > 5 ? `<p class="explore-hint">…and ${o.num_packets - 5} more packets.</p>` : ""}`;
}

function renderCrcStage(stage) {
  const o = stage.output_snapshot;
  return kvGrid([["CRC-32 checksum", "0x" + o.checksum_hex, "accent"]]);
}

function renderRoutingStage(stage) {
  const i = stage.input_snapshot;
  const o = stage.output_snapshot;
  const rows = kvGrid([
    ["Source \u2192 Target", `${i.source} \u2192 ${i.target}`],
    ["Shortest path", o.path.join(" \u2192 "), "accent"],
    ["Total cost", o.total_cost],
    ["Relaxation steps logged", o.num_relaxation_steps],
  ]);
  return `${rows}<div class="graph-output" id="routingGraphHolder"></div>`;
}

function renderChannelStage(stage) {
  const i = stage.input_snapshot, o = stage.output_snapshot;
  const rows = kvGrid([
    ["Bits transmitted", i.transmitted_bits_length],
    ["Bits flipped", o.num_flips],
    ["Observed BER", o.ber_observed],
    ["Theoretical BER", o.ber_theoretical],
  ]);
  const flipped = new Set(o.positions_flipped_preview || []);
  const before = i.transmitted_bits_preview || "";
  const after = o.received_bits_preview || "";
  const afterHtml = after.split("").map((bit, idx) =>
    flipped.has(idx) ? `<span class="flip">${bit}</span>` : bit
  ).join("");
  return `${rows}
    <p class="explore-hint">transmitted (first ${before.length} bits)</p>
    <div class="bit-strip">${before}</div>
    <p class="explore-hint">received — flipped bits highlighted</p>
    <div class="bit-strip">${afterHtml}</div>`;
}

function renderEccStage(stage) {
  const o = stage.output_snapshot;
  if (o.scheme === "hamming") {
    return kvGrid([
      ["Scheme", "Hamming(7,4)"],
      ["7-bit blocks", o.num_blocks],
      ["Blocks with a corrected bit", o.blocks_with_corrected_errors],
    ]);
  }
  return kvGrid([
    ["Scheme", "Reed\u2013Solomon over GF(2^8)"],
    ["Symbol blocks", o.num_blocks],
    ["Blocks corrected successfully", o.blocks_corrected_successfully],
    ["Max correctable errors / block", o.max_correctable_symbol_errors_per_block],
  ]);
}

function renderDecryptionStage(stage) {
  const o = stage.output_snapshot;
  return kvGrid([
    ["Recovered plaintext size", o.recovered_plaintext_size_bytes + " bytes"],
    ["AES session key recovered correctly", o.session_key_recovered_correctly ? "yes" : "no",
      o.session_key_recovered_correctly ? "accent" : "amber"],
  ]);
}

function renderDecodingStage(stage) {
  const o = stage.output_snapshot;
  return `${kvGrid([["Matches original, byte for byte", o.matches_original_byte_for_byte ? "yes" : "no",
      o.matches_original_byte_for_byte ? "accent" : "amber"]])}
    <div class="hexblock">${escapeHtml(o.recovered_text_preview)}</div>`;
}

// ============================================================================
// D3 visualizations
// ============================================================================
function drawGraph(container, topology, highlightPath) {
  container.innerHTML = "";
  const width = Math.max(container.clientWidth || 420, 420);
  const height = 240;
  const svg = d3.select(container).append("svg").attr("width", "100%").attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`).attr("class", "graph-svg");

  const nodes = topology.nodes.map((n) => ({ id: n }));
  const edgeSet = new Set();
  for (let i = 0; i < (highlightPath || []).length - 1; i++) {
    edgeSet.add(highlightPath[i] + "|" + highlightPath[i + 1]);
    edgeSet.add(highlightPath[i + 1] + "|" + highlightPath[i]);
  }
  const links = topology.edges.map((e) => ({
    source: e.source, target: e.target, weight: e.weight,
    onPath: edgeSet.has(e.source + "|" + e.target),
  }));

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id((d) => d.id).distance(90))
    .force("charge", d3.forceManyBody().strength(-260))
    .force("center", d3.forceCenter(width / 2, height / 2))
    .stop();
  for (let i = 0; i < 220; i++) sim.tick();

  const nodeById = new Map(nodes.map((n) => [n.id, n]));

  svg.selectAll(".edge-line").data(links).enter().append("line")
    .attr("class", (d) => "edge-line" + (d.onPath ? " on-path" : ""))
    .attr("x1", (d) => nodeById.get(d.source.id || d.source).x)
    .attr("y1", (d) => nodeById.get(d.source.id || d.source).y)
    .attr("x2", (d) => nodeById.get(d.target.id || d.target).x)
    .attr("y2", (d) => nodeById.get(d.target.id || d.target).y);

  svg.selectAll(".edge-weight").data(links).enter().append("text")
    .attr("class", "edge-weight")
    .attr("x", (d) => (nodeById.get(d.source.id || d.source).x + nodeById.get(d.target.id || d.target).x) / 2)
    .attr("y", (d) => (nodeById.get(d.source.id || d.source).y + nodeById.get(d.target.id || d.target).y) / 2)
    .text((d) => d.weight);

  const nodeG = svg.selectAll(".node-g").data(nodes).enter().append("g").attr("transform", (d) => `translate(${d.x},${d.y})`);
  nodeG.append("circle")
    .attr("class", (d) => "node-circle" + ((highlightPath || []).includes(d.id) ? " on-path" : ""))
    .attr("r", 15);
  nodeG.append("text").attr("text-anchor", "middle").attr("dy", 4).text((d) => d.id);
}

function buildHuffmanHierarchy(codeTable) {
  const root = { name: "", children: [] };
  const childrenByPath = { "": root };
  Object.entries(codeTable).forEach(([symbol, code]) => {
    let path = "";
    let node = root;
    for (const bit of code) {
      const nextPath = path + bit;
      if (!childrenByPath[nextPath]) {
        const child = { name: "", children: [], bit };
        node.children = node.children || [];
        node.children.push(child);
        childrenByPath[nextPath] = child;
      }
      node = childrenByPath[nextPath];
      path = nextPath;
    }
    node.name = symbol === " " ? "\u2423" : symbol;
    node.isLeaf = true;
  });
  return root;
}

function drawHuffmanTree(container, codeTable) {
  container.innerHTML = "";
  const data = buildHuffmanHierarchy(codeTable);
  const root = d3.hierarchy(data);
  const width = Math.max(container.clientWidth || 380, 380);
  const dx = 26;
  const dy = width / (root.height + 1.4);
  const treeLayout = d3.tree().nodeSize([dx, dy]);
  treeLayout(root);

  let x0 = Infinity, x1 = -Infinity;
  root.each((d) => { if (d.x > x1) x1 = d.x; if (d.x < x0) x0 = d.x; });
  const height = x1 - x0 + dx * 2;

  const svg = d3.select(container).append("svg")
    .attr("width", "100%").attr("height", height)
    .attr("viewBox", `${-dy * 0.15} ${x0 - dx} ${dy * (root.height + 1.3)} ${height}`);

  svg.selectAll(".link").data(root.links()).enter().append("path")
    .attr("class", "link")
    .attr("d", d3.linkHorizontal().x((d) => d.y).y((d) => d.x));

  const node = svg.selectAll(".node").data(root.descendants()).enter().append("g")
    .attr("class", (d) => "node" + (d.data.isLeaf ? " leaf" : ""))
    .attr("transform", (d) => `translate(${d.y},${d.x})`);

  node.append("circle").attr("r", (d) => (d.data.isLeaf ? 11 : 5));
  node.filter((d) => d.data.isLeaf).append("text").attr("text-anchor", "middle").attr("dy", 4).text((d) => d.data.name);
}

// ============================================================================
// Explore playgrounds
// ============================================================================
function initExplore() {
  $("huffmanBtn").addEventListener("click", async () => {
    const text = $("huffmanText").value || "example";
    try {
      const r = await postJSON("/api/encode/huffman", { text });
      drawHuffmanTree($("huffmanTreeSvg"), r.code_table);
      $("huffmanStats").innerHTML =
        `entropy: ${r.entropy.toFixed(3)} bits/symbol<br>avg code length: ${r.avg_code_length.toFixed(3)} bits/symbol<br>` +
        `${r.original_bits} \u2192 ${r.encoded_bits} bits`;
    } catch (e) { $("huffmanStats").textContent = "Error: " + e.message; }
  });

  let customEdges = null;
  async function refreshExploreGraph() {
    const holder = $("graphSvg");
    try {
      const topo = customEdges
        ? { nodes: [...new Set(customEdges.flatMap((e) => [e.source, e.target]))], edges: customEdges }
        : await getJSON("/api/routing/topology");
      if (!topo.nodes.length) { holder.innerHTML = ""; return; }
      const src = topo.nodes[0], tgt = topo.nodes[topo.nodes.length - 1];
      const pathReq = { source: src, target: tgt, edges: customEdges };
      const pathResult = await postJSON("/api/routing/shortest-path", pathReq);
      drawGraph(holder, topo, pathResult.path);
      const caption = document.createElement("p");
      caption.className = "explore-hint";
      caption.textContent = pathResult.reachable
        ? `Shortest path ${src} \u2192 ${tgt}: ${pathResult.path.join(" \u2192 ")} (cost ${pathResult.total_cost})`
        : `${tgt} is unreachable from ${src}.`;
      holder.appendChild(caption);
    } catch (e) {
      holder.innerHTML = `<p class="explore-hint">Error: ${escapeHtml(e.message)}</p>`;
    }
  }

  $("addEdgeBtn").addEventListener("click", async () => {
    const from = $("edgeFrom").value.trim();
    const to = $("edgeTo").value.trim();
    const weight = Number($("edgeWeight").value);
    if (!from || !to || !isFinite(weight) || weight < 0) return;
    if (!customEdges) {
      const topo = await getJSON("/api/routing/topology");
      customEdges = topo.edges.map((e) => ({ source: e.source, target: e.target, weight: e.weight }));
    }
    customEdges.push({ source: from, target: to, weight });
    refreshExploreGraph();
  });

  $("resetTopologyBtn").addEventListener("click", () => { customEdges = null; refreshExploreGraph(); });

  refreshExploreGraph();

  const flipSlider = $("explFlipSlider"), flipValue = $("explFlipValue");
  flipSlider.addEventListener("input", () => { flipValue.textContent = Number(flipSlider.value).toFixed(2); });

  $("channelBtn").addEventListener("click", async () => {
    const bits = $("channelBits").value.trim();
    if (!/^[01]+$/.test(bits)) { $("channelOutput").textContent = "bits must be only 0s and 1s"; return; }
    try {
      const r = await postJSON("/api/channel/transmit", { bits, flip_probability: Number(flipSlider.value) });
      const flipped = new Set(r.positions_flipped);
      const afterHtml = r.received_bits.split("").map((b, idx) => flipped.has(idx) ? `<span class="flip">${b}</span>` : b).join("");
      $("channelOutput").innerHTML =
        `<div class="bit-strip">${bits}</div><div class="bit-strip">${afterHtml}</div>` +
        `${r.num_flips} flips · observed BER ${r.ber_observed} · theoretical BER ${r.ber_theoretical}`;
    } catch (e) { $("channelOutput").textContent = "Error: " + e.message; }
  });

  $("aesBtn").addEventListener("click", async () => {
    const hex = $("aesPlaintext").value.trim();
    try {
      const r = await postJSON("/api/encrypt/aes", { plaintext_hex: hex });
      $("aesOutput").innerHTML = kvGrid([
        ["Key", r.key_hex, "violet"],
        ["IV", r.iv_hex, "violet"],
        ["Ciphertext", r.ciphertext_hex, "accent"],
        ["Round keys generated", r.round_keys.length],
      ]);
    } catch (e) { $("aesOutput").textContent = "Error: " + e.message; }
  });
}

// ============================================================================
// Init
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initControls();
  initExplore();
});
