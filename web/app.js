// 지하철 실내 길찾기: 사진 → 현재 위치 → 목적지 → 경로 안내
// 서버/외부 API 없이 휴대폰 브라우저 안에서 AI 모델(web/model/embed.onnx)이 돌아간다.

const I18N = {
  en: {
    loading: "Loading map…", loadingModel: "Loading AI model…", noData: "No map data yet for this station. Build it with mapper/build_map.py.",
    step1: "1. Where are you?", hint: "Take a photo of something distinctive nearby: a sign, exit number, gate or shop.",
    takePhoto: "📷 Take a photo", yourPhoto: "Your photo", mapView: "Map view",
    analyzing: "Finding your location…", matchOk: "Is this the same place?", matchLow: "Not sure. Is this the same place? If not, try a photo of a sign.",
    yes: "Yes, that's it", no: "No, show another", noMore: "No more candidates. Please take another photo.",
    step2: "2. Where do you want to go?", prev: "◀ Back", next: "Next ▶", relocate: "📷 Check my location", restart: "Start over",
    goTo: "To: {p}", eta: "About {m} min · {n} steps", walk: "Walk toward this view", arrived: "You have arrived: {p}", here: "You are already here.",
    unreachable: "No route found from here. Try another photo.", stepOf: "Step {i} of {n}",
  },
  ko: {
    loading: "지도를 불러오는 중…", loadingModel: "AI 모델을 불러오는 중…", noData: "이 역의 지도 데이터가 아직 없습니다. mapper/build_map.py 로 만들어 주세요.",
    step1: "1. 지금 어디에 있나요?", hint: "주변의 눈에 띄는 것을 찍어 주세요: 표지판, 출구 번호, 개찰구, 상점 등.",
    takePhoto: "📷 사진 찍기", yourPhoto: "내 사진", mapView: "지도 속 장면",
    analyzing: "위치를 찾는 중…", matchOk: "같은 장소인가요?", matchLow: "확실하지 않습니다. 같은 장소인가요? 아니면 표지판을 찍어 보세요.",
    yes: "네, 여기예요", no: "아니요, 다른 후보", noMore: "후보가 더 없습니다. 다시 찍어 주세요.",
    step2: "2. 어디로 가시나요?", prev: "◀ 이전", next: "다음 ▶", relocate: "📷 위치 다시 확인", restart: "처음부터",
    goTo: "목적지: {p}", eta: "약 {m}분 · {n}단계", walk: "이 장면 쪽으로 걸어가세요", arrived: "도착했습니다: {p}", here: "이미 목적지에 있습니다.",
    unreachable: "여기서 가는 길을 찾지 못했습니다. 다른 사진을 찍어 보세요.", stepOf: "{n}단계 중 {i}",
  },
  zh: {
    loading: "正在加载地图…", loadingModel: "正在加载AI模型…", noData: "该站暂无地图数据。",
    step1: "1. 您在哪里？", hint: "请拍摄附近明显的东西：指示牌、出口编号、检票口或商店。",
    takePhoto: "📷 拍照", yourPhoto: "您的照片", mapView: "地图画面",
    analyzing: "正在定位…", matchOk: "是同一个地方吗？", matchLow: "不确定。是同一个地方吗？如果不是，请拍指示牌。",
    yes: "是的", no: "不是，换一个", noMore: "没有更多候选，请重新拍照。",
    step2: "2. 您要去哪里？", prev: "◀ 上一步", next: "下一步 ▶", relocate: "📷 重新定位", restart: "重新开始",
    goTo: "目的地：{p}", eta: "约 {m} 分钟 · {n} 步", walk: "朝这个画面走", arrived: "已到达：{p}", here: "您已在目的地。",
    unreachable: "找不到路线，请重新拍照。", stepOf: "第 {i} 步，共 {n} 步",
  },
  ja: {
    loading: "地図を読み込み中…", loadingModel: "AIモデルを読み込み中…", noData: "この駅の地図データはまだありません。",
    step1: "1. 今どこにいますか？", hint: "近くの目立つものを撮ってください：案内板、出口番号、改札、お店など。",
    takePhoto: "📷 写真を撮る", yourPhoto: "あなたの写真", mapView: "地図の風景",
    analyzing: "位置を探しています…", matchOk: "同じ場所ですか？", matchLow: "自信がありません。同じ場所ですか？違う場合は案内板を撮ってください。",
    yes: "はい、ここです", no: "いいえ、次の候補", noMore: "候補がありません。もう一度撮ってください。",
    step2: "2. どこへ行きますか？", prev: "◀ 戻る", next: "次へ ▶", relocate: "📷 位置を再確認", restart: "最初から",
    goTo: "目的地：{p}", eta: "約{m}分 · {n}ステップ", walk: "この風景の方へ歩いてください", arrived: "到着しました：{p}", here: "すでに目的地にいます。",
    unreachable: "ここからの経路が見つかりません。別の写真を撮ってください。", stepOf: "{n}ステップ中 {i}",
  },
};

const MIN_CONFIDENT = 0.6; // 이보다 낮으면 "확실하지 않음" 표시
const STEP_EVERY_SEC = 8;  // 안내 사진 간격(초)
const $ = (id) => document.getElementById(id);

const params = new URLSearchParams(location.search);
const station = params.get("station") || "seoul";
const base = `data/${station}/`;

let lang = (navigator.language || "en").slice(0, 2);
try { lang = localStorage.getItem("lang") || lang; } catch {}
if (!I18N[lang]) lang = "en";
let graph, emb, adj, session;
let candidates = [], candIdx = 0, currentNode = null, destPlace = null, steps = [], stepIdx = 0;

const t = (key, vars = {}) => (I18N[lang][key] || I18N.en[key]).replace(/\{(\w+)\}/g, (_, k) => vars[k]);
const placeName = (p) => p.name[lang] || p.name.en || p.name.ko;

function applyLang() {
  document.documentElement.lang = lang;
  $("lang").value = lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
  if (graph) {
    $("title").textContent = graph.name[lang] || graph.name.en;
    renderPlaces();
    if (!$("route").hidden) renderStep();
  }
}

function show(...ids) {
  for (const id of ["locate", "dest", "route"]) $(id).hidden = !ids.includes(id);
}

async function init() {
  applyLang();
  $("status").textContent = t("loading");
  try {
    const res = await fetch(base + "graph.json");
    if (!res.ok) throw new Error(res.status);
    graph = await res.json();
  } catch {
    $("status").textContent = t("noData");
    return;
  }
  const bytes = Uint8Array.from(atob(graph.emb), (c) => c.charCodeAt(0));
  emb = new Int8Array(bytes.buffer);
  adj = graph.nodes.map(() => []);
  for (const [a, b, w] of graph.edges) {
    adj[a].push([b, w]);
    adj[b].push([a, w]);
  }

  $("status").textContent = t("loadingModel");
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/";
  session = await ort.InferenceSession.create("model/embed.onnx");
  $("status").textContent = "";
  applyLang();
  show("locate");
}

// common.py preprocess() 와 같은 처리: 가운데 정사각형 → 224x224 → RGB 0~255
async function preprocess(file) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(bmp.width, bmp.height);
  const sx = (bmp.width - s) / 2, sy = (bmp.height - s) / 2;
  // 큰 사진은 단계적으로 줄여야 계단 현상이 적다 (OpenCV INTER_AREA 와 비슷하게)
  let src = bmp, srcX = sx, srcY = sy, srcS = s;
  while (srcS > 448) {
    const half = Math.round(srcS / 2);
    const c = new OffscreenCanvas(half, half);
    const cx = c.getContext("2d");
    cx.imageSmoothingQuality = "high";
    cx.drawImage(src, srcX, srcY, srcS, srcS, 0, 0, half, half);
    src = c; srcX = 0; srcY = 0; srcS = half;
  }
  const ctx = $("canvas").getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(src, srcX, srcY, srcS, srcS, 0, 0, 224, 224);
  const px = ctx.getImageData(0, 0, 224, 224).data;
  const n = 224 * 224, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    x[i] = px[i * 4];
    x[n + i] = px[i * 4 + 1];
    x[2 * n + i] = px[i * 4 + 2];
  }
  return new ort.Tensor("float32", x, [1, 3, 224, 224]);
}

async function locate(file) {
  $("match").hidden = true;
  $("status").textContent = t("analyzing");
  $("myPhoto").src = URL.createObjectURL(file);
  const out = await session.run({ [session.inputNames[0]]: await preprocess(file) });
  const q = out[session.outputNames[0]].data;
  const d = graph.dim;
  const scores = graph.nodes.map((_, i) => {
    let s = 0;
    for (let k = 0; k < d; k++) s += q[k] * emb[i * d + k];
    return s / 127;
  });
  // 서로 떨어진 후보만 남긴다 (바로 옆 지점은 같은 장소이므로)
  const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  candidates = [];
  for (const [s, i] of order) {
    if (candidates.length >= 5) break;
    if (candidates.some((c) => adj[c.node].some(([j]) => j === i) || c.node === i)) continue;
    candidates.push({ node: i, score: s });
  }
  candIdx = 0;
  $("status").textContent = "";
  showCandidate();
}

function showCandidate() {
  if (candIdx >= candidates.length) {
    $("match").hidden = true;
    $("status").textContent = t("noMore");
    return;
  }
  const c = candidates[candIdx];
  $("matchImg").src = base + graph.nodes[c.node].img;
  $("matchMsg").textContent = t(c.score >= MIN_CONFIDENT ? "matchOk" : "matchLow");
  $("match").hidden = false;
}

function renderPlaces() {
  const box = $("places");
  box.innerHTML = "";
  for (const p of graph.places) {
    const b = document.createElement("button");
    b.className = "btn";
    b.textContent = placeName(p);
    b.onclick = () => startRoute(p);
    box.appendChild(b);
  }
}

function dijkstra(from, to) {
  const dist = new Float64Array(adj.length).fill(Infinity);
  const prev = new Int32Array(adj.length).fill(-1);
  const heap = [[0, from]];
  dist[from] = 0;
  while (heap.length) {
    let mi = 0;
    for (let i = 1; i < heap.length; i++) if (heap[i][0] < heap[mi][0]) mi = i;
    const [du, u] = heap.splice(mi, 1)[0];
    if (u === to) break;
    if (du > dist[u]) continue;
    for (const [v, w] of adj[u]) {
      if (du + w < dist[v]) {
        dist[v] = du + w;
        prev[v] = u;
        heap.push([dist[v], v]);
      }
    }
  }
  if (dist[to] === Infinity) return null;
  const path = [];
  for (let v = to; v !== -1; v = prev[v]) path.unshift(v);
  return { path, seconds: dist[to] };
}

function startRoute(place) {
  destPlace = place;
  const r = dijkstra(currentNode, place.node);
  if (!r) {
    $("status").textContent = t("unreachable");
    show("locate");
    return;
  }
  // 경로를 일정 시간 간격의 안내 사진으로 나눈다
  steps = [];
  let acc = 0;
  for (let k = 1; k < r.path.length; k++) {
    const w = adj[r.path[k - 1]].find(([v]) => v === r.path[k])[1];
    acc += w;
    if (acc >= STEP_EVERY_SEC || k === r.path.length - 1) {
      steps.push(r.path[k]);
      acc = 0;
    }
  }
  stepIdx = 0;
  $("routeTitle").textContent = t("goTo", { p: placeName(place) });
  $("routeInfo").textContent = steps.length
    ? t("eta", { m: Math.max(1, Math.round(r.seconds / 60)), n: steps.length })
    : t("here");
  $("status").textContent = "";
  show("route");
  renderStep();
}

function renderStep() {
  const last = stepIdx === steps.length - 1 || !steps.length;
  const node = steps.length ? steps[stepIdx] : currentNode;
  $("stepImg").src = base + graph.nodes[node].img;
  $("stepMsg").textContent = last
    ? t("arrived", { p: placeName(destPlace) })
    : `${t("stepOf", { i: stepIdx + 1, n: steps.length })} · ${t("walk")}`;
  $("prev").disabled = stepIdx === 0;
  $("next").disabled = last;
}

$("lang").onchange = (e) => {
  lang = e.target.value;
  try { localStorage.setItem("lang", lang); } catch {}
  applyLang();
};
$("photo").onchange = (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) locate(f).catch((err) => ($("status").textContent = String(err)));
};
$("matchNo").onclick = () => { candIdx++; showCandidate(); };
$("matchYes").onclick = () => {
  currentNode = candidates[candIdx].node;
  $("match").hidden = true;
  if (destPlace) return startRoute(destPlace); // 길 안내 중 위치 재확인 → 경로 다시 계산
  show("locate", "dest");
  $("dest").scrollIntoView({ behavior: "smooth" });
};
$("prev").onclick = () => { stepIdx--; renderStep(); };
$("next").onclick = () => { stepIdx++; renderStep(); };
$("relocate").onclick = () => { show("locate"); $("match").hidden = true; };
$("restart").onclick = () => { destPlace = null; currentNode = null; $("match").hidden = true; show("locate"); };

init().catch((err) => ($("status").textContent = String(err)));
