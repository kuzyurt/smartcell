const I18N = {
  en: {},
  de: {
    "nav.how":"So funktioniert's","nav.smooth":"Ruhige Bewegung","nav.play":"Spielen","nav.results":"Ergebnisse",
    "hero.eyebrow":"SO-101 · MuJoCo · eine Handgelenk-Kamera",
    "hero.title":"Ein Arm, der Teile von einem nie stehenden Band fängt.",
    "hero.lead":"Eine eingefrorene gelernte Policy, ein klassischer visueller Servo und ein Release-Primitiv greifen ein 32-mm-Teil vom laufenden Förderband und legen es in eine Kiste. Jetzt ohne Zittern.",
    "hero.cta":"▶ Zelle spielen","hero.cta2":"Glättung ansehen",
    "hero.s1":"platziert auf neuen Seeds (200 Versuche)","hero.s2t":"Eine Kamera","hero.s2":"kein Simulator-Objektzustand zur Laufzeit","hero.s3":"weniger Gelenk-Ruck mit ruhiger Bewegung",
    "how.k":"So funktioniert's","how.h":"Drei Ebenen, eine Kamera.","how.sub":"Eine gelernte Policy macht die grobe Bewegung, ein klassischer Servo die letzten Zentimeter, und ein propriozeptives Primitiv entscheidet, wann losgelassen wird.",
    "how.c1h":"Eingefrorene Policy","how.c1":"Liest 8 Graustufen-Bilder (280 ms) plus 12 Gelenkwerte und sagt einen 8×6-Aktions-Chunk bei 25 Hz voraus.",
    "how.c2h":"Visueller Servo","how.c2":"Projiziert Teil und Kiste auf die bekannte Arbeitsebene, führt dem bewegten Teil voraus und schließt am vorhergesagten Kontaktpunkt.",
    "how.c3h":"Release-Primitiv","how.c3":"Bestätigt festgehaltenes Teil, Anheben und tiefes Absenken, bevor die Backen öffnen. „Kiste berührt“ zählt nie als Ablage.",
    "sm.k":"Das Wackeln","sm.h":"Es funktionierte, aber es zitterte. Darum.","sm.sub":"Die Policy gibt alle 40 ms ein Arm-Ziel aus. Der Simulator hielt jedes Ziel konstant und sprang dann zum nächsten, der Positionsregler folgte also einer Treppe mit 25 Hz. Interpolation innerhalb jedes Schritts entfernt die Treppe, ohne die Policy anzufassen.",
    "sm.before":"VORHER","sm.after":"NACHHER","sm.old":"Original-Befehl","sm.new":"interpolierter Befehl","sm.mh":"Gemessen auf identischen Episoden",
    "pl.k":"Spielen","pl.h":"Starte die Zelle selbst.","pl.sub":"Die Simulation läuft auf dem Rechner, der diese Seite gestartet hat. Ändere Szene, Licht, Kistenposition und Bewegungsmodus, schalte den visuellen Servo aus und fliege mit der Kamera durch die Blender-Halle.",
    "pl.ready":"Bereit","pl.start":"▶ Start","pl.pause":"Pause","pl.next":"Nächstes Teil","pl.ep":"Episode","pl.picks":"Greifen","pl.places":"Ablegen","pl.jit":"Befehls-Jitter",
    "pl.hint":"Orbit-Ansicht: ziehen zum Drehen, scrollen zum Zoomen. Ohne GPU läuft die Simulation langsamer als Echtzeit.",
    "pl.scene":"Szene","pl.blender":"Blender-Halle","pl.stock":"Standardzelle","pl.cond":"Bedingung","pl.tierB":"Licht","pl.boxX":"Kiste","pl.motion":"Bewegung","pl.smooth":"ruhig","pl.orig":"original","pl.servo":"Visueller Servo","pl.cam":"Kamera","pl.speed":"Tempo",
    "rs.k":"Ergebnisse","rs.h":"Neue Seeds, eingefrorener Checkpoint.","rs.sub":"200 Versuche pro Bedingung. Ablage heißt: Teil losgelassen, bleibt in der Kiste und bewegt sich 0,4 s lang langsamer als 0,05 m/s. Die Testobjekte überschneiden sich nicht mit den 2.302 Trainings-Seeds.",
    "rs.c":"Bedingung","rs.tb":"tierB — Licht, Tönung, Montage-Jitter","rs.bx":"boxX — Kistenecken außerhalb des Trainings","rs.bl":"Blender-Halle, Zero-Shot (200 Versuche)","rs.no":"Blender-Halle, visueller Servo aus (Kontrolle)",
    "rs.note":"Nur Simulation, ein 32-mm-Teil, ein SO-101, ein Förderband. Das Glättungsprofil ist neu und separat ausgewertet; die obigen Zahlen gelten für das eingefrorene Originalprotokoll.",
    "run.k":"Starten","run.h":"Ein Befehl.","run.sub":"Einmal die Requirements installieren, dann diese Seite starten. Sie öffnet sich im Browser und simuliert lokal.",
    "of.h":"Auf deinem Rechner starten","of.p":"Diese Seite ist eine statische Kopie. Die Live-Simulation mit allen Einstellungen aus dem Video läuft lokal mit einem Befehl.",
    "ft.t":"Ein SO-101-Förderband-Pick-and-Place-Forschungsartefakt."
  }
};
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const store = { get: k => { try { return localStorage.getItem(k) } catch (e) { return null } }, set: (k, v) => { try { localStorage.setItem(k, v) } catch (e) {} } };

/* ---- i18n ---- */
const EN = {};
$$("[data-i18n]").forEach(el => EN[el.dataset.i18n] = el.innerHTML);
function setLang(l) {
  document.documentElement.lang = l; store.set("lang", l);
  $$("[data-i18n]").forEach(el => { const k = el.dataset.i18n; el.innerHTML = (l === "de" && I18N.de[k]) || EN[k]; });
  $$(".lang button").forEach(b => b.classList.toggle("on", b.dataset.lang === l));
  renderBars();
}
$$(".lang button").forEach(b => b.onclick = () => setLang(b.dataset.lang));

/* ---- nav ---- */
const nav = $("#nav");
addEventListener("scroll", () => nav.classList.toggle("solid", scrollY > innerHeight * .6), { passive: true });

/* ---- smoothing figures (data/smoothing.json written by tools/measure_smoothing.py) ---- */
let SM = null;
function renderBars() {
  const box = $("#bars"); if (!SM) return; const de = document.documentElement.lang === "de";
  const rows = [
    [de ? "Ruck (Jerk)" : "Jerk", SM.original.rms_jerk, SM.smooth.rms_jerk],
    [de ? "Beschleunigung" : "Acceleration", SM.original.rms_acc, SM.smooth.rms_acc],
    [de ? "Hochfrequenz" : "High-freq.", SM.original.hf_rms, SM.smooth.hf_rms],
  ];
  box.innerHTML = rows.map(([n, a, b]) => {
    const m = Math.max(a, b);
    return `<div><div class="bar-row"><span>${n}</span><div class="bar"><i style="width:${a / m * 100}%"></i></div><span class="val">100 %</span></div>
    <div class="bar-row"><span></span><div class="bar"><i class="good" style="width:${b / m * 100}%"></i></div><span class="val">${Math.round(b / a * 100)} %</span></div></div>`;
  }).join("");
  $("#smNote").textContent = (de ? "Rot: original · Grün: geglättet. " : "Red: original · green: smoothed. ") + SM.note;
  $("#heroCut").textContent = Math.round((1 - SM.smooth.rms_jerk / SM.original.rms_jerk) * 100) + " %";
}
fetch("media/smoothing.json").then(r => r.ok ? r.json() : null).then(j => { SM = j; renderBars(); }).catch(() => {});
/* play the two comparison videos in sync once visible */
const vOld = $("#vOld"), vNew = $("#vNew");
new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { vOld.currentTime = vNew.currentTime = 0; vOld.play().catch(() => {}); vNew.play().catch(() => {}); } else { vOld.pause(); vNew.pause(); } }), { threshold: .4 }).observe($(".vs"));

/* ---- play ---- */
const api = (p, b) => fetch(p, b ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(b) } : undefined).then(r => r.json());
let S = null, live = false;
const cfgPost = o => api("/api/config", o).then(render).catch(() => {});
$$(".seg").forEach(seg => seg.onclick = e => {
  const b = e.target.closest("button"); if (!b) return;
  const k = seg.dataset.key, v = k === "speed" ? +b.dataset.v : b.dataset.v;
  cfgPost({ [k]: v });
});
$("#servo").onchange = e => cfgPost({ servo: e.target.checked });
$("#seed").onchange = e => cfgPost({ seed: +e.target.value });
$("#dice").onclick = () => { const s = Math.floor(Math.random() * 1e6); $("#seed").value = s; cfgPost({ seed: s }); };
const ctl = a => api("/api/control", { action: a }).then(render);
const start = () => { if (!live) connect(); ctl(S && S.run.running ? "restart" : "start"); };
$("#btnStart").onclick = start; $("#bigStart").onclick = start;
$("#btnPause").onclick = () => ctl(S && S.run.paused ? "resume" : "pause");
$("#btnNext").onclick = () => ctl("restart");

function connect() {
  live = true;
  $("#mainImg").src = "/stream/main.mjpg"; $("#wristImg").src = "/stream/wrist.mjpg";
}
function render(st) {
  if (!st) return; S = st;
  $$(".seg").forEach(seg => { const k = seg.dataset.key, cur = String(st.cfg[k]); $$("button", seg).forEach(b => b.classList.toggle("on", b.dataset.v === cur)); });
  $("#servo").checked = st.cfg.servo; if (document.activeElement !== $("#seed")) $("#seed").value = st.cfg.seed;
  const s = st.stats, run = st.run;
  $("#roEp").textContent = s.episode; $("#roPick").textContent = `${s.picks}/${s.episode}`; $("#roPlace").textContent = `${s.places}/${s.episode}`;
  $("#roJit").textContent = run.running && s.phase === "policy" ? (s.jitter * 100).toFixed(1) : "–";
  $("#chipPhase").textContent = run.paused ? "paused" : s.phase; $("#chipFps").textContent = s.fps ? s.fps + " fps · " + st.info.device : st.info.device;
  $("#overlay").hidden = run.running;
  $("#btnStart").textContent = run.running ? (document.documentElement.lang === "de" ? "↻ Neustart" : "↻ Restart") : (document.documentElement.lang === "de" ? "▶ Start" : "▶ Start");
  $("#btnPause").textContent = run.paused ? "Resume" : (document.documentElement.lang === "de" ? "Pause" : "Pause");
  $("#screen").classList.toggle("orbit", st.cfg.camera === "orbit");
  $("#log").innerHTML = s.log.map(l => `<span class="dot ${l.place ? "ok" : "bad"}" title="episode ${l.ep}: pick ${l.pick}, place ${l.place}">${l.place ? "✓" : "✗"}</span>`).join("");
}
async function poll() { if (!polling) return; try { render(await api("/api/state")); } catch (e) {} setTimeout(poll, 400); }

/* orbit camera: drag + wheel */
(() => {
  const el = $("#screen"); let drag = null, t = 0;
  const send = o => { if (Date.now() - t > 60) { t = Date.now(); cfgPost({ orbit: o }); } };
  el.addEventListener("pointerdown", e => { if (S && S.cfg.camera === "orbit") { drag = { x: e.clientX, y: e.clientY, o: { ...S.cfg.orbit } }; el.setPointerCapture(e.pointerId); } });
  el.addEventListener("pointermove", e => { if (!drag) return; send({ az: drag.o.az - (e.clientX - drag.x) * .4, el: drag.o.el - (e.clientY - drag.y) * .3, dist: S.cfg.orbit.dist }); });
  el.addEventListener("pointerup", () => drag = null);
  el.addEventListener("wheel", e => { if (S && S.cfg.camera === "orbit") { e.preventDefault(); send({ dist: S.cfg.orbit.dist * (1 + Math.sign(e.deltaY) * .08) }); } }, { passive: false });
})();

setLang(store.get("lang") === "de" ? "de" : "en");
/* static hosting (e.g. Cloudflare Pages): no simulation server, show the recording and the commands */
let polling = true;
function offlineMode() {
  polling = false;
  const sub = $("#play .sub");
  sub.dataset.i18n = "of.sub";
  EN["of.sub"] = "A recorded run in the Blender-dressed hall. Scene, lighting, tote position, motion mode, visual servo and camera are all adjustable in the live version, which you start locally.";
  I18N.de["of.sub"] = "Ein aufgezeichneter Lauf in der Blender-Halle. Szene, Licht, Kistenposition, Bewegungsmodus, visueller Servo und Kamera sind in der Live-Version einstellbar, die du lokal startest.";
  const play = $(".play");
  play.replaceChildren($("#offline").content.cloneNode(true));
  $$("[data-i18n]", play).forEach(el => { EN[el.dataset.i18n] = EN[el.dataset.i18n] || el.innerHTML; });
  setLang(document.documentElement.lang);
}
api("/api/state").then(st => { render(st); connect(); }).catch(offlineMode);
poll();
