const I18N = {
  en: {},
  de: {
    "nav.how":"Funktionsweise","nav.smooth":"Ruhige Bewegung","nav.play":"Simulator","nav.results":"Ergebnisse",
    "hero.cta":"Simulator","hero.cta2":"Glättung ansehen",
    "hero.s1":"der Teile in der Kiste abgelegt, 200 Testläufe in der Standardszene","hero.s2t":"1 Kamera","hero.s2":"keine Objektpositionen aus dem Simulator","hero.s3":"weniger Ruck in den Gelenken nach dem Glätten der Armbefehle",
    "how.h":"Funktionsweise",
    "how.p1":"Ein kleines neuronales Netz, trainiert mit Vorführungen, steuert den Arm. Alle 80 ms sieht es sich die letzten acht Bilder der Handgelenkkamera und die Gelenkwinkel an und plant die nächsten Ziele für Arm und Greifer.",
    "how.p2":"Auf den letzten Zentimetern übernimmt ein von Hand geschriebener Bildregler. Er findet Teil und Kiste im Kamerabild und korrigiert den Arm, damit die Backen am fahrenden Teil schließen. Der Greifer öffnet erst, wenn das Teil gehalten, angehoben und in die Kiste abgesenkt wurde.",
    "sm.h":"Ruhigere Armbewegung","sm.sub":"Das Netz schickt alle 40 ms ein neues Armziel. Ursprünglich blieb jedes Ziel bis zum nächsten fest stehen, die Motoren folgten also einer Treppe und der Arm zitterte. Jetzt läuft das Ziel innerhalb jedes Schritts gleichmäßig zum nächsten Wert. Das Netz selbst ist unverändert.",
    "sm.before":"VORHER","sm.after":"NACHHER","sm.old":"ursprünglicher Befehl, 4× Zeitlupe","sm.new":"interpolierter Befehl, 4× Zeitlupe","sm.mh":"Gemessen auf identischen Episoden",
    "pl.h":"Simulator","pl.sub":"Die Simulation läuft auf dem Rechner, der diese Seite gestartet hat. Du kannst Szene, Licht, Kistenposition und Bewegungsmodus ändern, den Bildregler abschalten und die Kamera frei durch die Halle bewegen.",
    "pl.ready":"Bereit","pl.start":"Start","pl.pause":"Pause","pl.next":"Nächstes Teil","pl.ep":"Episode","pl.picks":"Gegriffen","pl.places":"Abgelegt","pl.jit":"Befehls-Jitter",
    "pl.hint":"Orbit-Ansicht: ziehen zum Drehen, scrollen zum Zoomen. Ohne GPU läuft die Simulation langsamer als Echtzeit.",
    "pl.scene":"Szene","pl.blender":"Blender-Halle","pl.stock":"Standardzelle","pl.cond":"Bedingung","pl.nominal":"Standard","pl.tierB":"Licht","pl.boxX":"Kiste versetzt","pl.motion":"Bewegung","pl.smooth":"geglättet","pl.orig":"original","pl.servo":"Bildregler","pl.cam":"Kamera","pl.speed":"Tempo","pl.rand":"zufällig",
    "rs.h":"Ergebnisse","rs.sub":"Jede Bedingung lief 200-mal mit Zufalls-Seeds, die im Training nie vorkamen, und das Netz wurde zwischen den Läufen nicht verändert. Als abgelegt zählt ein Lauf, wenn das Teil losgelassen wurde, in der Kiste liegt und zur Ruhe gekommen ist.",
    "rs.c":"Bedingung","rs.pick":"Gegriffen","rs.place":"Abgelegt","rs.nom":"Standardszene","rs.tb":"Zufälliges Licht, Teilfarbe und Kamerahalterung","rs.bx":"Kiste an Positionen, die im Training nicht vorkamen","rs.bl":"Eingerichtete Halle, im Training nie gesehen","rs.no":"Eingerichtete Halle, Bildregler aus (50 Läufe)",
    "rs.note":"Alle Zahlen stammen aus der Simulation mit einer Teileart und einem Förderband-Aufbau. Sie wurden vor der Bewegungsglättung gemessen; die Glättung wurde separat in 44 Läufen geprüft, alle abgelegt.",
    "run.h":"Lokal starten","run.sub":"Requirements installieren, dann den Server starten. Er öffnet diese Seite im Browser mit der Live-Simulation.",
    "of.h":"Auf deinem Rechner starten","of.p":"Diese Seite ist eine statische Kopie, das Video ist also eine Aufnahme. Die interaktive Version läuft auf deinem eigenen Rechner:",
    "ft.t":"SO-101 Pick-and-Place am Förderband in MuJoCo."
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

/* ---- smoothing figures (media/smoothing.json, written by tools/measure_smoothing.py) ---- */
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
  $("#smNote").textContent = de ? "Rot: ursprünglicher Befehl, grün: geglättet. Dieselben 8 Episoden, Gelenkwinkel mit 200 Hz gemessen. Mit Glättung wurden 44 von 44 Testläufen abgelegt." : "Red: original command, green: smoothed. Same 8 episodes, joint angles sampled at 200 Hz. With smoothing, 44 of 44 test runs placed the part.";
  $("#heroCut").textContent = Math.round((1 - SM.smooth.rms_jerk / SM.original.rms_jerk) * 100) + " %";
}
fetch("media/smoothing.json").then(r => r.ok ? r.json() : null).then(j => { SM = j; renderBars(); }).catch(() => {});
/* play the two comparison videos in sync once visible */
const vOld = $("#vOld"), vNew = $("#vNew");
new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { vOld.currentTime = vNew.currentTime = 0; vOld.play().catch(() => {}); vNew.play().catch(() => {}); } else { vOld.pause(); vNew.pause(); } }), { threshold: .4 }).observe($(".vs"));

/* ---- play ---- */
const api = (p, b) => fetch(p, b ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(b) } : undefined).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); });
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
  $("#btnStart").textContent = run.running ? (document.documentElement.lang === "de" ? "Neustart" : "Restart") : "Start";
  $("#btnPause").textContent = run.paused ? (document.documentElement.lang === "de" ? "Weiter" : "Resume") : "Pause";
  $("#screen").classList.toggle("orbit", st.cfg.camera === "orbit");
  $("#log").innerHTML = s.log.map(l => `<span class="dot ${l.place ? "ok" : "bad"}" title="episode ${l.ep}: pick ${l.pick}, place ${l.place}">${l.place ? "✓" : "✗"}</span>`).join("");
}
async function poll() { if (!polling) return; try { render(await api("/api/state")); } catch (e) {} if (polling) setTimeout(poll, 400); }

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
  EN["nav.play"] = "Demo";
  I18N.de["nav.play"] = "Demo";
  EN["hero.cta"] = "Watch demo";
  I18N.de["hero.cta"] = "Demo ansehen";
  EN["pl.h"] = "Recorded demo";
  I18N.de["pl.h"] = "Aufgezeichnete Demo";
  const sub = $("#play .sub");
  sub.dataset.i18n = "of.sub";
  EN["of.sub"] = "A recorded run in the furnished hall, with the wrist camera in the corner. In the live version, which runs on your own computer, the scene, lighting, box position, motion mode, vision controller and camera can all be changed.";
  I18N.de["of.sub"] = "Ein aufgezeichneter Lauf in der eingerichteten Halle, unten rechts das Bild der Handgelenkkamera. In der Live-Version auf deinem eigenen Rechner lassen sich Szene, Licht, Kistenposition, Bewegungsmodus, Bildregler und Kamera einstellen.";
  const play = $(".play");
  play.replaceChildren($("#offline").content.cloneNode(true));
  $$("[data-i18n]", play).forEach(el => { EN[el.dataset.i18n] = EN[el.dataset.i18n] || el.innerHTML; });
  setLang(document.documentElement.lang);
}
api("/api/state").then(st => { render(st); connect(); setTimeout(poll, 400); }).catch(offlineMode);
