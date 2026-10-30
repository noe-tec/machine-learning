/* ============================================================
   DL-EXTRAS — componentes propios del curso de Deep Learning.
   Se carga después de shstyle.js. No depende de Reveal.

   Temporizador de actividad:
     <div class="activity-timer" data-minutes="20"></div>
   Clic: iniciar / pausar. Doble clic: reiniciar.
   En el último minuto cambia de color; al llegar a 0 dice "¡Tiempo!".
   Los textos siguen el idioma de <html lang>: "es" (por omisión) o "en".
   Con data-editable muestra botones − y + para quitar o agregar un minuto
   mientras no está corriendo:
     <div class="activity-timer" data-minutes="20" data-editable></div>

   Interactivos de detección: .iou-demo (IoU con cajas arrastrables) y
   .nms-demo (NMS paso a paso con umbral ajustable). Ver references/visuals.md.
   ============================================================ */
(function () {
  'use strict';

  var LANG = (document.documentElement.lang || 'es').slice(0, 2).toLowerCase() === 'en' ? 'en' : 'es';
  var TXT = {
    es: {
      timerTitle: 'Clic: iniciar / pausar · Doble clic: reiniciar',
      minus: 'Quitar un minuto', plus: 'Agregar un minuto',
      done: '¡Tiempo!', running: 'corriendo', start: 'clic para iniciar', paused: 'en pausa',
      divide: 'dividir entre √d<sub>k</sub>', range: 'rango de puntajes', to: ' a ',
      largest: 'peso mayor', signal: 'señal de aprendizaje',
      inter: 'Intersección', union: 'Unión', drag: 'Arrastra las cajas',
      threshold: 'Umbral de IoU', step: 'Siguiente paso', reset: 'Reiniciar',
      nmsStart: 'Ordena por confianza: ', nmsKeep: 'Guardo ', nmsDrop: 'Borro ',
      nmsNone: 'no borro ninguna', nmsDone: 'Fin: quedan ', nmsIou: 'IoU '
    },
    en: {
      timerTitle: 'Click: start / pause · Double-click: reset',
      minus: 'Remove one minute', plus: 'Add one minute',
      done: "Time's up", running: 'running', start: 'click to start', paused: 'paused',
      divide: 'divide by √d<sub>k</sub>', range: 'score range', to: ' to ',
      largest: 'largest weight', signal: 'learning signal',
      inter: 'Intersection', union: 'Union', drag: 'Drag the boxes',
      threshold: 'IoU threshold', step: 'Next step', reset: 'Reset',
      nmsStart: 'Sort by confidence: ', nmsKeep: 'Keep ', nmsDrop: 'Remove ',
      nmsNone: 'nothing removed', nmsDone: 'Done: kept ', nmsIou: 'IoU '
    }
  }[LANG];

  function fmt(sec) {
    var m = Math.floor(sec / 60);
    var s = sec % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function setupTimer(el) {
    var total = Math.round(parseFloat(el.dataset.minutes || '10') * 60);
    var left = total;
    var tick = null;
    var clicks = null;

    el.setAttribute('role', 'timer');
    el.setAttribute('title', TXT.timerTitle);
    var editable = el.hasAttribute('data-editable');
    el.innerHTML = (editable ? '<button type="button" class="timer-adj" data-delta="-60" aria-label="' + TXT.minus + '">\u2212</button>' : '') +
      '<span class="timer-time"></span>' +
      (editable ? '<button type="button" class="timer-adj" data-delta="60" aria-label="' + TXT.plus + '">+</button>' : '') +
      '<span class="timer-state"></span>';
    var timeEl = el.querySelector('.timer-time');
    var stateEl = el.querySelector('.timer-state');

    function render() {
      timeEl.textContent = left > 0 ? fmt(left) : TXT.done;
      stateEl.textContent = left === 0 ? '' : tick ? TXT.running : left === total ? TXT.start : TXT.paused;
      el.classList.toggle('running', !!tick);
      el.classList.toggle('last-minute', left > 0 && left <= 60);
      el.classList.toggle('done', left === 0);
    }

    function stop() {
      if (tick) { clearInterval(tick); tick = null; }
    }

    function toggle() {
      if (left === 0) return;
      if (tick) { stop(); render(); return; }
      tick = setInterval(function () {
        left = Math.max(0, left - 1);
        if (left === 0) stop();
        render();
      }, 1000);
      render();
    }

    function reset() {
      stop();
      left = total;
      render();
    }

    // Ajuste de un minuto (solo con data-editable y con el reloj detenido).
    // Antes de iniciar cambia la duración; en pausa cambia el tiempo restante.
    function adjust(delta) {
      if (tick || left === 0) return;
      if (left === total) {
        total = Math.max(60, total + delta);
        left = total;
      } else {
        left = Math.max(1, left + delta);
      }
      render();
    }
    Array.prototype.forEach.call(el.querySelectorAll('.timer-adj'), function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        adjust(parseInt(btn.dataset.delta, 10));
      });
      btn.addEventListener('dblclick', function (e) { e.stopPropagation(); });
    });

    // Distinguir clic de doble clic sin que el doble clic inicie y pause.
    el.addEventListener('click', function (e) {
      e.stopPropagation();
      if (clicks) return;
      clicks = setTimeout(function () { clicks = null; toggle(); }, 250);
    });
    el.addEventListener('dblclick', function (e) {
      e.stopPropagation();
      if (clicks) { clearTimeout(clicks); clicks = null; }
      reset();
    });

    render();
  }

  /* ------------------------------------------------------------
     Softmax demo: why attention divides by sqrt(d_k).
       <div class="softmax-demo" data-tokens="8" data-seed="7"></div>
     One query and N keys with random components (mean 0, variance 1).
     A slider picks d_k; a switch turns the 1/sqrt(d_k) scaling on or off.
     Shows the N attention weights and the "learning signal":
     max_i p_i (1 - p_i), the largest softmax gradient, scaled so 0.25 = 100%.
     ------------------------------------------------------------ */
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function gaussians(rand, n) {
    var out = [];
    while (out.length < n) {
      var u = 1 - rand(), v = rand();
      var r = Math.sqrt(-2 * Math.log(u));
      out.push(r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v));
    }
    return out.slice(0, n);
  }

  function setupSoftmaxDemo(el) {
    var N = parseInt(el.dataset.tokens || '8', 10);
    var DIMS = [4, 16, 64, 256, 512];
    var MAXD = DIMS[DIMS.length - 1];
    var rand = mulberry32(parseInt(el.dataset.seed || '7', 10));
    var q = gaussians(rand, MAXD);
    var keys = [];
    for (var j = 0; j < N; j++) keys.push(gaussians(rand, MAXD));

    el.innerHTML =
      '<div class="sd-controls">' +
        '<label>d<sub>k</sub> = <strong class="sd-d"></strong>' +
        '<input class="sd-range" type="range" min="0" max="' + (DIMS.length - 1) + '" step="1" value="0"></label>' +
        '<label class="sd-switch"><input class="sd-scale" type="checkbox"> ' + TXT.divide + '</label>' +
      '</div>' +
      '<div class="sd-bars"></div>' +
      '<div class="sd-stats">' +
        '<span>' + TXT.range + ': <strong class="sd-range-out"></strong></span>' +
        '<span>' + TXT.largest + ': <strong class="sd-max"></strong></span>' +
        '<span>' + TXT.signal + ': <strong class="sd-grad"></strong></span>' +
      '</div>';

    var range = el.querySelector('.sd-range'), scale = el.querySelector('.sd-scale');
    var bars = el.querySelector('.sd-bars');
    for (var b = 0; b < N; b++) {
      bars.insertAdjacentHTML('beforeend',
        '<div class="sd-col"><div class="sd-track"><div class="sd-bar"></div></div>' +
        '<span class="sd-val"></span><span class="sd-lab">token ' + (b + 1) + '</span></div>');
    }

    function render() {
      var d = DIMS[parseInt(range.value, 10)];
      var div = scale.checked ? Math.sqrt(d) : 1;
      var s = keys.map(function (k) {
        var dot = 0;
        for (var i = 0; i < d; i++) dot += q[i] * k[i];
        return dot / div;
      });
      var m = Math.max.apply(null, s);
      var e = s.map(function (x) { return Math.exp(x - m); });
      var z = e.reduce(function (a, c) { return a + c; }, 0);
      var p = e.map(function (x) { return x / z; });
      var pmax = Math.max.apply(null, p);
      var grad = Math.max.apply(null, p.map(function (x) { return x * (1 - x); }));
      var cols = bars.querySelectorAll('.sd-col');
      for (var i = 0; i < N; i++) {
        cols[i].querySelector('.sd-bar').style.height = (p[i] * 100).toFixed(1) + '%';
        cols[i].querySelector('.sd-val').textContent = p[i] < 0.005 ? '0' : p[i].toFixed(2);
        cols[i].classList.toggle('sd-top', p[i] === pmax);
      }
      el.querySelector('.sd-d').textContent = d;
      el.querySelector('.sd-range-out').textContent =
        Math.min.apply(null, s).toFixed(1) + TXT.to + Math.max.apply(null, s).toFixed(1);
      el.querySelector('.sd-max').textContent = (pmax * 100).toFixed(0) + '%';
      el.querySelector('.sd-grad').textContent = (grad / 0.25 * 100).toFixed(0) + '%';
      el.classList.toggle('sd-saturated', pmax > 0.9);
    }

    // Keep the arrow keys for the slider, not for changing slides.
    el.addEventListener('keydown', function (ev) { ev.stopPropagation(); });
    range.addEventListener('input', render);
    scale.addEventListener('change', render);
    render();
  }


  /* ------------------------------------------------------------
     Cajas en una rejilla (compartido por iou-demo y nms-demo).
     Coordenadas en unidades de la rejilla: [x1, y1, x2, y2].
     ------------------------------------------------------------ */
  var SVGNS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function boxIou(a, b) {
    var ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    var iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    var inter = ix * iy;
    var area = function (r) { return (r[2] - r[0]) * (r[3] - r[1]); };
    var union = area(a) + area(b) - inter;
    return { inter: inter, union: union, iou: union > 0 ? inter / union : 0, ix: ix, iy: iy };
  }
  function drawGrid(svg, cols, rows, u) {
    var g = svgEl('g', { 'class': 'bx-grid' }, svg);
    for (var x = 0; x <= cols; x++) svgEl('line', { x1: x * u, y1: 0, x2: x * u, y2: rows * u }, g);
    for (var y = 0; y <= rows; y++) svgEl('line', { x1: 0, y1: y * u, x2: cols * u, y2: y * u }, g);
    return g;
  }
  function fmt2(v) { return v.toFixed(2); }

  /* ------------------------------------------------------------
     IoU demo: dos cajas que se arrastran sobre una rejilla.
       <div class="iou-demo" data-a="1,1,5,5" data-b="2,1,6,5" data-grid="12,7"></div>
     Muestra intersección, unión e IoU con áreas en celdas.
     ------------------------------------------------------------ */
  function setupIouDemo(el) {
    var grid = (el.dataset.grid || '12,7').split(',').map(Number);
    var U = 44, cols = grid[0], rows = grid[1];
    var boxes = [(el.dataset.a || '1,1,5,5').split(',').map(Number),
                 (el.dataset.b || '2,1,6,5').split(',').map(Number)];
    el.innerHTML = '';
    var svg = svgEl('svg', { viewBox: '-4 -4 ' + (cols * U + 8) + ' ' + (rows * U + 8), 'class': 'iou-svg' }, el);
    drawGrid(svg, cols, rows, U);
    var interR = svgEl('rect', { 'class': 'iou-inter' }, svg);
    var rects = boxes.map(function (b, i) {
      var r = svgEl('rect', { 'class': 'iou-box iou-box-' + (i ? 'b' : 'a'), rx: 4 }, svg);
      var t = svgEl('text', { 'class': 'iou-tag' }, svg);
      t.textContent = i ? 'B' : 'A';
      return { r: r, t: t };
    });
    var panel = document.createElement('div');
    panel.className = 'iou-panel';
    panel.innerHTML = '<p>' + TXT.inter + ': <strong class="iou-i"></strong></p>' +
      '<p>' + TXT.union + ': <strong class="iou-u"></strong></p>' +
      '<p class="iou-big">IoU = <strong class="iou-v"></strong></p>' +
      '<p class="iou-hint">' + TXT.drag + '</p>';
    el.appendChild(panel);

    function render() {
      boxes.forEach(function (b, i) {
        rects[i].r.setAttribute('x', b[0] * U); rects[i].r.setAttribute('y', b[1] * U);
        rects[i].r.setAttribute('width', (b[2] - b[0]) * U); rects[i].r.setAttribute('height', (b[3] - b[1]) * U);
        rects[i].t.setAttribute('x', b[0] * U + 8); rects[i].t.setAttribute('y', b[1] * U + 28);
      });
      var r = boxIou(boxes[0], boxes[1]);
      var x1 = Math.max(boxes[0][0], boxes[1][0]), y1 = Math.max(boxes[0][1], boxes[1][1]);
      interR.setAttribute('x', x1 * U); interR.setAttribute('y', y1 * U);
      interR.setAttribute('width', r.ix * U); interR.setAttribute('height', r.iy * U);
      panel.querySelector('.iou-i').textContent = r.inter;
      panel.querySelector('.iou-u').textContent = r.union;
      panel.querySelector('.iou-v').textContent = fmt2(r.iou);
    }

    var drag = null;
    function pt(ev) {
      var p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
      return p.matrixTransform(svg.getScreenCTM().inverse());
    }
    rects.forEach(function (o, i) {
      o.r.addEventListener('pointerdown', function (ev) {
        ev.stopPropagation(); ev.preventDefault();
        var p = pt(ev);
        drag = { i: i, dx: p.x / U - boxes[i][0], dy: p.y / U - boxes[i][1] };
        o.r.setPointerCapture(ev.pointerId);
      });
      o.r.addEventListener('pointermove', function (ev) {
        if (!drag || drag.i !== i) return;
        var p = pt(ev), b = boxes[i], w = b[2] - b[0], h = b[3] - b[1];
        var nx = Math.round(p.x / U - drag.dx), ny = Math.round(p.y / U - drag.dy);
        nx = Math.max(0, Math.min(cols - w, nx)); ny = Math.max(0, Math.min(rows - h, ny));
        boxes[i] = [nx, ny, nx + w, ny + h];
        render();
      });
      o.r.addEventListener('pointerup', function () { drag = null; });
    });
    el.addEventListener('keydown', function (ev) { ev.stopPropagation(); });
    render();
  }

  /* ------------------------------------------------------------
     NMS demo: supresión de no máximos paso a paso.
       <div class="nms-demo" data-grid="12,9" data-threshold="0.5"
            data-boxes="A:1,1,5,5,0.90,tl;B:2,1,6,5,0.75,tr"></div>
     El sexto campo (opcional) es la esquina de la etiqueta: tl, tr, bl, br (dentro) o al, ar (arriba).
     Botón "Siguiente paso" avanza; el deslizador cambia el umbral y reinicia.
     ------------------------------------------------------------ */
  function setupNmsDemo(el) {
    var grid = (el.dataset.grid || '12,9').split(',').map(Number);
    var U = 40, cols = grid[0], rows = grid[1];
    var boxes = (el.dataset.boxes || '').split(';').filter(Boolean).map(function (s) {
      var p = s.split(':'), raw = p[1].split(','), v = raw.slice(0, 5).map(Number);
      return { name: p[0], b: v.slice(0, 4), score: v[4], corner: raw[5] || 'tl' };
    });
    el.innerHTML =
      '<div class="nms-controls">' +
        '<label>' + TXT.threshold + ' = <strong class="nms-th"></strong>' +
        '<input class="nms-range" type="range" min="0.05" max="0.95" step="0.05"></label>' +
        '<button type="button" class="nms-btn nms-step">' + TXT.step + '</button>' +
        '<button type="button" class="nms-btn nms-reset">' + TXT.reset + '</button>' +
      '</div><div class="nms-body"></div>';
    var range = el.querySelector('.nms-range');
    range.value = el.dataset.threshold || '0.5';
    var body = el.querySelector('.nms-body');
    var svg = svgEl('svg', { viewBox: '-4 -4 ' + (cols * U + 8) + ' ' + (rows * U + 8), 'class': 'nms-svg' });
    body.appendChild(svg);
    var log = document.createElement('ol'); log.className = 'nms-log'; body.appendChild(log);
    drawGrid(svg, cols, rows, U);
    boxes.forEach(function (o) {
      o.g = svgEl('g', { 'class': 'nms-box' }, svg);
      svgEl('rect', { x: o.b[0] * U, y: o.b[1] * U, width: (o.b[2] - o.b[0]) * U, height: (o.b[3] - o.b[1]) * U, rx: 4 }, o.g);
      // Etiqueta en la esquina pedida para que no se encimen: tl, tr, bl, br (dentro)
      // o al, ar (arriba de la caja, por fuera).
      var right = o.corner.charAt(1) === 'r', v = o.corner.charAt(0);
      var t = svgEl('text', {
        x: (right ? o.b[2] * U - 6 : o.b[0] * U + 6),
        y: (v === 'b' ? o.b[3] * U - 8 : v === 'a' ? o.b[1] * U - 7 : o.b[1] * U + 24),
        'text-anchor': right ? 'end' : 'start'
      }, o.g);
      t.textContent = o.name + ' ' + o.score.toFixed(2);
    });

    var queue, kept, steps;
    function reset() {
      queue = boxes.slice().sort(function (a, b) { return b.score - a.score; });
      kept = []; steps = 0;
      boxes.forEach(function (o) { o.state = 'pending'; });
      log.innerHTML = '';
      var li = document.createElement('li');
      li.textContent = TXT.nmsStart + queue.map(function (o) { return o.name; }).join(', ');
      log.appendChild(li);
      render();
    }
    function step() {
      if (!queue.length) return;
      var th = parseFloat(range.value);
      var best = queue.shift(); best.state = 'kept'; kept.push(best);
      var dropped = [];
      queue = queue.filter(function (o) {
        var v = boxIou(best.b, o.b).iou;
        if (v > th) { o.state = 'removed'; dropped.push(o.name + ' (' + TXT.nmsIou + fmt2(v) + ')'); return false; }
        return true;
      });
      var li = document.createElement('li');
      li.textContent = TXT.nmsKeep + best.name + ' (' + best.score.toFixed(2) + '). ' +
        (dropped.length ? TXT.nmsDrop + dropped.join(', ') : TXT.nmsNone) + '.';
      log.appendChild(li);
      if (!queue.length) {
        var end = document.createElement('li'); end.className = 'nms-end';
        end.textContent = TXT.nmsDone + kept.map(function (o) { return o.name; }).join(', ');
        log.appendChild(end);
      }
      render();
    }
    function render() {
      el.querySelector('.nms-th').textContent = parseFloat(range.value).toFixed(2);
      boxes.forEach(function (o) { o.g.setAttribute('class', 'nms-box nms-' + o.state); });
      el.querySelector('.nms-step').disabled = !queue.length;
    }
    el.querySelector('.nms-step').addEventListener('click', function (e) { e.stopPropagation(); step(); });
    el.querySelector('.nms-reset').addEventListener('click', function (e) { e.stopPropagation(); reset(); });
    range.addEventListener('input', reset);
    el.addEventListener('keydown', function (ev) { ev.stopPropagation(); });
    reset();
  }

  function init() {
    var timers = document.querySelectorAll('.activity-timer');
    for (var i = 0; i < timers.length; i++) setupTimer(timers[i]);
    var demos = document.querySelectorAll('.softmax-demo');
    for (var k = 0; k < demos.length; k++) setupSoftmaxDemo(demos[k]);
    Array.prototype.forEach.call(document.querySelectorAll('.iou-demo'), setupIouDemo);
    Array.prototype.forEach.call(document.querySelectorAll('.nms-demo'), setupNmsDemo);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
