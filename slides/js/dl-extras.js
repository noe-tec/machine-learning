/* ============================================================
   DL-EXTRAS — componentes propios del curso de Deep Learning.
   Se carga después de shstyle.js. No depende de Reveal.

   Temporizador de actividad:
     <div class="activity-timer" data-minutes="20"></div>
   Clic: iniciar / pausar. Doble clic: reiniciar.
   En el último minuto cambia de color; al llegar a 0 dice "Time's up".
   ============================================================ */
(function () {
  'use strict';

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
    el.setAttribute('title', 'Click: start / pause · Double-click: reset');
    el.innerHTML = '<span class="timer-time"></span><span class="timer-state"></span>';
    var timeEl = el.querySelector('.timer-time');
    var stateEl = el.querySelector('.timer-state');

    function render() {
      timeEl.textContent = left > 0 ? fmt(left) : "Time's up";
      stateEl.textContent = left === 0 ? '' : tick ? 'running' : left === total ? 'click to start' : 'paused';
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
        '<label class="sd-switch"><input class="sd-scale" type="checkbox"> divide by √d<sub>k</sub></label>' +
      '</div>' +
      '<div class="sd-bars"></div>' +
      '<div class="sd-stats">' +
        '<span>score range: <strong class="sd-range-out"></strong></span>' +
        '<span>largest weight: <strong class="sd-max"></strong></span>' +
        '<span>learning signal: <strong class="sd-grad"></strong></span>' +
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
        Math.min.apply(null, s).toFixed(1) + ' to ' + Math.max.apply(null, s).toFixed(1);
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

  function init() {
    var timers = document.querySelectorAll('.activity-timer');
    for (var i = 0; i < timers.length; i++) setupTimer(timers[i]);
    var demos = document.querySelectorAll('.softmax-demo');
    for (var k = 0; k < demos.length; k++) setupSoftmaxDemo(demos[k]);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
