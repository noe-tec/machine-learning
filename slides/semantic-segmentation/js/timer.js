/* ============================================================
   TIMER — temporizador de actividad para shstyle.
   Se carga después de shstyle.js. No depende de Reveal.

   <div class="activity-timer" data-minutes="20"></div>
   Clic: iniciar / pausar. Doble clic: reiniciar.
   En el último minuto cambia de color; al llegar a 0 dice "¡Tiempo!".
   Los textos siguen el idioma de <html lang>: "es" o "en".
   Con data-editable muestra botones − y + para quitar o agregar un minuto
   mientras no está corriendo:
     <div class="activity-timer" data-minutes="20" data-editable></div>
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
      largest: 'peso mayor', signal: 'señal de aprendizaje'
    },
    en: {
      timerTitle: 'Click: start / pause · Double-click: reset',
      minus: 'Remove one minute', plus: 'Add one minute',
      done: "Time's up", running: 'running', start: 'click to start', paused: 'paused',
      divide: 'divide by √d<sub>k</sub>', range: 'score range', to: ' to ',
      largest: 'largest weight', signal: 'learning signal'
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

  function init() {
    var timers = document.querySelectorAll('.activity-timer');
    for (var i = 0; i < timers.length; i++) setupTimer(timers[i]);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
