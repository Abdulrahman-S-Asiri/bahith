// باحث — minimal frontend enhancements
(() => {
  // Highlight current nav item
  const path = window.location.pathname;
  document.querySelectorAll('nav.primary a').forEach((a) => {
    const href = a.getAttribute('href');
    if (href === '/' ? path === '/' : path.startsWith(href)) {
      a.classList.add('active');
    }
  });

  // Top-K stepper widget
  document.querySelectorAll('[data-stepper]').forEach((el) => {
    const out = el.querySelector('output');
    const input = el.querySelector('input[type=hidden]');
    const min = Number(el.dataset.min || 1);
    const max = Number(el.dataset.max || 10);
    const set = (v) => {
      const n = Math.max(min, Math.min(max, v));
      out.value = n;
      input.value = n;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    };
    el.querySelector('[data-step="-1"]').addEventListener('click', () => set(Number(out.value) - 1));
    el.querySelector('[data-step="1"]').addEventListener('click',  () => set(Number(out.value) + 1));
  });

  // Dim pill toggle visual state (radios already handle it via CSS,
  // but we keep an explicit class for nicer focus visuals)
  document.querySelectorAll('.dim-pills').forEach((g) => {
    const sync = () => {
      g.querySelectorAll('label').forEach((l) => {
        const r = document.getElementById(l.getAttribute('for'));
        l.classList.toggle('active', r && r.checked);
      });
    };
    g.addEventListener('change', sync);
    sync();
  });
})();
