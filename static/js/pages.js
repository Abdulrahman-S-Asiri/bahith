/* GitHub Pages UI. Retrieval stays on the configured, independent model service. */
(() => {
  "use strict";
  const basePath = document.querySelector('meta[name="bahith-base-path"]')?.content;
  if (!basePath) return;
  const apiSetting = document.querySelector('meta[name="bahith-api-base"]')?.content || "";
  let apiBase;
  try {
    const parsed = new URL(apiSetting);
    if (parsed.protocol === "https:" && !parsed.username && !parsed.password &&
        parsed.pathname === "/" && !parsed.search && !parsed.hash) apiBase = parsed.origin;
  } catch { /* An unconfigured service must never turn into a request to another site. */ }

  const localUrl = (path) => `${basePath}${path}`;
  const element = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  };
  const link = (text, path) => {
    const node = element("a", "text-link", text);
    node.href = localUrl(path);
    return node;
  };
  if (!apiBase) {
    const notice = document.querySelector('aside[aria-label="عن التجربة العامة"]');
    if (notice) notice.append(element("p", "", "التصفح متاح الآن. خدمة البحث المباشر قيد التجهيز."));
  }
  document.querySelectorAll("nav.primary a").forEach((node) => {
    const path = new URL(node.href).pathname;
    if (path === basePath ? location.pathname === path : location.pathname.startsWith(path)) {
      node.classList.add("active");
      node.setAttribute("aria-current", "page");
    }
  });

  const form = document.querySelector("[data-search-form]");
  const region = document.querySelector("#results-region");
  const loading = document.querySelector("[data-search-loading]");
  if (!form || !region || !loading) return;
  form.elements.q.maxLength = 2000;
  let controller;
  const state = (title, message, failed = false) => {
    const box = element("div", `state-card${failed ? " state-error" : ""}`);
    if (failed) box.setAttribute("role", "alert");
    box.append(element("h2", "", title), element("p", "", message));
    region.replaceChildren(box);
  };
  const showError = (status) => {
    const messages = {
      422: "اكتب سؤالًا لا يتجاوز 2000 حرف، ثم أعد المحاولة.",
      429: "خدمة البحث مشغولة الآن؛ انتظر قليلًا ثم أعد المحاولة.",
      503: "النموذج غير جاهز الآن. أعد المحاولة بعد قليل؛ يمكنك تصفّح النصوص ومصادرها في هذه الأثناء.",
    };
    state("تعذّر إكمال البحث", messages[status] ||
      "لم نتمكن من الاتصال بخدمة البحث. يمكنك تصفّح النصوص ومصادرها، ثم إعادة المحاولة لاحقًا.", true);
  };
  const renderVector = (embedding) => {
    if (!embedding || !Array.isArray(embedding.buckets) || !Number.isFinite(embedding.norm)) return null;
    const box = element("details", "embedding-viz");
    box.append(element("summary", "", "عرض التوزيع الرقمي لمتجه الاستعلام"));
    const head = element("div", "viz-head");
    head.append(element("p", "", `${embedding.dim} بُعدًا · معيار المتجه ${embedding.norm.toFixed(3)}`));
    box.append(head);
    const bars = element("div", "bars");
    bars.setAttribute("role", "img");
    bars.setAttribute("aria-label", "توزيع مقادير متجه الاستعلام على مجموعات من الأبعاد");
    for (const bucket of embedding.buckets.slice(0, 32)) {
      if (!Number.isFinite(bucket.height) || !Number.isFinite(bucket.value)) continue;
      const wrap = element("div", "bar-wrap");
      const bar = element("div", "bar");
      bar.style.height = `${Math.max(0, Math.min(100, bucket.height))}%`;
      bar.title = `الأبعاد ${bucket.start}–${bucket.end}، القيمة ${bucket.value.toFixed(3)}`;
      wrap.append(bar, element("span", "ltr", bucket.start));
      bars.append(wrap);
    }
    box.append(bars, element("p", "viz-note", "يعرض الرسم توزيعًا عدديًا لمقادير المتجه بعد الاقتطاع وإعادة التطبيع. الأعمدة لا تشرح معنى كلمات أو مفاهيم بعينها."));
    return box;
  };
  const renderResults = (payload) => {
    if (!Array.isArray(payload.results)) throw new Error("Invalid service response");
    if (!payload.results.length) {
      state("لا توجد نتائج", "جرّب صياغة أخرى للسؤال أو وسّع نطاق البحث.");
      return;
    }
    const fragment = document.createDocumentFragment();
    const heading = element("div", "results-heading");
    const titles = element("div");
    titles.append(element("p", "eyebrow", "النتائج"), element("h2", "", "أقرب المقاطع إلى سؤالك"));
    const meta = element("p", "results-meta");
    for (const text of [`${payload.results.length} نتائج`, `${payload.corpus_size} مقطعًا في النطاق`, `${payload.dim} بُعدًا`]) {
      meta.append(element("span", "", text));
    }
    if (payload.query_cached) meta.append(element("span", "", "أُعيد استخدام متجه الاستعلام"));
    heading.append(titles, meta);
    fragment.append(heading, element("p", "ranking-note", payload.method === "keyword"
      ? "الدرجات تقيس التطابق الكلمي داخل هذا النطاق."
      : "الدرجات ترتّب المقاطع داخل هذا النطاق ولا تمثل نسبة ثقة أو حكمًا على صحة النص."));
    const vector = renderVector(payload.embedding);
    if (vector) fragment.append(vector);
    const list = element("ol", "results");
    payload.results.slice(0, 10).forEach((result, index) => {
      const item = element("li", "result-card");
      item.append(element("div", "result-order", index + 1));
      const article = element("article");
      const top = element("div", "result-topline");
      const score = element("span", "score");
      score.append(element("span", "", payload.method === "hybrid" ? "درجة الترتيب " : payload.method === "keyword" ? "درجة التطابق " : "درجة التشابه "),
        element("b", "ltr", Number.isFinite(result.score) ? result.score.toFixed(3) : "—"));
      top.append(element("span", "source-name", result.source_title || "المجموعة التجريبية"), score);
      article.append(top, element("p", "result-text", result.text));
      const source = element("div", "result-source");
      const info = element("div");
      info.append(element("span", "demo-badge", "مثال مضمّن"));
      if (result.filename) info.append(element("span", "", result.filename));
      if (result.page) info.append(element("span", "", `الصفحة ${result.page}`));
      if (result.ordinal) info.append(element("span", "", `المقطع ${result.ordinal}`));
      const links = element("div", "result-links");
      if (result.document_id === "demo" && Number.isSafeInteger(result.id) && result.id > 0) {
        links.append(link("عرض المقطع", `passages/${result.id}/`), link("فتح المصدر", "documents/demo/source/"));
      }
      source.append(info, links);
      article.append(source);
      item.append(article);
      list.append(item);
    });
    fragment.append(list);
    region.replaceChildren(fragment);
  };
  const restoreForm = () => {
    const params = new URLSearchParams(location.search);
    for (const name of ["q", "method", "document_id", "dim", "top_k"]) {
      const field = form.elements.namedItem(name);
      const value = params.get(name);
      if (!field || value === null) continue;
      if (name === "q") field.value = value;
      else if (name === "dim") {
        for (const radio of form.querySelectorAll('input[name="dim"]')) radio.checked = radio.value === value;
        if (!form.querySelector('input[name="dim"]:checked')) form.querySelector('input[value="1024"]').checked = true;
      } else if ([...field.options].some((option) => option.value === value)) field.value = value;
    }
  };
  const runSearch = async (pushHistory = true) => {
    if (controller) controller.abort();
    const request = new AbortController();
    controller = request;
    loading.hidden = true;
    region.setAttribute("aria-busy", "false");
    const params = new URLSearchParams(new FormData(form));
    const query = (params.get("q") || "").trim();
    if (!query) { state("ابدأ بسؤال", "اكتب سؤالك في الأعلى للبحث في النصوص المضمّنة."); return; }
    if (query.length > 2000) { showError(422); return; }
    params.set("q", query);
    const pageUrl = `${form.action}?${params}`;
    if (pushHistory) history.pushState({}, "", pageUrl);
    else history.replaceState({}, "", pageUrl);
    if (!apiBase) {
      state("خدمة البحث قيد التجهيز", "الموقع متاح للتصفح، وسيتاح البحث بعد ربط خدمة النموذج المستقلة.");
      return;
    }
    loading.hidden = false;
    region.setAttribute("aria-busy", "true");
    const timeout = setTimeout(() => request.abort(), 30000);
    try {
      const response = await fetch(`${apiBase}/api/query?${params}`, {
        signal: request.signal, credentials: "omit", referrerPolicy: "no-referrer",
      });
      if (!response.ok) throw Object.assign(new Error("Search unavailable"), {status: response.status});
      const payload = await response.json();
      if (controller !== request) return;
      if (payload.status === "error" || payload.dim !== Number(params.get("dim")) || payload.method !== params.get("method")) {
        throw new Error("Inconsistent service response");
      }
      renderResults(payload);
    } catch (error) {
      if (controller === request) showError(error.status);
    } finally {
      clearTimeout(timeout);
      if (controller === request) {
        loading.hidden = true;
        region.setAttribute("aria-busy", "false");
      }
    }
  };
  form.addEventListener("submit", (event) => { event.preventDefault(); runSearch(); });
  form.addEventListener("change", (event) => {
    if (event.target.matches("select, input[type='radio']")) runSearch(false);
  });
  window.addEventListener("popstate", () => location.reload());
  restoreForm();
  if (new URLSearchParams(location.search).has("q")) runSearch(false);
})();
