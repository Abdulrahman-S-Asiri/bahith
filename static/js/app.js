(() => {
  const currentPath = window.location.pathname;
  document.querySelectorAll("nav.primary a").forEach((link) => {
    const href = link.getAttribute("href");
    const active = href === "/" ? currentPath === "/" : currentPath.startsWith(href);
    if (active) {
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
    }
  });

  document.querySelectorAll("form[data-confirm]").forEach((form) => {
    form.addEventListener("submit", (event) => {
      if (!window.confirm(form.dataset.confirm)) event.preventDefault();
    });
  });

  const form = document.querySelector("[data-search-form]");
  const region = document.querySelector("#results-region");
  const loading = document.querySelector("[data-search-loading]");
  if (!form || !region || !loading || !window.fetch || !window.AbortController) return;

  let controller;

  const setLoading = (isLoading) => {
    loading.hidden = !isLoading;
    region.setAttribute("aria-busy", String(isLoading));
  };

  const showError = (status) => {
    const box = document.createElement("div");
    box.className = "state-card state-error";
    box.setAttribute("role", "alert");
    const heading = document.createElement("h2");
    heading.textContent = "تعذّر إكمال البحث";
    const message = document.createElement("p");
    message.textContent = status === 429
      ? "التجربة مشغولة الآن أو وصلت إلى حد البحث المؤقت. انتظر قليلًا ثم أعد المحاولة."
      : status === 503
        ? "الخدمة غير جاهزة للبحث الآن. قد يستغرق تجهيز النموذج قليلًا؛ أعد المحاولة بعد لحظات."
        : "تحقق من الاتصال ثم أعد المحاولة. يمكنك أيضًا إرسال النموذج بالطريقة المعتادة بإعادة تحميل الصفحة.";
    box.append(heading, message);
    region.replaceChildren(box);
  };

  const runSearch = async (pushHistory = true) => {
    if (controller) controller.abort();
    const requestController = new AbortController();
    controller = requestController;
    const params = new URLSearchParams(new FormData(form));
    const pageUrl = `${form.action}?${params.toString()}`;
    setLoading(true);

    try {
      const response = await fetch(`/api/search?${params.toString()}`, {
        headers: { "X-Requested-With": "fetch" },
        signal: requestController.signal,
      });
      if (!response.ok) {
        const error = new Error(`Search failed: ${response.status}`);
        error.status = response.status;
        throw error;
      }
      const html = await response.text();
      if (controller !== requestController) return;
      region.innerHTML = html;
      if (pushHistory) window.history.pushState({}, "", pageUrl);
      else window.history.replaceState({}, "", pageUrl);
    } catch (error) {
      if (controller === requestController && error.name !== "AbortError") showError(error.status);
    } finally {
      if (controller === requestController) setLoading(false);
    }
  };

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
  });

  form.addEventListener("change", (event) => {
    if (event.target.matches("select, input[type='radio']")) runSearch(false);
  });

  window.addEventListener("popstate", () => window.location.reload());
})();
