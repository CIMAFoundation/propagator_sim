(() => {
  "use strict";

  const LANGS = ["en", "it"];
  const STORAGE_KEY = "propagator_lang";
  const dicts = {};
  let currentLang = "en";

  // Browsers configured to block site data throw on localStorage access
  // rather than returning null. That must not escape: init() runs
  // synchronously up to its first await, before window.i18n is assigned,
  // so an uncaught throw here leaves i18n undefined and takes the whole
  // UI down with a ReferenceError in app.js -- not just the language
  // switcher. Remembering the language is a convenience; losing it is
  // the correct degradation.
  function storedLang() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch {
      return null;
    }
  }

  function rememberLang(lang) {
    try {
      localStorage.setItem(STORAGE_KEY, lang);
    } catch {
      /* site data blocked: the choice just won't persist */
    }
  }

  function detectLang() {
    const stored = storedLang();
    if (stored && LANGS.includes(stored)) return stored;
    const browserLang = (navigator.language || navigator.languages?.[0] || "en").slice(0, 2);
    return LANGS.includes(browserLang) ? browserLang : "en";
  }

  async function loadDict(lang) {
    if (dicts[lang]) return dicts[lang];
    try {
      const res = await fetch(`/locales/${lang}.json`);
      // Check res.ok: fetch only rejects on network failure, so a 404 or
      // 500 arrives here as a successful response whose HTML body then
      // throws inside res.json().
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      dicts[lang] = await res.json();
    } catch (err) {
      // Deliberately not cached: storing {} here would make every later
      // lookup (including the English fallback other locales rely on)
      // silently resolve to the raw key forever. Leaving it unset lets a
      // later language switch retry the fetch.
      console.error(`Could not load the ${lang} locale`, err);
      return {};
    }
    return dicts[lang];
  }

  // Returns undefined when no locale defines the key, so callers can
  // tell "not translated" from "translated to something".
  function lookup(key) {
    const dict = dicts[currentLang] || {};
    const enDict = dicts.en || {};
    return dict[key] ?? enDict[key];
  }

  function t(key, vars) {
    let str = lookup(key) ?? key;
    if (vars) {
      for (const [name, value] of Object.entries(vars)) {
        str = str.replaceAll(`{${name}}`, value);
      }
    }
    return str;
  }

  function applyTranslations() {
    // Only overwrite when the key actually resolves. index.html ships
    // readable English in the markup, so if no dictionary loaded (the
    // locale files 404, the network is down, a proxy returns HTML)
    // assigning t()'s raw-key fallback would *destroy* that text and
    // leave the UI reading "run.button" -- strictly worse than leaving
    // the markup alone.
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const text = lookup(el.dataset.i18n);
      if (text !== undefined) el.textContent = text;
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      const title = lookup(el.dataset.i18nTitle);
      if (title !== undefined) el.title = title;
    });
    document.querySelectorAll("[data-lang-btn]").forEach((el) => {
      el.classList.toggle("active", el.dataset.langBtn === currentLang);
    });
  }

  async function setLang(lang) {
    if (!LANGS.includes(lang)) return;
    currentLang = lang;
    rememberLang(lang);
    document.documentElement.lang = lang;
    await loadDict(lang);
    applyTranslations();
    window.dispatchEvent(new CustomEvent("langchange", { detail: { lang } }));
  }

  function getLang() {
    return currentLang;
  }

  async function init() {
    currentLang = detectLang();
    document.documentElement.lang = currentLang;
    await Promise.all([loadDict("en"), currentLang !== "en" ? loadDict(currentLang) : null]);
    applyTranslations();
    window.dispatchEvent(new CustomEvent("langchange", { detail: { lang: currentLang } }));
  }

  window.i18n = { t, setLang, getLang, applyTranslations, ready: init() };
})();
