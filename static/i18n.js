/* ── Interface language ──
 * base.html sets window.VQ_LANG and, for zh-TW, window.VQ_I18N (static/i18n/zh-TW.json,
 * keyed by the English text). t("English text", {slot: value}) returns the string in the
 * visitor's language and fills {slot} placeholders. Anything missing falls back to English.
 */
(() => {
  const lang = window.VQ_LANG || "en";
  const dict = window.VQ_I18N || {};
  window.t = (text, values) => {
    let out = lang === "zh-TW" && Object.prototype.hasOwnProperty.call(dict, text) ? dict[text] : text;
    if (values) {
      out = out.replace(/\{(\w+)\}/g, (match, key) => (key in values ? String(values[key]) : match));
    }
    return out;
  };
  window.VQ_ZH = lang === "zh-TW";
})();
