// Generic DOM -> structured observation extractor (agentrs).
//
// This file is a single JavaScript *function expression*; Python evaluates it
// with Playwright's ``page.evaluate(source, options)``. It does not depend on
// any page globals, works on arbitrary pages and never executes model output.
//
// Modes
//   {mode: "extract", next, prefix, viewportOnly, maxElements, reportFocus}
//       Walks the DOM in document order and returns the visible interactive and
//       informative elements. Elements get a ``data-agentrs-id`` attribute
//       (``<prefix><n>``) the first time they are observed; the id sticks to the
//       DOM node for the lifetime of the document, so ids are stable within a
//       page load and new documents (navigations) get fresh ids. ``next`` is the
//       caller-owned counter, so ids are unique across page loads of an episode.
//   {mode: "status", id, viewportOnly}
//       Actionability facts about one element (found / visible / disabled /
//       role / editable / options / full text) for the action dispatcher.
//
// Subtrees marked ``data-agentrs-ignore`` (tooling overlays) are skipped.
(opts) => {
  const ID_ATTR = 'data-agentrs-id';
  const IGNORE_ATTR = 'data-agentrs-ignore';
  const ID_RE = /^[A-Za-z][A-Za-z0-9_-]{0,31}$/;
  const o = Object.assign({
    mode: 'extract', next: 1, prefix: 'a', viewportOnly: true, maxElements: 250,
    reportFocus: true, maxName: 200, id: null,
  }, opts || {});

  const INTERACTIVE = new Set([
    'button', 'link', 'textbox', 'searchbox', 'checkbox', 'radio', 'combobox', 'listbox', 'menuitem',
    'menuitemcheckbox', 'menuitemradio', 'switch', 'tab', 'slider', 'spinbutton', 'menubutton', 'treeitem',
  ]);
  const INFORMATIVE = new Set([
    'heading', 'alert', 'status', 'dialog', 'alertdialog', 'img', 'tooltip', 'note', 'log', 'timer',
    'progressbar', 'meter', 'marquee',
  ]);
  const NAME_FROM_CONTENT = new Set([
    'button', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'switch', 'tab', 'checkbox', 'radio',
    'treeitem', 'menubutton', 'heading', 'tooltip', 'option',
  ]);
  const SKIP_TAGS = new Set([
    'SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'HEAD', 'META', 'LINK', 'TITLE', 'OPTION', 'OPTGROUP',
    'DATALIST', 'BR', 'HR', 'IFRAME', 'OBJECT', 'EMBED', 'svg', 'SVG', 'CANVAS',
  ]);
  const INTERACTIVE_SELECTOR = [
    'a[href]', 'button', 'input:not([type=hidden])', 'select', 'textarea', 'summary', '[contenteditable=""]',
    '[contenteditable=true]', ...Array.from(INTERACTIVE).map((r) => `[role=${r}]`),
  ].join(',');

  const norm = (s) => String(s == null ? '' : s).replace(/\s+/g, ' ').trim();
  const clip = (s, n) => {
    s = norm(s);
    const m = n || o.maxName;
    return s.length > m ? s.slice(0, m - 1) + '…' : s;
  };

  function implicitRole(el) {
    const tag = el.tagName.toLowerCase();
    switch (tag) {
      case 'a': case 'area': return el.hasAttribute('href') ? 'link' : null;
      case 'button': {
        const hp = (el.getAttribute('aria-haspopup') || '').toLowerCase();
        return (hp === 'menu' || hp === 'true') ? 'menubutton' : 'button';
      }
      case 'summary': return 'button';
      case 'input': {
        const t = (el.getAttribute('type') || 'text').toLowerCase();
        if (t === 'hidden') return null;
        if (['button', 'submit', 'reset', 'image', 'file', 'color'].includes(t)) return 'button';
        if (t === 'checkbox') return 'checkbox';
        if (t === 'radio') return 'radio';
        if (t === 'range') return 'slider';
        if (t === 'number') return 'spinbutton';
        if (t === 'search') return 'searchbox';
        return 'textbox';
      }
      case 'textarea': return 'textbox';
      case 'select': return (el.multiple || el.size > 1) ? 'listbox' : 'combobox';
      case 'h1': case 'h2': case 'h3': case 'h4': case 'h5': case 'h6': return 'heading';
      case 'img': return norm(el.getAttribute('alt')) ? 'img' : null;
      case 'dialog': return 'dialog';
      case 'progress': return 'progressbar';
      case 'meter': return 'meter';
      case 'output': return 'status';
    }
    const ce = el.getAttribute('contenteditable');
    if (ce !== null && ce !== 'false' && el.isContentEditable) return 'textbox';
    return null;
  }

  function roleOf(el) {
    const explicit = (el.getAttribute('role') || '').trim().split(/\s+/)[0].toLowerCase();
    if (explicit && !['none', 'presentation', 'generic'].includes(explicit)) {
      if (explicit === 'button') {
        const hp = (el.getAttribute('aria-haspopup') || '').toLowerCase();
        if (hp === 'menu' || hp === 'true') return 'menubutton';
      }
      return explicit;
    }
    return implicitRole(el);
  }

  function styleVisible(el) {
    if (typeof el.checkVisibility === 'function') {
      if (!el.checkVisibility({checkOpacity: true, checkVisibilityCSS: true})) return false;
    } else {
      const cs = getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility !== 'visible' || Number(cs.opacity) === 0) return false;
    }
    const r = el.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  }
  // An element is "on screen" when at least half of it (or, for elements larger
  // than the viewport, half of the viewport) is inside the viewport.
  function inViewport(el) {
    const r = el.getBoundingClientRect();
    const vh = window.innerHeight, vw = window.innerWidth;
    const visH = Math.min(r.bottom, vh) - Math.max(r.top, 0);
    const visW = Math.min(r.right, vw) - Math.max(r.left, 0);
    if (visH <= 0 || visW <= 0) return false;
    return visH >= 0.5 * Math.min(r.height, vh) - 0.5 && visW >= 0.5 * Math.min(r.width, vw) - 0.5;
  }
  const visible = (el) => styleVisible(el) && (!o.viewportOnly || inViewport(el));

  function textOfIds(ids) {
    return ids.split(/\s+/).map((id) => {
      const e = id && document.getElementById(id);
      return e ? (e.innerText || e.textContent || '') : '';
    }).join(' ');
  }
  function labelText(el) {
    if (!el.labels || !el.labels.length) return '';
    return Array.from(el.labels).map((l) => {
      const c = l.cloneNode(true);  // drop nested controls so their text does not leak into the name
      c.querySelectorAll('input,select,textarea,button').forEach((x) => x.remove());
      return c.textContent || '';
    }).join(' ');
  }

  function accName(el, role) {
    const lb = el.getAttribute('aria-labelledby');
    if (lb) { const t = norm(textOfIds(lb)); if (t) return clip(t); }
    const al = norm(el.getAttribute('aria-label'));
    if (al) return clip(al);
    const tag = el.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
      const type = (el.type || '').toLowerCase();
      if (['button', 'submit', 'reset'].includes(type)) {
        return clip(el.value || (type === 'submit' ? 'Submit' : type === 'reset' ? 'Reset' : ''));
      }
      const lt = norm(labelText(el));
      if (lt) return clip(lt);
      if (type === 'image') return clip(el.getAttribute('alt'));
      return clip(el.getAttribute('title') || el.getAttribute('placeholder') || '');
    }
    if (tag === 'IMG') return clip(el.getAttribute('alt'));
    if (NAME_FROM_CONTENT.has(role) || INFORMATIVE.has(role)) {
      const t = norm(el.innerText || el.textContent);
      if (t) return clip(t);
    }
    if (el.labels && el.labels.length) { const t = norm(labelText(el)); if (t) return clip(t); }
    return clip(el.getAttribute('title') || el.getAttribute('placeholder') || '');
  }

  function valueOf(el, role) {
    if (el instanceof HTMLSelectElement) {
      if (el.multiple) return Array.from(el.selectedOptions).map((x) => norm(x.label || x.text)).join(', ');
      const s = el.selectedOptions;
      return (s.length && el.value !== '') ? norm(s[0].label || s[0].text) : '';
    }
    if (el instanceof HTMLInputElement) {
      const t = el.type;
      if (['checkbox', 'radio', 'button', 'submit', 'reset', 'image', 'file', 'color'].includes(t)) return null;
      if (t === 'password') return '•'.repeat(el.value.length);  // never expose secrets
      return el.value;
    }
    if (el instanceof HTMLTextAreaElement) return el.value;
    if (role === 'textbox' && el.isContentEditable) return el.innerText;
    if (['slider', 'spinbutton', 'progressbar', 'meter'].includes(role)) {
      const vn = el.getAttribute('aria-valuetext') || el.getAttribute('aria-valuenow');
      if (vn !== null) return vn;
      return (el.value !== undefined && el.value !== null) ? String(el.value) : null;
    }
    if (role === 'combobox') return el.getAttribute('aria-valuetext') || norm(el.innerText) || '';
    return null;
  }

  function checkedOf(el, role) {
    if (el instanceof HTMLInputElement && (el.type === 'checkbox' || el.type === 'radio')) return el.checked;
    const ac = el.getAttribute('aria-checked');
    if (['checkbox', 'switch', 'radio', 'menuitemcheckbox', 'menuitemradio'].includes(role)) {
      return ac === 'true' || ac === 'mixed';
    }
    if (ac !== null) return ac === 'true' || ac === 'mixed';
    const ap = el.getAttribute('aria-pressed');
    if (role === 'button' && ap !== null) return ap === 'true' || ap === 'mixed';
    return null;
  }

  function expandedOf(el) {
    const ae = el.getAttribute('aria-expanded');
    if (ae !== null) return ae === 'true';
    if (el.tagName === 'SUMMARY' && el.parentElement && el.parentElement.tagName === 'DETAILS') {
      return el.parentElement.open;
    }
    return null;
  }

  const disabledOf = (el) => {
    try { if (el.matches(':disabled')) return true; } catch (e) { /* non-form element */ }
    return !!el.closest('[aria-disabled="true"]');
  };

  function optionsOf(el, role) {
    if (el instanceof HTMLSelectElement) {
      return Array.from(el.options).filter((op) => !op.hidden && norm(op.label || op.text))
        .map((op) => norm(op.label || op.text));
    }
    if (role === 'listbox') {
      return Array.from(el.querySelectorAll('[role=option]')).map((op) => accName(op, 'option')).filter(Boolean);
    }
    return null;
  }

  const isEditable = (el, role) => {
    if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
      return ['textbox', 'searchbox', 'spinbutton'].includes(role) && !el.readOnly;
    }
    return (role === 'textbox' || role === 'searchbox') && el.isContentEditable;
  };

  const childrenOf = (el) => {
    const kids = Array.from(el.children);
    if (el.shadowRoot) kids.push(...Array.from(el.shadowRoot.children));
    return kids;
  };

  function directText(el) {
    let s = '';
    for (const c of el.childNodes) if (c.nodeType === Node.TEXT_NODE) s += c.nodeValue + ' ';
    return norm(s);
  }

  // ------------------------------------------------------------------ status mode
  function deepFind(root, id) {
    const hit = root.querySelector(`[${ID_ATTR}="${id}"]`);
    if (hit) return hit;
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) { const h = deepFind(el.shadowRoot, id); if (h) return h; }
    }
    return null;
  }

  if (o.mode === 'status') {
    const id = String(o.id || '');
    if (!ID_RE.test(id)) return {found: false};
    const el = deepFind(document, id);
    if (!el) return {found: false};
    const role = roleOf(el) || 'text';
    return {
      found: true,
      visible: visible(el),
      disabled: disabledOf(el),
      role,
      editable: isEditable(el, role),
      isSelect: el instanceof HTMLSelectElement,
      options: optionsOf(el, role),
      text: norm(el.innerText || el.textContent || accName(el, role)).slice(0, 4000),
    };
  }

  // ------------------------------------------------------------------ extract mode
  const out = [];
  const seen = new Set();
  let next = Math.max(1, Number(o.next) || 1);

  function assignId(el) {
    let id = el.getAttribute(ID_ATTR);
    if (!id || !ID_RE.test(id) || seen.has(id)) {  // fresh node, or a cloned/duplicated id
      id = o.prefix + (next++);
      el.setAttribute(ID_ATTR, id);
    }
    seen.add(id);
    return id;
  }

  function emit(el, role, name) {
    out.push({
      eid: assignId(el),
      role,
      name,
      value: valueOf(el, role),
      checked: checkedOf(el, role),
      expanded: expandedOf(el),
      disabled: disabledOf(el),
      focused: !!(o.reportFocus && el === document.activeElement),
      options: optionsOf(el, role),
    });
  }

  const hasInteractiveDescendant = (el) => !!el.querySelector(INTERACTIVE_SELECTOR);

  function visit(el) {
    if (out.length >= o.maxElements) return;
    if (SKIP_TAGS.has(el.tagName) || el.hasAttribute(IGNORE_ATTR) || el.getAttribute('aria-hidden') === 'true') return;
    if (getComputedStyle(el).display === 'none') return;
    // A <label> bound to a control names that control; it is not reported separately.
    if (el.tagName === 'LABEL' && el.control) { childrenOf(el).forEach(visit); return; }
    const role = roleOf(el);
    if (role && INTERACTIVE.has(role)) {
      if (visible(el)) emit(el, role, accName(el, role));
      return;  // the control's content is its name
    }
    if (role && INFORMATIVE.has(role)) {
      const inter = hasInteractiveDescendant(el);
      if (visible(el)) {
        const al = norm(el.getAttribute('aria-label')) || norm(textOfIds(el.getAttribute('aria-labelledby') || ''));
        emit(el, role, al ? clip(al) : clip(inter ? directText(el) : (el.innerText || el.textContent)));
      }
      if (inter) childrenOf(el).forEach(visit);
      return;
    }
    // Generic element: a block that directly contains text becomes a "text" element.
    const dt = directText(el);
    if (dt) {
      const inter = hasInteractiveDescendant(el);
      if (visible(el)) emit(el, 'text', clip(inter ? dt : (el.innerText || el.textContent)));
      if (!inter) return;  // whole subtree consumed
    }
    childrenOf(el).forEach(visit);
  }

  if (document.body) visit(document.body);

  const se = document.scrollingElement || document.documentElement;
  const above = se.scrollTop > 1;
  const below = se.scrollTop + window.innerHeight < se.scrollHeight - 1;
  const scroll = [above ? 'more content above' : null, below ? 'more content below' : null].filter(Boolean).join(', ');
  return {elements: out, next, title: document.title || '', url: location.href, scroll: scroll || null};
}
