'use strict';
// Regression tests for frontend/index.html — run with:  node --test tests/test_frontend.cjs
// Zero dependencies: node:test + node:vm + a deliberately tiny fake DOM.
// The fake DOM never parses HTML, so any `innerHTML = "<...>"` write is recorded
// verbatim (and flagged) instead of being interpreted — which is exactly the
// stored-XSS surface we want to keep closed.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const HTML_PATH = path.join(__dirname, '..', 'frontend', 'index.html');

// ---------------------------------------------------------------- fake DOM --
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

class FakeText {
  constructor(data) { this.nodeType = 3; this.data = String(data); this.parentNode = null; }
  get textContent() { return this.data; }
  serialize() { return esc(this.data); }
}

class FakeElement {
  constructor(tag, doc) {
    this.nodeType = 1; this.tagName = tag.toUpperCase(); this._doc = doc;
    this.childNodes = []; this.attrs = {}; this.style = {}; this.listeners = {};
    this.parentNode = null; this.disabled = false; this.value = ''; this.files = [];
    this.onclick = null; this._rawHTML = undefined;
  }
  get id() { return this.attrs.id || ''; } set id(v) { this.attrs.id = String(v); }
  get className() { return this.attrs.class || ''; } set className(v) { this.attrs.class = String(v); }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  get firstChild() { return this.childNodes[0] || null; }
  get children() { return this.childNodes.filter((n) => n.nodeType === 1); }
  appendChild(n) { if (n.parentNode) n.parentNode.removeChild(n); n.parentNode = this; this.childNodes.push(n); this._rawHTML = undefined; return n; }
  append(...ns) { ns.forEach((n) => this.appendChild(typeof n === 'string' ? new FakeText(n) : n)); }
  removeChild(n) { const i = this.childNodes.indexOf(n); if (i < 0) throw new Error('NotFoundError'); this.childNodes.splice(i, 1); n.parentNode = null; return n; }
  replaceChildren(...ns) { this.childNodes.forEach((c) => { c.parentNode = null; }); this.childNodes = []; this._rawHTML = undefined; ns.forEach((n) => this.appendChild(n)); }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(''); }
  set textContent(v) { this.replaceChildren(); if (v !== null && v !== undefined && v !== '') this.appendChild(new FakeText(v)); }
  // innerHTML is *not* parsed: raw markup is kept verbatim so tests can see it.
  get innerHTML() { return this._rawHTML !== undefined ? this._rawHTML : this.childNodes.map((c) => c.serialize()).join(''); }
  set innerHTML(v) { this.childNodes = []; this._rawHTML = String(v); this._doc.innerHTMLWrites.push({ id: this.id, html: String(v) }); }
  serialize() {
    const attrs = Object.entries(this.attrs).map(([k, v]) => ` ${k}="${esc(v)}"`).join('');
    return `<${this.tagName.toLowerCase()}${attrs}>${this.innerHTML}</${this.tagName.toLowerCase()}>`;
  }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  dispatchEvent(ev) { (this.listeners[ev.type] || []).forEach((fn) => fn(ev)); }
  // Mirrors the real DOM: HTMLElement.click() on a disabled control is a no-op.
  click() { if (this.disabled) return undefined; const r = this.onclick ? this.onclick() : undefined; this.dispatchEvent({ type: 'click' }); this._lastClick = r; return r; }
}

function buildDocument(html) {
  const doc = { innerHTMLWrites: [], _byId: {} };
  doc.createElement = (tag) => new FakeElement(tag, doc);
  doc.createTextNode = (t) => new FakeText(t);
  doc.getElementById = (id) => doc._byId[id] || null;
  // Pre-create every element that declares an id in the markup (no HTML parsing needed).
  for (const m of html.matchAll(/<(\w+)\b[^>]*\sid="([^"]+)"/g)) {
    const el = new FakeElement(m[1], doc); el.id = m[2]; doc._byId[m[2]] = el;
  }
  return doc;
}

// ---------------------------------------------------------- fetch stubbing --
const STATUS_TEXT = { 200: 'OK', 400: 'Bad Request', 422: 'Unprocessable Entity', 500: 'Internal Server Error', 502: 'Bad Gateway', 503: 'Service Unavailable' };

function res(status, body, contentType) {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  const ctype = contentType || (typeof body === 'string' ? 'text/plain; charset=utf-8' : 'application/json');
  let used = false;
  const consume = () => { if (used) throw new TypeError('Failed to execute: body stream already read'); used = true; };
  return {
    ok: status >= 200 && status < 300, status, statusText: STATUS_TEXT[status] || '',
    headers: { get: (k) => (k.toLowerCase() === 'content-type' ? ctype : null) },
    get bodyUsed() { return used; },
    text: async () => { consume(); return text; },
    json: async () => { consume(); return JSON.parse(text); },
  };
}

const HEALTH = { status: 'ok', llm_provider: 'testing', embedding_provider: 'testing', documents: 1, chunks: 2 };
const EVIL_NAME = '<img src=x onerror=alert(1)>.pdf';
const EVIL_SNIPPET = 'Totals: <script>alert("pwned")</script> & 2 < 3 "quoted"';

// Drain the microtask/macrotask queue without arbitrary sleeps.
async function flush() { for (let i = 0; i < 8; i++) await new Promise((r) => setImmediate(r)); }

async function load(routes = {}) {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const m = html.match(/<script>([\s\S]*?)<\/script>/);
  assert.ok(m, 'index.html must contain an inline <script>');
  const document = buildDocument(html);
  const calls = [];
  const handlers = {
    '/api/health': () => res(200, HEALTH),
    '/api/documents': () => res(200, [{ doc_id: 'd1', source: 'ok.pdf', pages: 1, chunks: 2 }]),
    ...routes,
  };
  const fetch = (url, opts = {}) => {
    calls.push({ url, opts });
    const h = handlers[url];
    if (!h) return Promise.reject(new TypeError('Failed to fetch'));
    return Promise.resolve(h(opts, calls));
  };
  class FormData { constructor() { this.entries = []; } append(k, v) { this.entries.push([k, v]); } }
  const sandbox = { document, fetch, FormData, console, setTimeout, clearTimeout, URLSearchParams, JSON, Error, TypeError, Promise, Array, Object, String };
  sandbox.window = sandbox;
  vm.runInNewContext(m[1], sandbox, { filename: 'index.html' });
  await flush(); // initial refresh()
  const $ = (id) => document.getElementById(id);
  return { document, calls, $, flush, urls: () => calls.map((c) => c.url) };
}

// Hostile tags from payloads must never appear unescaped (structural span/div/p wrappers are fine).
const noRawTags = (s) => !/<(img|script|b|h1|html)\b/i.test(s);

// ------------------------------------------------------------------- tests --
test('refresh(): document list renders hostile filenames as text, never markup', async () => {
  const { $, document } = await load({
    '/api/documents': () => res(200, [{ doc_id: 'd1', source: EVIL_NAME, pages: 3, chunks: 7 }]),
  });
  assert.equal($('status').textContent, 'testing · 1 docs · 2 chunks');
  const docs = $('docs');
  assert.equal(docs.children.length, 1);
  assert.equal(docs.children[0].className, 'doc');
  assert.equal(docs.children[0].textContent, `${EVIL_NAME} · 3p · 7 chunks`);
  assert.ok(noRawTags(docs.innerHTML), `raw markup leaked into docs list: ${docs.innerHTML}`);
  assert.ok(docs.innerHTML.includes('&lt;img'), 'filename should be escaped in serialized output');
  assert.deepEqual(document.innerHTMLWrites, []);
});

test('refresh(): non-2xx health goes to "offline" instead of rendering undefined fields', async () => {
  const { $ } = await load({ '/api/health': () => res(503, { detail: 'warming up' }) });
  assert.equal($('status').textContent, 'offline');
});

test('refresh(): non-2xx document list does not blow up the status badge', async () => {
  const { $ } = await load({ '/api/documents': () => res(500, 'Internal Server Error') });
  assert.equal($('status').textContent, 'offline');
  assert.equal($('docs').children.length, 0);
});

test('query: hostile source + PDF snippet render as text; meta line preserved', async () => {
  const { $, document } = await load({
    '/api/query': () => res(200, {
      answer: 'Totals are on page 3 <b>bold?</b>', model: 'echo',
      citations: [
        { doc_id: 'd1', source: EVIL_NAME, page: 3, chunk_index: 0, score: 0.8123, snippet: EVIL_SNIPPET },
        { doc_id: 'd1', source: 'plain.pdf', page: null, chunk_index: 1, score: 0.5, snippet: 'second' },
      ],
    }),
  });
  $('question').value = 'what are the totals?';
  await $('askBtn').onclick();

  assert.equal($('resultCard').style.display, 'block');
  assert.equal($('answer').textContent, 'Totals are on page 3 <b>bold?</b>');
  assert.ok(noRawTags($('answer').innerHTML));

  const cites = $('citations').children;
  assert.equal(cites.length, 2);
  assert.equal(cites[0].className, 'cite');
  const [meta, snip] = cites[0].children;
  assert.equal(meta.className, 'meta');
  assert.equal(snip.className, 'snip');
  assert.equal(meta.textContent, `[1] ${EVIL_NAME} · p.3 · score 0.8123`);
  assert.equal(snip.textContent, EVIL_SNIPPET);
  assert.equal(cites[1].children[0].textContent, '[2] plain.pdf · score 0.5');
  assert.ok(noRawTags($('citations').innerHTML), `raw markup leaked into citations: ${$('citations').innerHTML}`);
  assert.ok($('citations').innerHTML.includes('&lt;script&gt;'));
  assert.deepEqual(document.innerHTMLWrites, []);
  assert.equal($('askBtn').disabled, false);
});

test('query: empty citations shows the muted "No sources." note', async () => {
  const { $ } = await load({ '/api/query': () => res(200, { answer: 'n/a', model: 'echo', citations: [] }) });
  $('question').value = 'anything';
  await $('askBtn').onclick();
  assert.equal($('citations').children.length, 1);
  assert.equal($('citations').children[0].className, 'muted');
  assert.equal($('citations').children[0].textContent, 'No sources.');
});

test('query: JSON {detail: string} error surfaces the detail', async () => {
  const { $ } = await load({ '/api/query': () => res(400, { detail: 'No documents indexed yet.' }) });
  $('question').value = 'q';
  await $('askBtn').onclick();
  assert.match($('answer').textContent, /^Error: /);
  assert.ok($('answer').textContent.includes('No documents indexed yet.'));
  assert.equal($('askBtn').disabled, false);
});

test('query: plain-text 500 yields a useful status, not a JSON parse exception', async () => {
  const { $ } = await load({ '/api/query': () => res(500, 'Internal Server Error') });
  $('question').value = 'q';
  await $('askBtn').onclick();
  const msg = $('answer').textContent;
  assert.match(msg, /^Error: /);
  assert.ok(msg.includes('500'), `expected HTTP status in message, got: ${msg}`);
  assert.ok(!/Unexpected token|JSON/i.test(msg), `leaked parser error: ${msg}`);
  assert.equal($('citations').children.length, 0);
  assert.equal($('askBtn').disabled, false);
});

test('query: HTML 502 from a gateway is summarised, never rendered as markup', async () => {
  const body = '<html><head><title>502 Bad Gateway</title></head><body><h1>502 Bad Gateway</h1><hr>nginx</body></html>';
  const { $, document } = await load({ '/api/query': () => res(502, body, 'text/html') });
  $('question').value = 'q';
  await $('askBtn').onclick();
  const msg = $('answer').textContent;
  assert.match(msg, /^Error: /);
  assert.ok(msg.includes('502'), `expected 502 in message, got: ${msg}`);
  assert.ok(!msg.includes('<h1>') && !msg.includes('<html>'), `HTML body dumped into UI: ${msg}`);
  assert.ok(!/Unexpected token|JSON/i.test(msg), `leaked parser error: ${msg}`);
  assert.deepEqual(document.innerHTMLWrites, []);
});

test('query: failure after a successful answer removes stale citations', async () => {
  let fail = false;
  const { $ } = await load({
    '/api/query': () => (fail
      ? res(500, 'boom')
      : res(200, { answer: 'ok', model: 'echo', citations: [{ doc_id: 'd1', source: 'a.pdf', page: 1, chunk_index: 0, score: 0.9, snippet: 's' }] })),
  });
  $('question').value = 'q';
  await $('askBtn').onclick();
  assert.equal($('citations').children.length, 1);
  fail = true;
  await $('askBtn').onclick();
  assert.match($('answer').textContent, /^Error: /);
  assert.equal($('citations').children.length, 0, 'stale citations must be cleared on failure');
});

test('query: malformed 200 body (not an answer) is reported, not rendered as "undefined"', async () => {
  const { $ } = await load({ '/api/query': () => res(200, { detail: 'weird' }) });
  $('question').value = 'q';
  await $('askBtn').onclick();
  assert.match($('answer').textContent, /^Error: /);
  assert.ok(!$('answer').textContent.includes('undefined'));
});

test('query: Enter while a request is in flight does not dispatch a second request', async () => {
  let release;
  const pending = new Promise((r) => { release = r; });
  const { $, calls, flush } = await load({
    '/api/query': () => pending.then(() => res(200, { answer: 'ok', model: 'echo', citations: [] })),
  });
  $('question').value = 'q';
  const first = $('askBtn').onclick();          // in flight
  await flush();
  assert.equal($('askBtn').disabled, true);
  $('question').dispatchEvent({ type: 'keydown', key: 'Enter', preventDefault() {} });
  $('askBtn').click();                          // disabled → no-op
  const second = $('askBtn').onclick();         // handler invoked directly must also be guarded
  await flush();
  assert.equal(calls.filter((c) => c.url === '/api/query').length, 1, 'overlapping query dispatched');
  release(); await first; await second; await flush();
  assert.equal($('askBtn').disabled, false);
  assert.equal($('answer').textContent, 'ok');
  // Enter now works again (one more request)
  $('question').dispatchEvent({ type: 'keydown', key: 'Enter', preventDefault() {} });
  await flush();
  assert.equal(calls.filter((c) => c.url === '/api/query').length, 2);
});

test('query: blank question is ignored and leaves the button enabled', async () => {
  const { $, calls } = await load();
  $('question').value = '   ';
  await $('askBtn').onclick();
  assert.equal(calls.filter((c) => c.url === '/api/query').length, 0);
  assert.equal($('askBtn').disabled, false);
});

test('upload: success reports count and refreshes the document list', async () => {
  const { $, urls } = await load({
    '/api/upload': () => res(200, { documents: [{ doc_id: 'd2', source: 'new.pdf', pages: 2, chunks: 5 }] }),
  });
  const before = urls().filter((u) => u === '/api/documents').length;
  $('files').files = [{ name: 'new.pdf' }];
  await $('uploadBtn').onclick();
  await flush();
  assert.equal($('uploadMsg').textContent, 'Indexed 1 document(s).');
  assert.ok(urls().filter((u) => u === '/api/documents').length > before, 'document list not refreshed');
  assert.equal($('uploadBtn').disabled, false);
});

test('upload: FastAPI 422 validation array is flattened, not "[object Object]"', async () => {
  const { $ } = await load({
    '/api/upload': () => res(422, { detail: [{ loc: ['body', 'files'], msg: 'field required', type: 'value_error.missing' }] }),
  });
  $('files').files = [{ name: 'x.pdf' }];
  await $('uploadBtn').onclick();
  const msg = $('uploadMsg').textContent;
  assert.match(msg, /^Error: /);
  assert.ok(msg.includes('field required'), `validation message lost: ${msg}`);
  assert.ok(!msg.includes('[object Object]') && !msg.includes('undefined'), `garbage message: ${msg}`);
  assert.equal($('uploadBtn').disabled, false);
});

test('upload: 422 {detail: string} from the parser path is shown verbatim', async () => {
  const { $ } = await load({ '/api/upload': () => res(422, { detail: 'No extractable text in scan.pdf.' }) });
  $('files').files = [{ name: 'scan.pdf' }];
  await $('uploadBtn').onclick();
  assert.ok($('uploadMsg').textContent.includes('No extractable text in scan.pdf.'));
});

test('upload: plain-text 500 and HTML 502 give status-bearing messages and re-enable the button', async () => {
  for (const [status, body, ctype] of [[500, 'Internal Server Error', undefined], [502, '<html><body><h1>502</h1></body></html>', 'text/html']]) {
    const { $, document } = await load({ '/api/upload': () => res(status, body, ctype) });
    $('files').files = [{ name: 'x.pdf' }];
    await $('uploadBtn').onclick();
    const msg = $('uploadMsg').textContent;
    assert.match(msg, /^Error: /);
    assert.ok(msg.includes(String(status)), `expected ${status} in: ${msg}`);
    assert.ok(!msg.includes('<h1>'), `HTML dumped: ${msg}`);
    assert.ok(!/Unexpected token|JSON/i.test(msg), `leaked parser error: ${msg}`);
    assert.equal($('uploadBtn').disabled, false);
    assert.deepEqual(document.innerHTMLWrites, []);
  }
});

test('upload: network failure is reported and button re-enabled', async () => {
  const { $ } = await load({ '/api/upload': () => { throw new TypeError('Failed to fetch'); } });
  $('files').files = [{ name: 'x.pdf' }];
  await $('uploadBtn').onclick();
  assert.equal($('uploadMsg').textContent, 'Error: Failed to fetch');
  assert.equal($('uploadBtn').disabled, false);
});

test('upload: no files selected → prompt, no request', async () => {
  const { $, calls } = await load();
  $('files').files = [];
  await $('uploadBtn').onclick();
  assert.equal($('uploadMsg').textContent, 'Pick at least one PDF.');
  assert.equal(calls.filter((c) => c.url === '/api/upload').length, 0);
});

test('script never writes innerHTML anywhere (static check)', () => {
  const html = fs.readFileSync(HTML_PATH, 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  assert.ok(!/\.innerHTML\s*=/.test(script), 'innerHTML assignment found in inline script');
  assert.ok(!/insertAdjacentHTML|outerHTML\s*=|document\.write/.test(script));
});
