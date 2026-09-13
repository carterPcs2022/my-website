"""Small first-party browser console for the Zane/P.I.X.A.L. API.

The console deliberately lives beside the API so there is no separate frontend
repository to keep synchronized. It consumes the structured ``pixal_reply``
field returned by ``POST /chat`` and keeps Zane and P.I.X.A.L. visually distinct.
"""

WEB_APP_HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#07111f">
  <title>Zane &amp; P.I.X.A.L.</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #050a12;
      --panel: rgba(12, 22, 37, .82);
      --panel-2: rgba(17, 30, 49, .72);
      --line: rgba(160, 205, 255, .14);
      --text: #eef6ff;
      --muted: #91a5bd;
      --zane: #73b9ff;
      --pixal: #7ff0e2;
      --danger: #ff7f9d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", system-ui, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at 20% 0%, rgba(76, 153, 255, .16), transparent 34%),
        radial-gradient(circle at 90% 10%, rgba(83, 232, 210, .12), transparent 30%),
        var(--bg);
    }
    .shell { width: min(980px, 100%); margin: 0 auto; padding: 24px 16px 32px; }
    .topbar { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .orb { width: 44px; height: 44px; border-radius: 50%; background: radial-gradient(circle, #dff7ff 0 7%, #72c2ff 20%, #285d96 48%, transparent 70%); box-shadow: 0 0 28px rgba(115, 185, 255, .34); }
    h1 { margin: 0; font-size: 20px; letter-spacing: -.02em; }
    .status { display: flex; align-items: center; gap: 7px; color: var(--muted); font-size: 12px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--pixal); box-shadow: 0 0 10px rgba(127, 240, 226, .65); }
    .dot.bad { background: var(--danger); box-shadow: none; }
    .messages { min-height: 58vh; display: flex; flex-direction: column; gap: 12px; }
    .empty { margin: auto; text-align: center; color: var(--muted); max-width: 520px; padding: 36px 12px; }
    .empty strong { color: var(--text); }
    .msg { border: 1px solid var(--line); border-radius: 20px; padding: 15px 17px; background: var(--panel); backdrop-filter: blur(18px); box-shadow: 0 10px 35px rgba(0,0,0,.16); }
    .msg.zane { border-left: 3px solid var(--zane); }
    .msg.pixal { border-left: 3px solid var(--pixal); background: var(--panel-2); }
    .speaker { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 7px; }
    .speaker .mark { width: 7px; height: 7px; border-radius: 50%; background: var(--zane); box-shadow: 0 0 10px rgba(115,185,255,.5); }
    .pixal .speaker .mark { background: var(--pixal); box-shadow: 0 0 10px rgba(127,240,226,.5); }
    .text { white-space: pre-wrap; line-height: 1.55; font-size: 15px; }
    .composer { position: sticky; bottom: 14px; display: flex; gap: 10px; margin-top: 18px; padding: 10px; border: 1px solid var(--line); border-radius: 18px; background: rgba(7, 14, 24, .88); backdrop-filter: blur(20px); }
    textarea { flex: 1; min-height: 48px; max-height: 150px; resize: vertical; border: 0; outline: 0; background: transparent; color: var(--text); padding: 12px; font: inherit; }
    button { border: 0; border-radius: 13px; padding: 0 18px; font-weight: 700; color: #041019; background: #bdefff; cursor: pointer; }
    button:disabled { opacity: .45; cursor: default; }
    .meta { color: var(--muted); font-size: 11px; margin-top: 8px; }
    @media (max-width: 620px) { .shell { padding: 16px 10px 24px; } .topbar { align-items: flex-start; } .composer { bottom: 8px; } button { padding: 0 14px; } }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div class="brand"><div class="orb" aria-hidden="true"></div><div><h1>Zane &amp; P.I.X.A.L.</h1><div class="status"><span id="statusDot" class="dot"></span><span id="statusText">Checking systems…</span></div></div></div>
    </header>

    <section id="messages" class="messages" aria-live="polite">
      <div class="empty"><strong>Dual companion console</strong><br>Messages from Zane and P.I.X.A.L. will appear as separate identities.</div>
    </section>

    <form id="composer" class="composer">
      <textarea id="input" placeholder="Talk to Zane…" aria-label="Message"></textarea>
      <button id="send" type="submit">Send</button>
    </form>
  </main>

  <script>
    const sessionId = localStorage.getItem('zane_session_id') || crypto.randomUUID();
    localStorage.setItem('zane_session_id', sessionId);
    const messages = document.getElementById('messages');
    const input = document.getElementById('input');
    const send = document.getElementById('send');
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    function addMessage(kind, text, meta = '') {
      if (!text) return;
      const empty = messages.querySelector('.empty');
      if (empty) empty.remove();
      const card = document.createElement('article');
      card.className = `msg ${kind}`;
      const name = kind === 'pixal' ? 'P.I.X.A.L.' : 'Zane';
      card.innerHTML = `<div class="speaker"><span class="mark"></span>${name}</div><div class="text"></div>${meta ? `<div class="meta"></div>` : ''}`;
      card.querySelector('.text').textContent = text;
      if (meta) card.querySelector('.meta').textContent = meta;
      messages.appendChild(card);
      card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async function checkReady() {
      try {
        const response = await fetch('/ready', { cache: 'no-store' });
        const data = await response.json();
        const ok = response.ok && data.status === 'ready' && data.pixal === 'ok';
        statusDot.classList.toggle('bad', !ok);
        statusText.textContent = ok ? 'Zane + P.I.X.A.L. online' : 'Companion systems unavailable';
      } catch (_) {
        statusDot.classList.add('bad');
        statusText.textContent = 'Connection unavailable';
      }
    }

    async function sendMessage(message) {
      send.disabled = true;
      input.disabled = true;
      try {
        const response = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message, session_id: sessionId })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
        addMessage('zane', data.reply, `${Math.round(data.elapsed_ms)} ms`);
        addMessage('pixal', data.pixal_reply || 'P.I.X.A.L. had no additional recommendation for this turn.');
        if (data.audio_base64 && data.audio_format) {
          const audio = new Audio(`data:audio/${data.audio_format.split('_')[0]};base64,${data.audio_base64}`);
          audio.play().catch(() => {});
        }
      } catch (error) {
        addMessage('pixal', `I could not complete the companion cycle: ${error.message}`);
        statusDot.classList.add('bad');
        statusText.textContent = 'Request failed';
      } finally {
        send.disabled = false;
        input.disabled = false;
        input.focus();
      }
    }

    document.getElementById('composer').addEventListener('submit', event => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      input.value = '';
      addMessage('zane', message, 'You');
      sendMessage(message);
    });

    input.addEventListener('keydown', event => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        document.getElementById('composer').requestSubmit();
      }
    });

    checkReady();
  </script>
</body>
</html>'''
