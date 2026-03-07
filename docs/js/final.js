function renderFilePreview(){
  filePreviewContainer.innerHTML = "";

  function shortName(name){
    const clean = name.replace(/\.[^/.]+$/, ""); // remove extension
    return clean.slice(0,25); // max 25 chars
  }
  uploadedFiles.forEach((file, index) => {
    const sizeKB = (file.size / 1024).toFixed(1);

    const pill = document.createElement("div");
    pill.className = "file-pill";

    if(file.type === 'image'){
      pill.innerHTML = `
      <img src="${file.url}" class="image-thumb" alt="${file.name}" style="margin-right:8px">
      <div style="display:flex;flex-direction:column;gap:4px">
        <div style="font-weight:700">${shortName(file.name)}</div>
        <div style="font-size:12px;color:var(--muted)">
          ${sizeKB} KB • ${file.tokens} tokens
      </div>
    </div>
    <div style="margin-left:auto">
      <button data-index="${index}">✕</button>
    </div>
    `;
    } else {
      pill.innerHTML = `
        📄 ${shortName(file.name)} (${sizeKB} KB • ${file.tokens} tokens)
        <button data-index="${index}">✕</button>
      `;
    }

    filePreviewContainer.appendChild(pill);
  });

  document.querySelectorAll(".file-pill button").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const i = Number(btn.getAttribute("data-index"));
      uploadedFiles.splice(i,1);
      renderFilePreview();
    });
  });

  updateTokenDisplay();
}

/* ---------- Letters / Tips / Share functions (wired to side buttons) ---------- */
function openLettersModal(){
  lettersModal.style.display = 'block';
  showOverlay();
  loadLetters();
}
async function loadLetters(){
  try{
    const res = await fetch(`${API_BASE}/letters`);
    if(!res.ok) { modalLettersContent.textContent = 'No letters'; return; }
    const data = await res.json();
    const name = (data.letters && data.letters.length) ? data.letters[0] : null;
    if(!name) { modalLettersContent.textContent = 'No letters found'; return; }
    const r2 = await fetch(`${API_BASE}/letters/` + encodeURIComponent(name));
    const text = await r2.text();
    modalLettersContent.textContent = text;
  }catch(e){ modalLettersContent.textContent = 'Unable to load letters'; console.error(e); }
}
closeLettersBtn && closeLettersBtn.addEventListener('click', ()=> { lettersModal.style.display='none'; hideOverlay(); });

function openTipModal(){
  tipModal.style.display = 'block';
  showOverlay();
}
closeTipBtn && closeTipBtn.addEventListener('click', ()=> { tipModal.style.display='none'; hideOverlay(); });

copyUpiBtn && (copyUpiBtn.onclick = async ()=> {
  try{
    await navigator.clipboard.writeText('9358588509@fam');
    alert('UPI copied: 9358588509@fam');
  }catch(e){
    alert('Copy failed. UPI: 9358588509@fam');
  }
});

function shareChat(){
  const shareData = { title: 'ZultX AI', text: 'ZultX — Fast, Adaptive Reasoning AI. Try it now.', url: 'https://zultx.github.io/ZultX-v2/' };
  if(navigator.share){
    navigator.share(shareData).catch(()=>{});
  } else {
    navigator.clipboard.writeText(shareData.url).then(()=> alert('Link copied!')).catch(()=> prompt('Copy & share this link:', shareData.url));
  }
}

/* wire side buttons safely */
sideLetters && sideLetters.addEventListener('click', openLettersModal);
sideTip && sideTip.addEventListener('click', openTipModal);
sideShare && sideShare.addEventListener('click', shareChat);

/* ---------- History storage helpers ---------- */
function makeConvoId(prefix='convo'){ return `${prefix}_${Date.now()}`; }
function getAllHistory(){
  try{ return JSON.parse(localStorage.getItem('zultx_history') || '{}'); } catch(e){ return {}; }
}
function setAllHistory(obj){ localStorage.setItem('zultx_history', JSON.stringify(obj)); }

const dropOverlay = document.getElementById("dropOverlay");

window.addEventListener("dragover", e=>{
  e.preventDefault();
  dropOverlay.classList.add("active");
});

window.addEventListener("dragleave", e=>{
  if(e.clientX === 0 && e.clientY === 0){
    dropOverlay.classList.remove("active");
  }
});

window.addEventListener("drop", async e=>{
  e.preventDefault();
  dropOverlay.classList.remove("active");

  const files = Array.from(e.dataTransfer.files);
  fileInput.files = e.dataTransfer.files;
  fileInput.dispatchEvent(new Event("change"));
});

function refreshHistoryList(){
  const all = getAllHistory();
  const keys = Object.keys(all).sort((a,b)=> all[b].created - all[a].created);
  historyList.innerHTML = '';

  for(const k of keys){
    const item = all[k];

    // 🔥 GOD-TIER TITLE LOGIC
    const firstUserMsg =
      item.messages?.find(m => m.role === 'user')?.text || '';

    const title =
      firstUserMsg.length > 36
        ? firstUserMsg.slice(0,36) + '…'
        : firstUserMsg || 'New Chat';

    // Optional: preview line (last message)
    const last =
      item.messages?.[item.messages.length - 1] || null;

    const preview =
      last?.text?.length > 48
        ? last.text.slice(0,48) + '…'
        : last?.text || 'New chat';

    const el = document.createElement('div');
    el.className = 'history-item';

    el.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:6px">
        <div style="font-weight:700">${title}</div>
        <div class="meta">${preview}</div>
      </div>
      <div class="small-muted">
        ${new Date(item.created).toLocaleString()}
      </div>
    `;

    el.onclick = ()=> openConversation(k);

    el.oncontextmenu = (e)=> {
      e.preventDefault();
      confirmAndDelete(k);
    };

    let pressTimer;
    el.addEventListener('touchstart', ()=>{
      pressTimer = setTimeout(()=> confirmAndDelete(k), 650);
    });
    el.addEventListener('touchend', ()=>{
      clearTimeout(pressTimer);
    });
    historyList.appendChild(el);
  }
}


/* ---------- messaging UI helpers ---------- */
function addBubble(kind, text, stream = true){
  const d = document.createElement('div');
  d.className = 'msg ' + (kind === 'user' ? 'user' : 'ai');
  if(kind === 'ai'){
    if(stream) fakeStream(d, text, 2);
    else { d.innerHTML = marked.parse(text); d.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b)); enhanceCodeBlocks(d); }
  } else {
    d.textContent = text;
  }
  chatArea.appendChild(d);
  chatArea.scrollTop = chatArea.scrollHeight;
}
function addUserMsg(text){
  addBubble('user', text);
  saveMessageToHistory(currentConversationId, 'user', text);
  updateTempVisibility();
}
function addAiMsg(text){
  addBubble('ai', text);
  saveMessageToHistory(currentConversationId, 'ai', text);
  updateTempVisibility();
}
function addSystemMsg(text){
  const d = document.createElement('div');
  d.style.textAlign='center'; d.style.color='var(--muted)'; d.textContent = text;
  chatArea.appendChild(d); chatArea.scrollTop = chatArea.scrollHeight;
}

function fakeStream(element, fullText, speed = 10) {
  element.innerHTML = "";
  let i = 0;
  function type() {
    if (i < fullText.length) {
      element.textContent += fullText[i++];
      chatArea.scrollTop = chatArea.scrollHeight;
      setTimeout(type, speed);
    } else {
      element.innerHTML = marked.parse(fullText);
      element.querySelectorAll("pre code").forEach(block => hljs.highlightElement(block));
      enhanceCodeBlocks(element);
    }
  }
  type();
}
function enhanceCodeBlocks(container){
  container.querySelectorAll('pre code').forEach(block => {

    const pre = block.parentElement;
    if(pre.classList.contains('code-enhanced')) return;

    pre.classList.add('code-enhanced');

    const className = block.className || '';
    const match = className.match(/language-(\w+)/);
    const lang = match ? match[1].toLowerCase() : 'code';
   
    pre.setAttribute('data-lang', lang.toUpperCase());

    /* COPY BUTTON */
    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Copy';
    copyBtn.className = 'copy-btn';

    copyBtn.onclick = () => {
      navigator.clipboard.writeText(block.innerText);
      copyBtn.textContent = 'Copied';
      setTimeout(()=>copyBtn.textContent='Copy',1500);
    };

    pre.appendChild(copyBtn);

    /* RUN BUTTON (only for JS / HTML) */

    if(lang === "javascript" || lang === "js" || lang === "html"){

      const runBtn = document.createElement("button");
      runBtn.textContent = "Run";
      runBtn.className = "run-btn";
    
      const iframe = document.createElement("iframe");
      iframe.className = "code-output";
      iframe.style.display = "none";

      runBtn.onclick = () => {
       if (iframe.style.display === "block") {
        // RESET
        iframe.srcdoc = "";
        iframe.style.display = "none";
        runBtn.textContent = "Run";
       } 
       else {
         // RUN
         iframe.style.display = "block";
         runBtn.textContent = "Reset";
         runCodeSandbox(block.innerText, lang, iframe);
      }

      };
     
      pre.appendChild(runBtn);
      pre.appendChild(iframe);
      
    }

  });
}

/* ---------- history save/load ---------- */
function saveMessageToHistory(convoId, role, text){
  if(tempMode) return;
  const all = getAllHistory();
  all[convoId] = all[convoId] || { id: convoId, created: Date.now(), messages: [] };
  all[convoId].messages.push({ role, text, t: Date.now() });
  setAllHistory(all);
  refreshHistoryList();
}
function loadConversation(convoId){
  const all = getAllHistory();
  return all[convoId] || null;
}
function localSaveInit(convoId){
  localStorage.setItem("zultx_last_convo", convoId); // ADD THIS
  const all = getAllHistory();
  if(!all[convoId]) all[convoId] = { id: convoId, created: Date.now(), messages: [] };
  setAllHistory(all);
}

function showUploadSpinner(){
  document.getElementById('uploadSpinner').style.display = 'block';
}

function hideUploadSpinner(){
  document.getElementById('uploadSpinner').style.display = 'none';
}

/* ---------- Streaming sendMessage() ---------- */
async function sendMessage(){
  const text = input.value.trim();
  if(!text) return;
  centerScreen.style.display = 'none'; chatArea.style.display = 'flex';
  if(!currentConversationId){
    currentConversationId = makeConvoId();
    localSaveInit(currentConversationId);
  }
  addUserMsg(text);
  input.value = '';
  sendBtn.disabled = true;

  let aiDiv = document.createElement('div'); aiDiv.className = 'msg ai thinking'; aiDiv.innerHTML = `<div class="ai-pulse"></div>`;
  chatArea.appendChild(aiDiv);
  try{
    const token = getToken();
    const sessionId = ensureSessionId();
    const headers = { "Accept":"application/json", "X-Session-Id": sessionId };
    if(token) headers["Authorization"] = "Bearer " + token;
    let finalMessage = text;
    if (uploadedFiles.length > 0) {
      const combinedFiles = uploadedFiles.map(f =>
        `File: ${f.name}\n${f.text}`
      ).join("\n\n");
      finalMessage = `
      User message:
      ${text}
      Attached files:
      ${combinedFiles}
      `;
}
    const res = await fetch(`${API_BASE}/ask?q=${encodeURIComponent(finalMessage)}`, { method:"GET", headers });
    if(!res.ok) throw new Error('Server error');
    const data = await res.json();
    const answer = data?.answer || "No response.";
    aiDiv.classList.remove("thinking");
    fakeStream(aiDiv, answer, 0);
    saveMessageToHistory(currentConversationId, 'ai', answer);
  }catch(err){
    console.error(err);
    aiDiv.textContent = "⚠️ Connection error.";
  }finally{ renderFilePreview(); updateTokenDisplay(); sendBtn.disabled = false; updateTempVisibility(); }
}

/* ---------- Open / New conversation ---------- */
startBtn && startBtn.addEventListener('click', ()=> {
  tempMode = false;
  currentConversationId = makeConvoId();
  localSaveInit(currentConversationId);
  chatArea.innerHTML = '';
  centerScreen.style.display = 'none';
  chatArea.style.display = 'flex';
  addSystemMsg('New chat started.');
  updateTempVisibility();
  refreshHistoryList();
});
newChatBtn && newChatBtn.addEventListener('click', ()=> {
  tempMode = false;
  currentConversationId = makeConvoId();
  localSaveInit(currentConversationId);
  chatArea.innerHTML = '';
  centerScreen.style.display = 'none';
  chatArea.style.display = 'flex';
  addSystemMsg('New chat started.');
  updateTempVisibility();
  refreshHistoryList();
});
function openConversation(convoId){
  tempMode = false;
  localStorage.setItem("zultx_last_convo", convoId);
  currentConversationId = convoId;
  const convo = loadConversation(convoId);
  chatArea.innerHTML = '';
  centerScreen.style.display = 'none';
  chatArea.style.display = 'flex';
  if(convo && convo.messages) for(const m of convo.messages) addBubble(m.role === 'user' ? 'user' : 'ai', m.text, false);
  updateTempVisibility();
}

function confirmAndDelete(convoId){
  if(!confirm("Delete this chat forever?")) return;
  const all = getAllHistory();
  if(all[convoId]) delete all[convoId];
  setAllHistory(all);
  refreshHistoryList();
  if(currentConversationId === convoId){
    currentConversationId = null;
    chatArea.innerHTML = '';
    centerScreen.style.display = 'flex';
    chatArea.style.display = 'none';
  }
}

/* ---------- Bind send & enter behavior ---------- */
sendBtn && sendBtn.addEventListener('click', sendMessage);
input && input.addEventListener('keydown', function(e) {
  if(e.key === "Enter"){
    if(e.shiftKey) return;
    if(isMobile) return;
    e.preventDefault();
    sendMessage();
  }
});
input && input.addEventListener("input", ()=> { input.style.height = "auto"; const maxHeight = 100; input.style.height = Math.min(input.scrollHeight, maxHeight) + "px"; });
/* ---------- Init conversation correctly ---------- */

const savedConvo = localStorage.getItem("zultx_last_convo");

if(savedConvo){
  currentConversationId = savedConvo;
  openConversation(savedConvo);
}else{
  currentConversationId = makeConvoId();
  localSaveInit(currentConversationId);
}

refreshHistoryList();
centerScreen.style.display = 'flex';
chatArea.style.display = 'none';
updateTempVisibility();
  

/* ---------- Premium UX improvements ---------- */
function openPremium(mode='login'){
  premiumMode = mode;
  p_email.style.display = mode === 'signup' ? 'block' : 'none';
  p_submit.textContent = mode === 'signup' ? 'Create account' : 'Login';
  p_toggle.textContent = mode === 'signup' ? 'Switch to Login' : 'Switch to Signup';
  p_msg.textContent = '';
  p_username.value = ''; p_password.value = '';
  premiumAuth.style.display = 'flex'; premiumAuth.classList.add('active');
}
function closePremium(){
  premiumAuth.classList.remove('active'); premiumAuth.style.display = 'none';
  hideOverlay();
}
premiumClose && premiumClose.addEventListener('click', closePremium);
p_toggle && p_toggle.addEventListener('click', ()=> openPremium(p_toggle.textContent.includes('Signup') ? 'signup' : 'login'));
p_guest && p_guest.addEventListener('click', ()=> { ensureSessionId(); closePremium(); p_msg.textContent = 'Guest session active'; setTimeout(()=> p_msg.textContent = '',1500); });
p_eye && p_eye.addEventListener('click', ()=> { if(p_password.type === 'password'){ p_password.type='text'; p_eye.textContent='⊘'; } else { p_password.type='password'; p_eye.textContent='👁️'; } });

async function checkUsernameAvailable(un){
  try{
    const r = await fetch(`${API_BASE}/check_username?username=${encodeURIComponent(un)}`);
    if(!r.ok) return false;
    const d = await r.json();
    return !!d.available;
  }catch(e){ return false; }
}

p_submit && p_submit.addEventListener('click', async ()=>{
  const u = p_username.value.trim(); const pass = p_password.value; const em = p_email.value.trim();
  if(!u || !pass){ p_msg.textContent = 'username & password required'; return; }
  if(premiumMode === 'signup'){ p_msg.textContent = 'Checking availability...'; const ok = await checkUsernameAvailable(u); if(!ok){ p_msg.textContent = 'Username not available'; return; } }
  p_msg.textContent = 'Contacting server...';
  const url = premiumMode === 'signup' ? `${API_BASE}/signup` : `${API_BASE}/login`;
  const body = premiumMode === 'signup' ? { username:u, password:pass, email:em } : { username:u, password:pass };
  try{
    const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await res.json();
    if(!res.ok || !data.ok){ p_msg.textContent = data.error || 'Auth failed'; return; }
    setToken(data.token);
    localStorage.setItem('zultx_username', data.user.username);
    p_msg.textContent = 'Welcome!';
    setTimeout(()=> { p_msg.textContent=''; closePremium(); refreshAccountUI(); }, 900);
  }catch(err){ console.error(err); p_msg.textContent = 'Network error'; }
});

/* popup close on ESC */
document.addEventListener('keydown', (e)=> {
  if(e.key === 'Escape'){ closeAllSideModals(); closePremium(); }
});

/* utility to close all open side modals/panels */
function closeAllSideModals(){
  tempModal.style.display='none';
  lettersModal.style.display='none';
  tipModal.style.display='none';
  sidePanel.classList.remove('open');
  hideOverlay();
}

/* ---------- Auth helpers (same as before) ---------- */
function ensureSessionId(){
  let sid = localStorage.getItem('zultx_session');
  if(!sid){ sid = 'sess_' + Math.random().toString(36).slice(2) + '_' + Date.now(); localStorage.setItem('zultx_session', sid); }
  return sid;
}
function setToken(token){ if(token) localStorage.setItem('zultx_token', token); else localStorage.removeItem('zultx_token'); }
function getToken(){ return localStorage.getItem('zultx_token'); }
function loggedIn(){ return !!getToken(); }

/* account row creation */
(function addLeftAccountRow(){
  const side = document.querySelector('.side');
  if(!side) return;
  const row = document.createElement('div');
  row.className = 'account-row';
  row.innerHTML = `<div><div style="font-weight:800">Account</div><div class="account-welcome" id="accWelcome">Not signed in</div></div><div><button id="accBtn" class="account-btn login">Login</button></div>`;
  side.insertBefore(row, side.firstChild);
  const accBtn = document.getElementById('accBtn');
  accBtn && accBtn.addEventListener('click', ()=>{
    if(loggedIn()){
      if(confirm("Log out?")){ setToken(null); localStorage.removeItem('zultx_username'); refreshAccountUI(); p_msg.textContent = 'Logged out'; setTimeout(()=> p_msg.textContent='',1500); }
    } else openPremium('login');
  });
})();

function refreshAccountUI(){
  const btn = document.getElementById('accBtn');
  const welcome = document.getElementById('accWelcome');
  const sideName = document.getElementById('sideUserName');

  const username = localStorage.getItem('zultx_username');

  if(loggedIn()){
    btn.classList.remove('login');
    btn.classList.add('logout');
    btn.textContent = 'Logout';

    welcome.textContent = 'Welcome — ' + (username || 'User');
    if(sideName) sideName.textContent = username || 'User';
  } else {
    btn.classList.add('login');
    btn.classList.remove('logout');
    btn.textContent = 'Login';

    welcome.textContent = 'Not signed in';
    if(sideName) sideName.textContent = 'User';

  }
}

/* ===== PLUS PANEL UI (clean grid + click-outside close) ===== */
(function setupPlusPanel(){
  // create panel and bottom-sheet (clean grid)
  const plusPanel = document.createElement('div');
  plusPanel.id = 'plusPanel';
  plusPanel.className = 'plus-panel';
  plusPanel.innerHTML = `
    <button class="mode-btn" id="cameraBtn" title="Open camera">
      <span>📷</span>
      <div>Camera</div>
    </button>
    <button class="mode-btn" id="videoBtn" title="Video (soon)">
      <span>🎞️</span>
      <div>Video</div>
    </button>
    <div class="modes" aria-hidden="true">
      <div class="mode-pill" data-mode="concise">Concise</div>
      <div class="mode-pill" data-mode="creative">Creative</div>
      <div class="mode-pill" data-mode="technical">Technical</div>
    </div>
  `;
  // insert into document near compose tools
  const composeInner = document.querySelector('.compose-inner');
  composeInner && composeInner.appendChild(plusPanel);

  // bottom-sheet (re-usable)
  const bottomSheet = document.getElementById('plusBottomSheet') || (function createSheet(){
    const bs = document.createElement('div');
    bs.id = 'plusBottomSheet';
    bs.className = 'bottom-sheet';
    bs.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div style="font-weight:800">Uploads</div>
        <button id="closeSheet" class="menu-btn">Close</button>
      </div>
      <div id="thumbArea" class="thumb-row"></div>
      <div id="uploadStatus" class="upload-status">No recent uploads</div>
    `;
    document.body.appendChild(bs);
    return bs;
  })();

  // open/close behavior (toggle panel & bottom sheet)
  function closePlusPanel(){
    plusPanel.classList.remove('active');
    bottomSheet.classList.remove('open');
    hideOverlay();
  }

  function openPlusPanel(){
    plusPanel.classList.add('active');
    // keep bottomSheet closed unless we actually upload images
    // but show minimal overlay so user can click out
    showOverlay();
    // keep overlay click to close
  }

  plusBtn.addEventListener('click', (e)=>{
    e.stopPropagation();

    plusBtn.classList.toggle("plus-active-bg");

    if(plusPanel.classList.contains('active')) {
      closePlusPanel();
    } else {
      openPlusPanel();
    }
  });

  // close panel when clicking outside
  document.addEventListener('click', (ev)=>{
    if(!plusPanel.contains(ev.target) && ev.target !== plusBtn && !plusBtn.contains(ev.target)){
      closePlusPanel();
    }
  });

  // close on overlay click too (overlay already used by other modals)
  document.getElementById('z-overlay')?.addEventListener('click', closePlusPanel);

  // modes (visual only)
  plusPanel.querySelectorAll('.mode-pill').forEach(p=>{
    p.addEventListener('click', (ev)=>{
      plusPanel.querySelectorAll('.mode-pill').forEach(x=>x.classList.remove('active'));
      p.classList.add('active');
      // small toast
      const t = document.createElement('div');
      t.textContent = `${p.dataset.mode} mode selected`;
      t.className = 'mode-toast';
      document.body.appendChild(t);
      requestAnimationFrame(() => t.classList.add('show'));
      setTimeout(() => {
        t.classList.remove('show');
        setTimeout(() => t.remove(), 300);
      }, 1200);
    });
  });

  // wire camera / image / video buttons
  const cameraInput = document.getElementById('cameraInput');
  const videoInput = document.getElementById('videoInput');

  // helper to close panel and stop propagation
  function preActionClose(e){
    if(e && e.stopPropagation) e.stopPropagation();
    closePlusPanel();
  }

  plusPanel.querySelector('#cameraBtn').addEventListener('click', (e)=> {
    preActionClose(e);
    cameraInput.value = '';
    cameraInput.click();
  });


  plusPanel.querySelector('#videoBtn').addEventListener('click', (e)=> {
    preActionClose(e);
    alert('Video uploading coming soon — watch this space!');
  });

  // show/hide bottom sheet: close button
  bottomSheet.querySelector('#closeSheet')?.addEventListener('click', ()=> {
    bottomSheet.classList.remove('open');
    hideOverlay();
  });

  // handle camera capture / image change — reuse existing handlers but ensure sheet opens
  cameraInput.addEventListener('change', async (ev)=>{
    const file = ev.target.files && ev.target.files[0];
    if(!file) return;
    bottomSheet.classList.add('open');
    showUploadSpinner();
    // call your same handler locally (copied behavior)
    await (async function handleImageFile_local(file){
      try{
        const form = new FormData();
        form.append('image', file);
        const res = await fetch(`${API_BASE}/upload-image`, { method:'POST', body: form });
        if(res.ok){
          const data = await res.json();
          uploadedFiles.push({
            name: data.filename || file.name,
            size: file.size,
            url: data.url || '',
            type: 'image',
            tokens: estimateImageTokens(file)
          });
          document.getElementById('uploadStatus').textContent = `Uploaded ${file.name}`;
        } else {
          throw new Error('Upload failed');
        }
      } catch(err){
        const url = URL.createObjectURL(file);
        uploadedFiles.push({
          name: file.name,
          size: file.size,
          url,
          type: 'image',
          tokens: estimateImageTokens(file),
          local:true
        });
        document.getElementById('uploadStatus').textContent = `Preview added: ${file.name} (local)`;
      } finally {
        hideUploadSpinner();
        renderFilePreview();
        populateThumbs();
      }
    })(file);
  });

  // thumbnail population function for bottomSheet (keeps in-sync)
  function populateThumbs(){
    const area = document.getElementById('thumbArea');
    if(!area) return;
    area.innerHTML = '';
    uploadedFiles.filter(f=> f.type === 'image').forEach((f,i)=>{
      const img = document.createElement('img');
      img.className = 'image-thumb';
      img.src = f.url || (f.local ? f.url : '');
      img.alt = f.name;
      const wrapper = document.createElement('div');
      wrapper.style.display='flex';
      wrapper.style.flexDirection='column';
      wrapper.style.alignItems='center';
      wrapper.style.gap='6px';
      const btn = document.createElement('button');
      btn.textContent='✕';
      btn.style.fontSize='12px';
      btn.style.border='none';
      btn.style.background='transparent';
      btn.style.color='var(--muted)';
      btn.addEventListener('click', ()=> {
        uploadedFiles.splice(i,1);
        renderFilePreview();
        populateThumbs();
      });
      wrapper.appendChild(img);
      wrapper.appendChild(btn);
      area.appendChild(wrapper);
    });
  }

  // expose small helper to global scope for re-use if needed
  window.populateThumbs = populateThumbs;

})();

refreshAccountUI();
loadSuggestions();

/* ---------- small helpers: click outside chat area closes side on mobile (improve UX) ---------- */
const scrollDownBtn = document.getElementById("scrollDownBtn");

chatArea.addEventListener("scroll", () => {
  const isNearBottom =
    chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 100;

  if(!isNearBottom){
    scrollDownBtn.style.display = "flex";
  } else {
    scrollDownBtn.style.display = "none";
  }
});

function toggleScrollButton(){
  const nearBottom =
    chatArea.scrollHeight - chatArea.scrollTop - chatArea.clientHeight < 80;

  scrollDownBtn.style.display = nearBottom ? "none" : "flex";
}

chatArea.addEventListener("scroll", toggleScrollButton);

scrollDownBtn.addEventListener("click", ()=>{
  chatArea.scrollTo({
    top: chatArea.scrollHeight,
    behavior: "smooth"
  });
});


document.addEventListener('click', (e)=> {
  // if side is open and click outside sidePanel on mobile -> close
  if(sidePanel.classList.contains('open') && isMobile){
    if(!sidePanel.contains(e.target) && !mobileMenuBtn.contains(e.target)){
      sidePanel.classList.remove('open');
      hideOverlay();
    }
  }
});

/* ---------- bootstrap done ---------- */
console.log('ZultX UI loaded — tidy & robust.');
console.log('WELCOME TO ZULTX UX!');
