fileInput.setAttribute("multiple", "true");
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files);
  if (!files.length) return;

  for (const file of files) {

    // ===== TXT FILES =====
    if (file.name.toLowerCase().endsWith(".txt")) {

      showUploadSpinner();

      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/upload-txt`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      hideUploadSpinner();

      if (!data.ok) {
        alert(data.error);
        continue;
      }

      uploadedFiles.push({
        name: data.filename,
        size: file.size,
        text: data.text,
        type: "text",
        tokens: estimateTokens(data.text)
      });
    }

    // ===== IMAGES =====
    else if (file.type.startsWith("image/")) {

      const url = URL.createObjectURL(file);

      uploadedFiles.push({
        name: file.name,
        size: file.size,
        url: url,
        type: "image",
        tokens: estimateImageTokens(file),
        local: true
      });
    }

    else {
      alert("Only .txt files and images are allowed 💙");
    }
  }

  renderFilePreview();
  fileInput.value = "";
});

function estimateTokens(text){
  return Math.ceil(text.length / 4);
}


const filePreviewContainer = document.createElement("div");
filePreviewContainer.className = "file-preview-container";
document.querySelector(".compose").prepend(filePreviewContainer);

attachBtn.addEventListener("click", () => {
  fileInput.click();
});


/* ---------- Safe helpers ---------- */
function $id(id){ return document.getElementById(id); }
function showOverlay(){ overlay.style.display = 'block'; overlay.setAttribute('aria-hidden','false'); }
function hideOverlay(){ overlay.style.display = 'none'; overlay.setAttribute('aria-hidden','true'); }

/* Close everything on overlay click (mobile UX) */
overlay.addEventListener('click', ()=> {
  closeAllSideModals();
});
function showUploadSpinner(){
  document.getElementById("uploadSpinner").style.display="block";
}

function hideUploadSpinner(){
  document.getElementById("uploadSpinner").style.display="none";
}
/* ---------- Theme logic (single source of truth) ---------- */
function applyInitialTheme(){
  const saved = localStorage.getItem('zultx_theme');
  if(saved === 'light') document.documentElement.className = 'light';
  else if(saved === 'dark') document.documentElement.className = 'dark';
  else {
    const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
    document.documentElement.className = prefersLight ? 'light' : 'dark';
  }
  updateThemeVisuals();
}
function toggleTheme(){
  const cur = document.documentElement.className === 'light' ? 'light' : 'dark';
  const next = cur === 'light' ? 'dark' : 'light';
  document.documentElement.className = next;
  localStorage.setItem('zultx_theme', next);
  updateThemeVisuals();
}
function updateThemeVisuals(){
  const isLight = document.documentElement.className === 'light';
  // sideTheme button icon
  if(sideTheme) sideTheme.textContent = isLight ? '🌞' : '🌙';
  // temp mini theme inside the small circle on mobile
  // code block highlighting re-check if present
}
sideTheme && sideTheme.addEventListener('click', toggleTheme);
applyInitialTheme();

/* ---------- Mobile menu toggle ---------- */
mobileMenuBtn && mobileMenuBtn.addEventListener('click', ()=>{
  sidePanel.classList.toggle('open');
  if(sidePanel.classList.contains('open')) showOverlay(); else hideOverlay();
});
function runCodeSandbox(code, lang, iframe){

  iframe.style.display = "block";

  if(lang === "html"){

    iframe.srcdoc = `
    <html>
    <head>
    <style>

    body{
      margin:0;
      padding:14px;
      font-family:monospace;

      background:linear-gradient(
        180deg,
        #fffafa,
        #ffe4ec
      );

      color:#111;
    }

    </style>
    </head>
    <body>

    ${code}

    </body>
    </html>
    `;

  }

  if(lang === "javascript" || lang === "js"){

    iframe.srcdoc = `
    <html>
    <head>

    <style>

    body{
      margin:0;
      padding:14px;
      font-family:monospace;
      font-size:14px;

      background:linear-gradient(
        180deg,
        #fffafa,
        #ffe4ec
      );

      color:#111;
    }

    #out{
      white-space:pre-wrap;
    }

    </style>

    </head>

    <body>

    <pre id="out"></pre>

    <script>

    const out = document.getElementById("out");

    const log = console.log;

    console.log = function(...args){
      const line = args.join(" ");
      out.innerHTML += "<div>"+line+"</div>";
      log.apply(console,args);
    };

    window.onerror = function(msg){
      out.textContent += "Error: " + msg;
    };

    try{
      ${code}
    }catch(e){
      out.textContent += e;
    }

    <\/script>

    <\/body>
    <\/html>
    `;
  }
}

function updateTokenDisplay(){
  const inputTokens = estimateTokens(input.value);
  const fileTokens = uploadedFiles.reduce((a,b)=>a+b.tokens,0);
  const total = inputTokens + fileTokens;

  const counter = document.getElementById("tokenCounter");
  if(counter){
    counter.textContent = `${total} tokens total`;
  }
}

input.addEventListener("input", updateTokenDisplay);

function estimateImageTokens(file){
  // simple token estimate for images
  return Math.ceil(file.size / 1000);
}

/* ---------- Temp button behavior (mobile-only) ---------- */
function updateTempVisibility(){
  if(!tempChatBtn) return;

  const hasMessages = !!chatArea.querySelector('.msg');

  if(hasMessages){
    tempChatBtn.style.display = 'none';
  } else {
    tempChatBtn.style.display = 'flex';
  }
}

tempChatBtn && tempChatBtn.addEventListener('click', ()=>{
  // open modal or start temp chat
  populateTempList();
  tempModal.style.display = 'block';
  showOverlay();
});
async function sendMessageFromSuggestion(text){
  if(!text) return;

  if(!currentConversationId){
    currentConversationId = makeConvoId();
    localSaveInit(currentConversationId);
  }

  addUserMsg(text);

  sendBtn.disabled = true;

  let aiDiv = document.createElement('div');
  aiDiv.className = 'msg ai thinking';
  aiDiv.innerHTML = `<div class="ai-pulse"></div>`;
  chatArea.appendChild(aiDiv);

  try{
    const token = getToken();
    const sessionId = ensureSessionId();
    const headers = { "Accept":"application/json", "X-Session-Id": sessionId };
    if(token) headers["Authorization"] = "Bearer " + token;

    const res = await fetch(`${API_BASE}/ask?q=${encodeURIComponent(text)}&source=suggestion`, {
      method:"GET",
      headers
    });

    if(!res.ok) throw new Error("Server error");

    const data = await res.json();
    const answer = data?.answer || "No response.";

    aiDiv.classList.remove("thinking");
    fakeStream(aiDiv, answer, 0);

    saveMessageToHistory(currentConversationId, 'ai', answer);

  } catch(err){
    console.error(err);
    aiDiv.classList.remove("thinking");
    aiDiv.textContent = "⚠️ Connection error.";
  } finally {
    sendBtn.disabled = false;
  }
}


/* start a fresh temporary chat (not saved) */
function createTempConversation(){
  tempMode = true;
  currentConversationId = makeConvoId('temp');
  chatArea.innerHTML = '';
  centerScreen.style.display = 'none';
  chatArea.style.display = 'flex';
  addSystemMsg('Temporary chat started (will not be saved).');
  updateTempVisibility();
}
async function loadSuggestions() {
  const grid = document.getElementById("suggestGrid");
  grid.innerHTML = "Loading suggestions...";

  try {
    const token = localStorage.getItem("zultx_token");
    const headers = token ? { Authorization: "Bearer " + token } : {};

    const res = await fetch(`${API_BASE}/suggestions`, { headers });
    const data = await res.json();

    grid.innerHTML = "";

    data.suggestions.forEach(text => {
      const card = document.createElement("div");
      card.className = "suggest-card";
      card.innerText = text;

      card.onclick = () => {

      centerScreen.style.display = 'none';
      chatArea.style.display = 'flex';
      suggestGrid.style.display = 'none';

     currentConversationId = null;
     chatArea.innerHTML = '';

     sendMessageFromSuggestion(text);
};


     grid.appendChild(card);
    });

  } catch (err) {
    grid.innerHTML = "Could not load suggestions.";
  }
}
/* helper used by side button mapping */
function startTemporaryChat(){
  // mobile- and desktop-consistent entrypoint for temporary chats
  // on desktop we push a temp conversation but also show a toast hint
  createTempConversation();
  tempModal.style.display = 'none';
  hideOverlay();
}

/* populate temp list — temp-conversations are saved under 'temp_' to local history if you choose to persist them */
function populateTempList(){
  tempList.innerHTML = '';
  const all = getAllHistory();
  const tempKeys = Object.keys(all).filter(k => k.startsWith('temp_')).sort((a,b)=> all[b].created - all[a].created);

  const newBtn = document.createElement('div');
  newBtn.className = 'menu-btn';
  newBtn.textContent = '+ New temporary chat';
  newBtn.onclick = ()=> { createTempConversation(); tempModal.style.display='none'; hideOverlay(); };
  tempList.appendChild(newBtn);

  if(tempKeys.length === 0){
    const p = document.createElement('div'); p.className='small-muted'; p.textContent='No temporary chats found (they are not saved by default).';
    tempList.appendChild(p);
    return;
  }

  for(const k of tempKeys){
    const entry = all[k];
    const el = document.createElement('div');
    el.className = 'temp-entry';
    el.innerHTML = `<div style="display:flex;flex-direction:column"><strong>${entry.id}</strong><div class="small-muted">${entry.messages && entry.messages.length ? entry.messages[entry.messages.length-1].text.slice(0,60) : 'Empty'}</div></div><div style="display:flex;gap:8px"><button data-id="${k}" class="open-temp">Open</button><button data-id="${k}" class="del-temp">Delete</button></div>`;
    tempList.appendChild(el);
  }

  tempList.querySelectorAll('.open-temp').forEach(btn=>{
    btn.onclick = ()=> {
      const id = btn.dataset.id;
      tempMode = true;
      currentConversationId = id;
      const convo = loadConversation(id);
      chatArea.innerHTML = '';
      centerScreen.style.display = 'none';
      chatArea.style.display = 'flex';
      if(convo && convo.messages) convo.messages.forEach(m=> addBubble(m.role === 'user' ? 'user' : 'ai', m.text, false));
      tempModal.style.display = 'none';
      hideOverlay();
      updateTempVisibility();
    };
  });
  tempList.querySelectorAll('.del-temp').forEach(btn=>{
    btn.onclick = async ()=>{
      const id = btn.dataset.id;
      if(!confirm("Delete this temporary chat?")) return;
      const al = getAllHistory();
      delete al[id];
      setAllHistory(al);
      populateTempList();
      refreshHistoryList();
    };
  });
}

/* close temp modal */
closeTempBtn && closeTempBtn.addEventListener('click', ()=> { tempModal.style.display = 'none'; hideOverlay(); });
