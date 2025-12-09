// content.js - E-learNIT: IMPROVED PHRASE CONSTRUCTION
const API_URL = 'http://localhost:5001';

let isRecognizing = false;
let recognitionInterval = null;
let signSequence = [];
let lastDetectedSign = null;
let lastSignTime = Date.now();
let overlayElement = null;
let isPhraseConstructing = false; // NEW: Flag to pause during phrase construction
let settings = {
  showOverlay: true,
  autoPhrase: false,
  soundAlerts: false,
  language: 'tunisian',
  minConfidence: 0.65,
  duplicateTimeWindow: 3000 // 3 seconds to filter duplicates
};

let speechSynthesis = window.speechSynthesis;
let isSpeaking = false;

console.log('🤟 E-learNIT Sign Language Extension loaded');

// ============================================
// DEDUPLICATION HELPER
// ============================================
function cleanSignSequence(signs) {
  if (signs.length === 0) return [];
  
  const cleaned = [signs[0]];
  
  for (let i = 1; i < signs.length; i++) {
    // Only add if different from previous sign
    if (signs[i] !== signs[i - 1]) {
      cleaned.push(signs[i]);
    }
  }
  
  console.log('🧹 Cleaned sequence:', signs.length, '→', cleaned.length);
  return cleaned;
}

// ============================================
// CREATE OVERLAY UI
// ============================================
function createOverlay() {
  if (overlayElement) return;
  
  const logoUrl = chrome.runtime.getURL("logo-White.png");
  
  overlayElement = document.createElement('div');
  overlayElement.id = 'elearnit-overlay';
  overlayElement.innerHTML = `
    <div class="elearnit-container">
      <div class="elearnit-header">
        <img class="elearnit-logo" src="${logoUrl}" alt="E-learNIT Logo" style="width: 40px; height: 40px; margin-right: 8px;">
        <span class="elearnit-title">E-learNIT</span>
        <button class="elearnit-close" id="closeOverlay">✕</button>
      </div>
      <div class="elearnit-content">
        <div class="elearnit-status">
          <span class="elearnit-indicator" id="statusIndicator"></span>
          <span id="statusText">Ready to recognize...</span>
        </div>
        
        <div class="elearnit-current-sign">
          <div class="sign-label">Current Sign</div>
          <div class="sign-value" id="currentSign">-</div>
          <div class="sign-confidence" id="confidence">-</div>
        </div>
        
        <div class="elearnit-sequence">
          <div class="sequence-label">
            Sign Sequence (<span id="signCount">0</span> signs)
            <span id="cleanedCount" style="opacity: 0.6; font-size: 10px;"></span>
          </div>
          <div class="sequence-value" id="signSequence">No signs detected</div>
          <div class="sequence-actions" style="margin-top: 10px; display: flex; gap: 8px;">
            <button id="constructPhraseBtn" class="elearnit-btn elearnit-btn-primary" disabled>
              📝 Construct Phrase
            </button>
            <button id="pauseBtn" class="elearnit-btn elearnit-btn-warning" style="background: #f59e0b;">
              ⏸️ Pause
            </button>
            <button id="clearSequenceBtn" class="elearnit-btn elearnit-btn-secondary">
              🗑️ Clear
            </button>
          </div>
        </div>
        
        <div class="elearnit-phrase" id="phraseContainer" style="display: none;">
          <div class="phrase-label">📝 Constructed Phrase</div>
          <div class="phrase-value" id="phraseTunisian" style="font-size: 16px; font-weight: 700; margin-bottom: 10px; color: #fbbf24;"></div>
          <div class="phrase-value" id="phraseFrench" style="font-size: 14px; margin-bottom: 8px; color: #a5b4fc;"></div>
          <div class="phrase-translation" id="phraseEnglish" style="font-size: 13px; color: #86efac;"></div>
          <div class="phrase-context" id="phraseContext" style="font-size: 11px; opacity: 0.85; margin-top: 10px; padding: 8px; background: rgba(255,255,255,0.1); border-radius: 6px; font-style: italic;"></div>
          <div class="phrase-actions" style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
            <button id="speakPhraseBtn" class="elearnit-btn elearnit-btn-speak">
              🔊 Speak Aloud
            </button>
            <button id="sendToMeetBtn" class="elearnit-btn elearnit-btn-success">
              💬 Send to Chat
            </button>
            <select id="languageSelect" class="elearnit-select">
              <option value="tunisian">🇹🇳 Tunisian</option>
              <option value="french">🇫🇷 Français</option>
              <option value="english">🇬🇧 English</option>
            </select>
          </div>
        </div>
        
        <div class="elearnit-history" id="historyContainer" style="margin-top: 15px; max-height: 120px; overflow-y: auto; display: none;">
          <div class="history-label">📋 History</div>
          <div id="historyList" style="font-size: 12px; opacity: 0.9;"></div>
        </div>
        
        <div class="elearnit-debug" id="debugInfo" style="font-size: 10px; opacity: 0.7; margin-top: 10px;">
          Ready...
        </div>
      </div>
    </div>
  `;
  
  document.body.appendChild(overlayElement);
  
  // Event listeners
  document.getElementById('closeOverlay').addEventListener('click', () => {
    overlayElement.style.display = 'none';
  });
  
  document.getElementById('constructPhraseBtn').addEventListener('click', constructPhrase);
  
  document.getElementById('pauseBtn').addEventListener('click', togglePause);
  
  document.getElementById('clearSequenceBtn').addEventListener('click', () => {
    signSequence = [];
    lastDetectedSign = null;
    updateSequenceDisplay();
    hidePhraseContainer();
    updateDebugInfo('Sequence cleared');
    showNotification('🗑️ Sequence cleared', 'info');
  });
  
  document.getElementById('speakPhraseBtn').addEventListener('click', speakPhrase);
  
  document.getElementById('sendToMeetBtn').addEventListener('click', sendPhraseToMeetChat);
  
  document.getElementById('languageSelect').addEventListener('change', (e) => {
    settings.language = e.target.value;
    updateDebugInfo(`Language: ${e.target.value}`);
  });
  
  console.log('✅ Overlay created');
}

// ============================================
// PAUSE/RESUME TOGGLE
// ============================================
function togglePause() {
  isPhraseConstructing = !isPhraseConstructing;
  const pauseBtn = document.getElementById('pauseBtn');
  const statusIndicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');
  
  if (isPhraseConstructing) {
    pauseBtn.innerHTML = '▶️ Resume';
    pauseBtn.style.background = '#10b981';
    statusIndicator.style.background = '#f59e0b';
    statusText.textContent = 'Recognition PAUSED';
    updateDebugInfo('⏸️ Recognition paused');
    showNotification('⏸️ Recognition paused', 'info');
  } else {
    pauseBtn.innerHTML = '⏸️ Pause';
    pauseBtn.style.background = '#f59e0b';
    statusIndicator.style.background = '#4ade80';
    statusText.textContent = 'Recognizing...';
    updateDebugInfo('▶️ Recognition resumed');
    showNotification('▶️ Recognition resumed', 'info');
  }
}

function updateDebugInfo(message) {
  const debugInfo = document.getElementById('debugInfo');
  if (debugInfo) {
    const timestamp = new Date().toLocaleTimeString();
    debugInfo.textContent = `[${timestamp}] ${message}`;
  }
  console.log(`🔍 ${message}`);
}

function updateSequenceDisplay() {
  const sequenceDiv = document.getElementById('signSequence');
  const constructBtn = document.getElementById('constructPhraseBtn');
  const signCount = document.getElementById('signCount');
  const cleanedCount = document.getElementById('cleanedCount');
  
  const cleanedSigns = cleanSignSequence(signSequence);
  
  if (signCount) {
    signCount.textContent = signSequence.length;
  }
  
  if (cleanedCount) {
    cleanedCount.textContent = cleanedSigns.length !== signSequence.length 
      ? `(${cleanedSigns.length} unique)` 
      : '';
  }
  
  if (sequenceDiv) {
    const recentSigns = cleanedSigns.slice(-10);
    sequenceDiv.textContent = recentSigns.join(' → ') || 'No signs detected';
  }
  
  if (constructBtn) {
    constructBtn.disabled = cleanedSigns.length < 2;
  }
}

function hidePhraseContainer() {
  const phraseContainer = document.getElementById('phraseContainer');
  if (phraseContainer) {
    phraseContainer.style.display = 'none';
  }
}

// ============================================
// VIDEO CAPTURE
// ============================================
function captureVideoFrame() {
  const videoElements = document.querySelectorAll('video');
  
  if (videoElements.length === 0) return null;
  
  for (let video of videoElements) {
    if (video.videoWidth < 50 || video.videoHeight < 50) continue;
    if (video.paused || video.ended) continue;
    if (video.readyState < 2) continue;
    
    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      return canvas.toDataURL('image/jpeg', 0.8);
    } catch (error) {
      console.error('Capture error:', error);
    }
  }
  
  return null;
}

// ============================================
// SIGN RECOGNITION
// ============================================
async function recognizeSign(imageData) {
  if (!imageData) return null;
  
  // SKIP if paused
  if (isPhraseConstructing) {
    updateDebugInfo('⏸️ Recognition paused, skipping...');
    return null;
  }
  
  try {
    const response = await fetch(`${API_URL}/recognize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData })
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const result = await response.json();
    
    if (result.success && result.confidence >= settings.minConfidence) {
      const currentTime = Date.now();
      const timeSinceLastSign = currentTime - lastSignTime;
      
      // DEDUPLICATION: Only add if different OR enough time passed
      if (result.sign !== lastDetectedSign || timeSinceLastSign > settings.duplicateTimeWindow) {
        updateOverlay(result);
        signSequence.push(result.sign);
        lastDetectedSign = result.sign;
        lastSignTime = currentTime;
        
        updateSequenceDisplay();
        addToHistory('Sign', result.sign, result.confidence);
        updateDebugInfo(`✅ ${result.sign} (${(result.confidence * 100).toFixed(0)}%)`);
        
        if (settings.soundAlerts) {
          playNotificationSound();
        }
      } else {
        updateDebugInfo(`🔄 Duplicate skipped: ${result.sign}`);
      }
      
      return result;
    } else if (result.success) {
      updateDebugInfo(`⚠️ Low confidence: ${result.sign} (${(result.confidence * 100).toFixed(0)}%)`);
    }
  } catch (error) {
    console.error('Recognition error:', error);
    updateDebugInfo(`❌ Error: ${error.message}`);
  }
  
  return null;
}

function updateOverlay(result) {
  if (!overlayElement || overlayElement.style.display === 'none') return;
  
  const currentSign = document.getElementById('currentSign');
  const confidence = document.getElementById('confidence');
  
  if (currentSign) {
    currentSign.textContent = result.sign;
    currentSign.style.color = result.confidence > 0.7 ? '#4ade80' : '#fbbf24';
  }
  
  if (confidence) {
    confidence.textContent = `${(result.confidence * 100).toFixed(0)}% confidence`;
  }
}

function addToHistory(type, text, confidence = null) {
  const historyContainer = document.getElementById('historyContainer');
  const historyList = document.getElementById('historyList');
  
  if (historyContainer) historyContainer.style.display = 'block';
  
  if (historyList) {
    const entry = document.createElement('div');
    entry.style.padding = '4px 0';
    entry.style.borderBottom = '1px solid rgba(255,255,255,0.1)';
    const confText = confidence ? ` (${(confidence * 100).toFixed(0)}%)` : '';
    entry.innerHTML = `<span style="opacity: 0.6;">${new Date().toLocaleTimeString()}</span> [${type}] → ${text}${confText}`;
    historyList.insertBefore(entry, historyList.firstChild);
    
    while (historyList.children.length > 10) {
      historyList.removeChild(historyList.lastChild);
    }
  }
}

// ============================================
// NOTIFICATION
// ============================================
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.style.cssText = `
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 16px 24px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    z-index: 10000000;
    font-family: 'Segoe UI', sans-serif;
    font-size: 14px;
    font-weight: 600;
    animation: slideInRight 0.3s;
    ${type === 'success' ? 'background: rgba(16, 185, 129, 0.95); color: white;' : ''}
    ${type === 'error' ? 'background: rgba(239, 68, 68, 0.95); color: white;' : ''}
    ${type === 'info' ? 'background: rgba(59, 130, 246, 0.95); color: white;' : ''}
  `;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => notification.remove(), 3000);
}

function playNotificationSound() {
  try {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);
  } catch (error) {
    console.error('Sound error:', error);
  }
}

// ============================================
// PHRASE CONSTRUCTION (IMPROVED)
// ============================================
async function constructPhrase() {
  const cleanedSigns = cleanSignSequence(signSequence);
  
  if (cleanedSigns.length < 2) {
    showNotification('❌ Need at least 2 unique signs', 'error');
    return;
  }
  
  try {
    // AUTO-PAUSE recognition during phrase construction
    const wasRecognizing = !isPhraseConstructing;
    if (wasRecognizing) {
      isPhraseConstructing = true;
      updateDebugInfo('⏸️ Auto-paused for phrase construction');
    }
    
    updateDebugInfo(`🔤 Constructing phrase from ${cleanedSigns.length} signs...`);
    showNotification('🔄 Constructing phrase...', 'info');
    
    // Send CLEANED sequence to backend
    const response = await fetch(`${API_URL}/construct_phrase`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signs: cleanedSigns }) // Send cleaned signs!
    });
    
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    
    const result = await response.json();
    
    console.log('📦 Backend response:', result);
    
    if (result.success) {
      displayPhrase(result);
      addToHistory('Phrase', result.phrase_tunisian || result.phrase_french || result.phrase_english);
      showNotification('✅ Phrase constructed!', 'success');
      updateDebugInfo('✅ Phrase ready');
    } else {
      showNotification('❌ Could not construct phrase', 'error');
      updateDebugInfo('❌ Phrase construction failed');
    }
    
    // Keep paused after phrase construction (user can manually resume)
    
  } catch (error) {
    console.error('Phrase error:', error);
    showNotification('❌ Error: ' + error.message, 'error');
    updateDebugInfo('❌ Error: ' + error.message);
  }
}

// ============================================
// DISPLAY PHRASE (IMPROVED)
// ============================================
function displayPhrase(result) {
  const phraseContainer = document.getElementById('phraseContainer');
  const phraseTunisian = document.getElementById('phraseTunisian');
  const phraseFrench = document.getElementById('phraseFrench');
  const phraseEnglish = document.getElementById('phraseEnglish');
  const phraseContext = document.getElementById('phraseContext');
  
  console.log('📝 Displaying phrase:', result);
  
  if (phraseContainer) phraseContainer.style.display = 'block';
  
  // Tunisian phrase
  if (phraseTunisian) {
    const tunisianText = result.phrase_tunisian || 'N/A';
    phraseTunisian.innerHTML = `🇹🇳 <strong>Tunisian:</strong> ${tunisianText}`;
  }
  
  // French phrase
  if (phraseFrench) {
    const frenchText = result.phrase_french || 'N/A';
    phraseFrench.innerHTML = `🇫🇷 <strong>French:</strong> ${frenchText}`;
  }
  
  // English phrase
  if (phraseEnglish) {
    const englishText = result.phrase_english || 'N/A';
    phraseEnglish.innerHTML = `🇬🇧 <strong>English:</strong> ${englishText}`;
  }
  
  // Context (VERY IMPORTANT!)
  if (phraseContext) {
    if (result.context && result.context.trim() !== '') {
      phraseContext.innerHTML = `💡 <strong>Context:</strong> ${result.context}`;
      phraseContext.style.display = 'block';
    } else {
      phraseContext.style.display = 'none';
    }
  }
  
  window.currentPhrase = result;
}

// ============================================
// SPEECH SYNTHESIS
// ============================================
function speakPhrase() {
  if (!window.currentPhrase) {
    updateDebugInfo('No phrase to speak');
    showNotification('❌ No phrase available', 'error');
    return;
  }
  
  if (isSpeaking) {
    speechSynthesis.cancel();
    isSpeaking = false;
    updateSpeakButton(false);
    return;
  }
  
  const language = document.getElementById('languageSelect').value;
  let text = '';
  let lang = '';
  
  switch (language) {
    case 'tunisian':
      text = window.currentPhrase.phrase_tunisian || window.currentPhrase.phrase_french;
      lang = 'fr-FR'; // French voice for Tunisian
      break;
    case 'french':
      text = window.currentPhrase.phrase_french;
      lang = 'fr-FR';
      break;
    case 'english':
      text = window.currentPhrase.phrase_english;
      lang = 'en-US';
      break;
  }
  
  if (!text || text === 'N/A') {
    updateDebugInfo('No text to speak');
    showNotification('❌ No text in selected language', 'error');
    return;
  }
  
  console.log('🔊 Speaking:', text, 'in', lang);
  updateDebugInfo(`🔊 Speaking...`);
  
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = lang;
  utterance.rate = 0.85;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;
  
  const voices = speechSynthesis.getVoices();
  const preferredVoice = voices.find(v => v.lang.startsWith(lang.split('-')[0]));
  if (preferredVoice) {
    utterance.voice = preferredVoice;
  }
  
  utterance.onstart = () => {
    isSpeaking = true;
    updateSpeakButton(true);
    showNotification('🔊 Speaking...', 'info');
  };
  
  utterance.onend = () => {
    isSpeaking = false;
    updateSpeakButton(false);
    updateDebugInfo('✅ Finished speaking');
  };
  
  utterance.onerror = (event) => {
    console.error('Speech error:', event);
    isSpeaking = false;
    updateSpeakButton(false);
    showNotification('❌ Speech error', 'error');
  };
  
  speechSynthesis.speak(utterance);
  addToHistory('Speech', text);
}

function updateSpeakButton(speaking) {
  const speakBtn = document.getElementById('speakPhraseBtn');
  if (!speakBtn) return;
  
  if (speaking) {
    speakBtn.innerHTML = '⏸️ Stop';
    speakBtn.classList.add('speaking');
  } else {
    speakBtn.innerHTML = '🔊 Speak Aloud';
    speakBtn.classList.remove('speaking');
  }
}

// ============================================
// SEND TO CHAT (Multi-platform support)
// ============================================
function sendPhraseToMeetChat() {
  if (!window.currentPhrase) {
    showNotification('❌ No phrase available', 'error');
    return;
  }
  
  const language = document.getElementById('languageSelect').value;
  let text = '';
  
  switch (language) {
    case 'tunisian':
      text = window.currentPhrase.phrase_tunisian;
      break;
    case 'french':
      text = window.currentPhrase.phrase_french;
      break;
    case 'english':
      text = window.currentPhrase.phrase_english;
      break;
  }
  
  if (!text || text === 'N/A') {
    showNotification('❌ No text in selected language', 'error');
    return;
  }
  
  console.log('📤 Sending to chat:', text);
  updateDebugInfo('📤 Detecting platform and sending...');
  
  // Detect platform
  const url = window.location.href;
  let platform = 'unknown';
  
  if (url.includes('meet.google.com')) {
    platform = 'google_meet';
  } else if (url.includes('zoom.us')) {
    platform = 'zoom';
  } else if (url.includes('teams.microsoft.com')) {
    platform = 'teams';
  }
  
  console.log('🌐 Detected platform:', platform);
  updateDebugInfo(`📱 Platform: ${platform}`);
  
  // Platform-specific chat selectors
  let chatInput = null;
  let sendButton = null;
  
  if (platform === 'google_meet') {
    // Google Meet selectors
    const meetSelectors = [
      'textarea[placeholder*="message" i]',
      'textarea[aria-label*="message" i]',
      '[contenteditable="true"][role="textbox"]',
      'input[placeholder*="message" i]'
    ];
    
    for (const selector of meetSelectors) {
      chatInput = document.querySelector(selector);
      if (chatInput) break;
    }
    
    sendButton = document.querySelector(
      '[aria-label*="Send" i], [aria-label*="Envoyer" i], button[jsname="ufXA2e"]'
    );
  } 
  else if (platform === 'teams') {
    // Microsoft Teams selectors
    const teamsSelectors = [
      '[data-tid="ckeditor-chatInputBox"]',
      'div[role="textbox"][contenteditable="true"]',
      '[aria-label*="Type a message" i]',
      '[aria-label*="Écrire un message" i]',
      'div.ck-content[contenteditable="true"]',
      'div[data-track-module="chatInput"]'
    ];
    
    for (const selector of teamsSelectors) {
      chatInput = document.querySelector(selector);
      if (chatInput) {
        console.log('✅ Found Teams chat input:', selector);
        break;
      }
    }
    
    // Teams send button selectors
    const teamsSendSelectors = [
      'button[data-tid="newMessageCommands-send"]',
      'button[aria-label*="Send" i]',
      'button[aria-label*="Envoyer" i]',
      'button[title*="Send" i]',
      'button[data-track-module="sendButton"]'
    ];
    
    for (const selector of teamsSendSelectors) {
      sendButton = document.querySelector(selector);
      if (sendButton) {
        console.log('✅ Found Teams send button:', selector);
        break;
      }
    }
  } 
  else if (platform === 'zoom') {
    // Zoom selectors
    const zoomSelectors = [
      'textarea[placeholder*="message" i]',
      'textarea.chat-box__chat-textarea',
      '[contenteditable="true"][role="textbox"]'
    ];
    
    for (const selector of zoomSelectors) {
      chatInput = document.querySelector(selector);
      if (chatInput) break;
    }
    
    sendButton = document.querySelector(
      'button.chat-box__send-btn, button[aria-label*="Send" i]'
    );
  }
  
  if (!chatInput) {
    showNotification(`❌ Please open the ${platform.replace('_', ' ')} chat first!`, 'error');
    updateDebugInfo(`❌ Chat input not found for ${platform}`);
    return;
  }
  
  console.log('✅ Found chat input:', chatInput);
  
  try {
    chatInput.focus();
    
    // Insert text based on input type
    if (chatInput.contentEditable === 'true' || chatInput.getAttribute('contenteditable') === 'true') {
      // ContentEditable (Teams, some Meet views)
      console.log('📝 Using contenteditable method');
      
      // Clear existing content
      chatInput.innerHTML = '';
      
      // Insert text
      if (platform === 'teams') {
        // Teams uses CKEditor-like structure
        const paragraph = document.createElement('p');
        paragraph.textContent = text;
        chatInput.appendChild(paragraph);
      } else {
        chatInput.textContent = text;
      }
      
      // Trigger input events
      chatInput.dispatchEvent(new Event('input', { bubbles: true }));
      chatInput.dispatchEvent(new Event('change', { bubbles: true }));
      
      // For Teams, also trigger keyup
      if (platform === 'teams') {
        chatInput.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));
        chatInput.dispatchEvent(new InputEvent('input', { 
          bubbles: true,
          cancelable: true,
          inputType: 'insertText',
          data: text
        }));
      }
      
    } else {
      // Textarea/Input (Zoom, some Meet views)
      console.log('📝 Using textarea method');
      
      const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value'
      )?.set || Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value'
      )?.set;
      
      if (nativeInputValueSetter) {
        nativeInputValueSetter.call(chatInput, text);
      } else {
        chatInput.value = text;
      }
      
      chatInput.dispatchEvent(new Event('input', { bubbles: true }));
      chatInput.dispatchEvent(new Event('change', { bubbles: true }));
    }
    
    // Send message
    setTimeout(() => {
      if (sendButton && sendButton.offsetParent !== null) {
        // Button is visible and clickable
        console.log('✅ Clicking send button');
        sendButton.click();
        showNotification('✅ Message sent!', 'success');
        updateDebugInfo('✅ Message sent via button');
      } else {
        // Try Enter key as fallback
        console.log('⌨️ Using Enter key');
        chatInput.dispatchEvent(new KeyboardEvent('keydown', {
          key: 'Enter',
          code: 'Enter',
          keyCode: 13,
          which: 13,
          bubbles: true,
          cancelable: true
        }));
        showNotification('✅ Message sent!', 'success');
        updateDebugInfo('✅ Message sent via Enter key');
      }
    }, 300); // Slightly longer delay for Teams
    
  } catch (error) {
    console.error('❌ Error sending:', error);
    showNotification('❌ Error: ' + error.message, 'error');
    updateDebugInfo('❌ Error: ' + error.message);
  }
}

// ============================================
// START/STOP RECOGNITION
// ============================================
function startRecognition() {
  if (isRecognizing) return;
  
  isRecognizing = true;
  isPhraseConstructing = false;
  
  if (settings.showOverlay) {
    createOverlay();
    overlayElement.style.display = 'block';
  }
  
  updateDebugInfo('🚀 Recognition started');
  
  setTimeout(() => {
    const frame = captureVideoFrame();
    if (frame) recognizeSign(frame);
  }, 1000);
  
  recognitionInterval = setInterval(async () => {
    if (!isPhraseConstructing) { // Only recognize if not paused
      const frame = captureVideoFrame();
      if (frame) await recognizeSign(frame);
    }
  }, 4000);
}

function stopRecognition() {
  isRecognizing = false;
  
  if (recognitionInterval) {
    clearInterval(recognitionInterval);
    recognitionInterval = null;
  }
  
  if (overlayElement) {
    overlayElement.style.display = 'none';
  }
  
  updateDebugInfo('🛑 Recognition stopped');
}

// ============================================
// MESSAGE LISTENER
// ============================================
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'START_RECOGNITION') {
    settings = { ...settings, ...message.settings };
    startRecognition();
    sendResponse({ status: 'started' });
  } else if (message.action === 'STOP_RECOGNITION') {
    stopRecognition();
    sendResponse({ status: 'stopped' });
  }
  
  return true;
});

window.speechSynthesis.onvoiceschanged = () => {
  const voices = window.speechSynthesis.getVoices();
  console.log('🎤 Available voices:', voices.length);
};

console.log('✅ E-learNIT ready - Sign → Text → Speech');