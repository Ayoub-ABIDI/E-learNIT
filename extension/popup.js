// popup.js - Extension control logic with multi-platform support
const API_URL = 'http://localhost:5001';

let isRecognizing = false;
let recentSigns = [];
let currentPlatform = 'unknown';

// Detect current platform
async function detectPlatform() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const url = tab.url || '';
    
    if (url.includes('meet.google.com')) {
      currentPlatform = 'Google Meet';
      return '📗 Google Meet';
    } else if (url.includes('zoom.us')) {
      currentPlatform = 'Zoom';
      return '🔵 Zoom';
    } else if (url.includes('teams.microsoft.com')) {
      currentPlatform = 'Microsoft Teams';
      return '🟣 Microsoft Teams';
    } else {
      currentPlatform = 'Unknown';
      return '❓ Unknown Platform';
    }
  } catch (error) {
    console.error('Error detecting platform:', error);
    return '❓ Unknown';
  }
}

// Check server status on load
async function checkServerStatus() {
  const serverDot = document.getElementById('serverDot');
  const serverStatus = document.getElementById('serverStatus');
  
  try {
    const response = await fetch(`${API_URL}/health`);
    const data = await response.json();
    
    if (data.status === 'healthy') {
      serverDot.className = 'dot green';
      serverStatus.textContent = `Connected (${data.database_size} signs)`;
      return true;
    }
  } catch (error) {
    serverDot.className = 'dot red';
    serverStatus.textContent = 'Offline';
    return false;
  }
  return false;
}

// Update platform display
async function updatePlatformDisplay() {
  const platform = await detectPlatform();
  const platformDisplay = document.createElement('div');
  platformDisplay.className = 'status-row';
  platformDisplay.innerHTML = `
    <span class="status-label">🌐 Platform</span>
    <div class="status-indicator">
      <span class="dot ${currentPlatform !== 'Unknown' ? 'green' : 'yellow'}"></span>
      <span>${platform}</span>
    </div>
  `;
  
  const statusCard = document.querySelector('.status-card');
  if (statusCard) {
    statusCard.appendChild(platformDisplay);
  }
}

// Toggle recognition
document.getElementById('toggleBtn').addEventListener('click', async () => {
  const serverOnline = await checkServerStatus();
  
  if (!serverOnline) {
    alert('⚠️ Backend server is offline!\n\nPlease run:\npython backend_api.py');
    return;
  }
  
  // Check if on supported platform
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab.url || '';
  
  const supportedPlatforms = [
    'meet.google.com',
    'zoom.us',
    'teams.microsoft.com'
  ];
  
  const onSupportedPlatform = supportedPlatforms.some(platform => url.includes(platform));
  
  if (!onSupportedPlatform) {
    alert('⚠️ Please navigate to a supported platform:\n\n• Google Meet\n• Zoom\n• Microsoft Teams');
    return;
  }
  
  isRecognizing = !isRecognizing;
  
  const toggleBtn = document.getElementById('toggleBtn');
  const toggleIcon = document.getElementById('toggleIcon');
  const toggleText = document.getElementById('toggleText');
  const videoDot = document.getElementById('videoDot');
  const videoStatus = document.getElementById('videoStatus');
  const recognitionDot = document.getElementById('recognitionDot');
  const recognitionStatus = document.getElementById('recognitionStatus');
  
  if (isRecognizing) {
    toggleIcon.textContent = '⏸️';
    toggleText.textContent = 'Stop Recognition';
    toggleBtn.style.background = '#f87171';
    toggleBtn.style.color = 'white';
    
    videoDot.className = 'dot green';
    videoStatus.textContent = 'Active';
    recognitionDot.className = 'dot green';
    recognitionStatus.textContent = `Running on ${currentPlatform}`;
    
    // Send message to content script to start
    chrome.tabs.sendMessage(tab.id, { 
      action: 'START_RECOGNITION',
      settings: {
        showOverlay: document.getElementById('overlayToggle').checked,
        autoPhrase: document.getElementById('autoPhraseToggle').checked,
        soundAlerts: document.getElementById('soundToggle').checked,
        minConfidence: 0.65
      }
    });
  } else {
    toggleIcon.textContent = '▶️';
    toggleText.textContent = 'Start Recognition';
    toggleBtn.style.background = 'white';
    toggleBtn.style.color = '#3b5bdb';
    
    videoDot.className = 'dot yellow';
    videoStatus.textContent = 'Inactive';
    recognitionDot.className = 'dot yellow';
    recognitionStatus.textContent = 'Ready';
    
    // Send message to content script to stop
    chrome.tabs.sendMessage(tab.id, { action: 'STOP_RECOGNITION' });
  }
  
  // Save state
  chrome.storage.local.set({ isRecognizing });
});

// Clear history
document.getElementById('clearBtn').addEventListener('click', () => {
  recentSigns = [];
  updateRecentSigns();
  chrome.storage.local.set({ recentSigns: [] });
  
  // Visual feedback
  const signsList = document.getElementById('signsList');
  signsList.style.transition = 'opacity 0.3s';
  signsList.style.opacity = '0';
  setTimeout(() => {
    signsList.style.opacity = '1';
  }, 300);
});

// Update recent signs display
function updateRecentSigns() {
  const signsList = document.getElementById('signsList');
  
  if (recentSigns.length === 0) {
    signsList.innerHTML = '<span class="sign-chip">No signs detected yet</span>';
  } else {
    signsList.innerHTML = recentSigns.slice(-10).reverse().map(signData => {
      const sign = typeof signData === 'string' ? signData : signData.sign;
      const platform = signData.platform || '';
      const platformEmoji = 
        platform.includes('Google') ? '📗' :
        platform.includes('Zoom') ? '🔵' :
        platform.includes('Teams') ? '🟣' : '';
      
      return `<span class="sign-chip">${platformEmoji} ${sign}</span>`;
    }).join('');
  }
}

// Listen for updates from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'SIGN_DETECTED') {
    recentSigns.push({
      sign: message.sign,
      platform: currentPlatform,
      timestamp: Date.now()
    });
    updateRecentSigns();
    chrome.storage.local.set({ recentSigns });
  }
});

// Load saved state on popup open
chrome.storage.local.get(['isRecognizing', 'recentSigns'], (result) => {
  if (result.recentSigns) {
    recentSigns = result.recentSigns;
    updateRecentSigns();
  }
});

// Initialize on load
(async function init() {
  await checkServerStatus();
  await updatePlatformDisplay();
  
  // Re-check server status every 5 seconds
  setInterval(checkServerStatus, 5000);
})();

console.log('✅ E-learNIT popup loaded');
console.log('🌐 Multi-platform support: Google Meet, Zoom, Microsoft Teams');