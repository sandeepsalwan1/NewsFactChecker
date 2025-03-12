// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Get DOM elements
  const checkPageButton = document.getElementById('checkPage');
  const settingsButton = document.getElementById('settingsBtn');
  const settingsPanel = document.getElementById('settingsPanel');
  const saveSettingsButton = document.getElementById('saveSettings');
  const confidenceThreshold = document.getElementById('confidenceThreshold');
  const confidenceValue = document.getElementById('confidenceValue');
  const loader = document.getElementById('loader');
  const results = document.getElementById('results');
  const summary = document.getElementById('summary');
  const factsList = document.getElementById('facts');

  // Load saved settings
  chrome.storage.sync.get({
    confidenceThreshold: 70,
    sources: ['wikipedia', 'news', 'factCheckers']
  }, function(items) {
    confidenceThreshold.value = items.confidenceThreshold;
    confidenceValue.textContent = items.confidenceThreshold + '%';
    
    // Select the saved sources
    const sourcesSelect = document.getElementById('checkingSources');
    for (let i = 0; i < sourcesSelect.options.length; i++) {
      sourcesSelect.options[i].selected = items.sources.includes(sourcesSelect.options[i].value);
    }
  });

  // Update confidence value display when slider changes
  confidenceThreshold.addEventListener('input', function() {
    confidenceValue.textContent = this.value + '%';
  });

  // Toggle settings panel visibility
  settingsButton.addEventListener('click', function() {
    settingsPanel.style.display = settingsPanel.style.display === 'none' ? 'block' : 'none';
  });

  // Save settings
  saveSettingsButton.addEventListener('click', function() {
    const sourcesSelect = document.getElementById('checkingSources');
    const selectedSources = Array.from(sourcesSelect.selectedOptions).map(option => option.value);
    
    chrome.storage.sync.set({
      confidenceThreshold: confidenceThreshold.value,
      sources: selectedSources
    }, function() {
      settingsPanel.style.display = 'none';
      // Show a saved notification
      const notification = document.createElement('div');
      notification.textContent = 'Settings saved!';
      notification.style.color = 'green';
      notification.style.marginTop = '8px';
      notification.style.textAlign = 'center';
      document.querySelector('.settings').appendChild(notification);
      
      // Remove notification after 2 seconds
      setTimeout(() => {
        notification.remove();
      }, 2000);
    });
  });

  // Check the current page for facts
  checkPageButton.addEventListener('click', function() {
    // Show loader and hide previous results
    loader.style.display = 'block';
    results.style.display = 'none';
    summary.innerHTML = '';
    factsList.innerHTML = '';
    
    // Get the active tab
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
      const activeTab = tabs[0];
      
      // Send a message to the content script to analyze the page
      chrome.tabs.sendMessage(activeTab.id, {action: "analyze"}, function(response) {
        if (chrome.runtime.lastError) {
          // Handle error when content script is not loaded
          displayError("Could not connect to the page. Please refresh and try again.");
          return;
        }
        
        if (response && response.status === "analyzing") {
          // Start polling for results from the background script
          checkAnalysisStatus(activeTab.id);
        } else {
          displayError("Failed to start analysis. Please try again.");
        }
      });
    });
  });

  // Poll for analysis results
  function checkAnalysisStatus(tabId) {
    chrome.runtime.sendMessage({action: "getAnalysisStatus", tabId: tabId}, function(response) {
      if (response && response.status === "complete") {
        displayResults(response.results);
      } else if (response && response.status === "error") {
        displayError(response.message || "An error occurred during analysis.");
      } else {
        // Continue polling
        setTimeout(() => checkAnalysisStatus(tabId), 1000);
      }
    });
  }

  // Display error message
  function displayError(message) {
    loader.style.display = 'none';
    results.style.display = 'block';
    summary.innerHTML = `<div style="color: red;">${message}</div>`;
  }

  // Display analysis results
  function displayResults(data) {
    loader.style.display = 'none';
    results.style.display = 'block';
    
    // Count corrections
    const correctedFacts = data.facts.filter(fact => !fact.isTrue && fact.correction);
    
    // Display summary
    const accuracyPercentage = data.overallAccuracy;
    const accuracyClass = accuracyPercentage >= 80 ? 'success' : accuracyPercentage >= 60 ? 'warning' : 'danger';
    
    summary.innerHTML = `
      <h3>Article Analysis</h3>
      <p>Overall Accuracy: <span style="color: var(--${accuracyClass}-color); font-weight: bold;">${accuracyPercentage}%</span></p>
      <p>${data.summarySentence || ''}</p>
      ${correctedFacts.length > 0 ? 
        `<p><strong>${correctedFacts.length} numerical ${correctedFacts.length === 1 ? 'value' : 'values'} corrected</strong></p>` : 
        ''}
    `;
    
    // Display individual fact checks
    if (data.facts && data.facts.length > 0) {
      data.facts.forEach(fact => {
        const factItem = document.createElement('div');
        factItem.className = `fact-item fact-${fact.isTrue ? 'true' : 'uncertain' ? 'uncertain' : 'false'}`;
        
        let correctionHTML = '';
        if (!fact.isTrue && fact.correction) {
          correctionHTML = `
            <div class="correction">
              <span style="text-decoration: line-through; color: var(--danger-color);">${fact.correction.originalValue}</span>
              <span style="margin: 0 5px;">→</span>
              <span style="font-weight: bold; color: var(--success-color);">${fact.correction.correctedValue}</span>
            </div>
          `;
        }
        
        // Create enhanced source link with more detailed information
        let sourceLinkHTML = '';
        if (fact.sourceURL && fact.evidenceText) {
          sourceLinkHTML = `
            <div class="source-info">
              <div class="source-label">Source:</div>
              <div class="source-text">${fact.evidenceText}</div>
              <a href="${fact.sourceURL}" target="_blank" class="source-link">
                View Source Document
              </a>
            </div>
          `;
        }
        
        factItem.innerHTML = `
          <div class="claim"><strong>Claim:</strong> "${fact.claim}"</div>
          <div class="verdict">
            <strong>Verdict:</strong> 
            <span class="verdict-${fact.isTrue ? 'true' : 'false'}">${fact.isTrue ? 'True' : 'False'}</span>
          </div>
          ${correctionHTML}
          ${fact.explanation ? `<div class="explanation"><strong>Explanation:</strong> ${fact.explanation}</div>` : ''}
          ${sourceLinkHTML}
          <div class="confidence">
            <strong>Confidence:</strong> 
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${fact.confidence}%; background-color: var(--${fact.confidence >= 70 ? 'success' : 'warning'}-color);"></div>
            </div>
            <span class="confidence-value">${fact.confidence}%</span>
          </div>
        `;
        
        factsList.appendChild(factItem);
      });
    } else {
      factsList.innerHTML = '<p>No specific facts were analyzed in this article.</p>';
    }
  }
}); 