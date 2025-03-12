# NewsFactChecker

A Chrome extension that identifies and corrects factual inaccuracies in online articles, with a focus on numerical claims.

## Overview

NewsFactChecker is a browser extension that analyzes online articles for factual accuracy. It specifically focuses on identifying incorrect numerical information (like dollar amounts, percentages, statistics) and provides the correct values alongside the article text.

## Features

- **Numerical Fact Checking**: Automatically identifies and corrects incorrect numerical values in articles
- **Visual Highlighting**: Marks factually incorrect information directly on the webpage with corrections
- **Correction Footnotes**: Adds a summary of all corrections at the bottom of the article
- **Detailed Analysis**: Provides a summary of the article's overall factual accuracy
- **Evidence Sources**: Links to authoritative sources for each correction

## Examples of Corrections

The extension can identify and correct various types of numerical inaccuracies such as:

- A news claim states "$215 billion" but the true figure is "$320 billion"
- An article mentions "45%" when the actual percentage is "37%"
- A report says "500 people" when the correct number is "720 people"

## Installation

### From Chrome Web Store (Recommended)

1. Visit the Chrome Web Store page for NewsFactChecker (link will be available after publishing)
2. Click "Add to Chrome"
3. Confirm by clicking "Add extension"

### From Source (Development)

1. Clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in the top-right corner)
4. Click "Load unpacked" and select the directory where you cloned the repository
5. The extension should now be installed and ready to use

## Testing the Extension

To test the extension with the latest accuracy improvements:

1. Load the extension in developer mode as described above
2. Open the included `demo.html` file in Chrome
3. Click the NewsFactChecker icon in your browser toolbar
4. Click "Check This Article" to analyze the demo content
5. Observe how the extension:
   - Correctly highlights complete percentage values (fixes partial highlighting issue)
   - Provides specific links to authoritative sources for each correction
   - Shows evidence in tooltips when hovering over corrections
   - Displays a comprehensive summary of corrections with source links

## Usage

1. Navigate to any news article or blog post
2. Click the NewsFactChecker icon in your browser toolbar
3. Click "Check This Article" to begin analysis
4. View the overall accuracy rating and specific corrections in the popup
5. Factually incorrect statements will be highlighted directly on the webpage, with hover tooltips showing the correct values
6. A summary of all corrections is added at the bottom of the article
7. Click on any source link to see the evidence for a correction

## Publishing to Chrome Web Store

If you want to publish this extension to the Chrome Web Store, follow these steps:

### Step 1: Prepare Your Extension

1. Make sure you have proper icon files in the `icons` folder (16x16, 48x48, and 128x128 PNG files)
2. Update the manifest.json file with your details
3. Create a ZIP file of your extension folder (don't include .git directory or unnecessary files)

### Step 2: Create a Developer Account

1. Go to the [Chrome Web Store Developer Dashboard](https://chrome.google.com/webstore/devconsole/)
2. Sign in with your Google account
3. Pay the one-time $5 registration fee if you haven't already

### Step 3: Create a New Item

1. Click "New Item" in the developer dashboard
2. Upload your ZIP file
3. Fill in the required information:
   - Detailed description
   - At least 1-3 screenshots (1280x800 or 640x400)
   - Store icon (128x128)
   - Category (Productivity)
   - Language
4. Complete the privacy practices section
5. Submit for review

The review process typically takes a few business days. Once approved, your extension will be available in the Chrome Web Store.

## Technical Implementation

The extension uses pattern matching and heuristics to:

1. Identify claims containing numerical values (currency, percentages, statistics)
2. Determine if the values are accurate
3. Generate corrected values when inaccuracies are found
4. Present corrections in an easy-to-understand format
5. Link to authoritative sources for each correction

In a production environment, this would be enhanced with:
- Real fact databases for verification
- Machine learning for claim extraction and verification
- API connections to trusted data sources

## Future Improvements

- Connect to factual databases to perform real verification (rather than simulation)
- Implement machine learning for better claim extraction
- Add user feedback mechanisms to improve accuracy
- Support more languages and numerical formats
- Create a database of common factual errors and their corrections

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.