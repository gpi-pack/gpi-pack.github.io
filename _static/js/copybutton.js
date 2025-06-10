document.addEventListener('DOMContentLoaded', function() {
    // Find all code blocks
    const codeBlocks = document.querySelectorAll('div.highlight pre');
    
    // Add copy button to each code block
    codeBlocks.forEach(function(codeBlock) {
      // Create button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.textContent = 'Copy';
      
      // Add button to the code block container
      const container = codeBlock.parentNode;
      container.style.position = 'relative';
      container.appendChild(copyButton);
      
      // Add click event
      copyButton.addEventListener('click', function() {
        // Get the text content
        const code = codeBlock.textContent;
        
        // Copy to clipboard
        navigator.clipboard.writeText(code).then(function() {
          // Show success feedback
          copyButton.textContent = 'Copied!';
          setTimeout(function() {
            copyButton.textContent = 'Copy';
          }, 2000);
        }).catch(function(error) {
          console.error('Failed to copy: ', error);
          copyButton.textContent = 'Error!';
          setTimeout(function() {
            copyButton.textContent = 'Copy';
          }, 2000);
        });
      });
    });
  });