document.addEventListener('DOMContentLoaded', function() {
    const dropZone = document.getElementById('drop-zone');
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.querySelector('input[type="file"]');

    if (fileInput) {
      fileInput.id = 'file-input';
    }

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');

      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const files = e.dataTransfer.files;

        if (fileInput && fileInput.files) {
          const dataTransfer = new DataTransfer();
          dataTransfer.items.add(files[0]);
          fileInput.files = dataTransfer.files;
          updateDropzoneWithFileName(files[0].name);
        }
      }
    });

    dropZone.addEventListener('click', () => {
      fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        const fileName = e.target.files[0].name;
        updateDropzoneWithFileName(fileName);
      }
    });

    function updateDropzoneWithFileName(fileName) {
        const uploadText = dropZone.querySelector('.upload-text');
        const uploadSubtext = dropZone.querySelector('.upload-subtext');

        uploadText.textContent = 'Selected: ' + fileName;
        uploadSubtext.textContent = 'Click "Upload Files" to proceed';
      }
    });