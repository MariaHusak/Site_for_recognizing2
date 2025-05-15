document.addEventListener('DOMContentLoaded', function() {
    const html = document.documentElement;
    const themeBackground = document.getElementById('theme-background');
    const appContainer = document.querySelector('.app-container');

    const container = document.querySelector('.container');
    if (container) {
      container.classList.remove('glass-card');
    }

    function updateBackgroundImage() {
      const isDarkMode = html.classList.contains('dark');
      const imagePath = isDarkMode
        ? "/static/img/dark.png"
        : "/static/img/light.png";

      themeBackground.src = imagePath;
    }

    updateBackgroundImage();

    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
      themeToggle.addEventListener('click', () => {
        setTimeout(updateBackgroundImage, 50);
      });
    }

    window.addEventListener('load', updateBackgroundImage);
  });