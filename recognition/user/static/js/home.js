document.addEventListener('DOMContentLoaded', function() {
  const html = document.documentElement;
  const themeToggle = document.getElementById('theme-toggle');
  const sunIcon = document.getElementById('sun-icon');
  const moonIcon = document.getElementById('moon-icon');
  const background = document.getElementById('background');

  if (themeToggle && sunIcon && moonIcon && background) {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);

    themeToggle.addEventListener('click', () => {
      const newTheme = html.classList.contains('dark') ? 'light' : 'dark';
      setTheme(newTheme);
      localStorage.setItem('theme', newTheme);
    });

    function setTheme(theme) {
      html.className = theme;
      sunIcon.style.display = theme === 'dark' ? 'block' : 'none';
      moonIcon.style.display = theme === 'light' ? 'block' : 'none';
      updateBackground(theme);
    }

    function updateBackground(theme) {
      const bgImage = theme === 'light'
        ? 'url(/static/img/light-background.png)'
        : 'url(/static/img/dark-background.png)';

      background.style.backgroundImage = bgImage;
    }
  } else {
    console.error('Неможливо знайти один або кілька необхідних елементів DOM');
  }
});