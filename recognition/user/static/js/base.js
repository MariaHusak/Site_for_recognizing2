document.addEventListener('DOMContentLoaded', function() {
  const html = document.documentElement;
  const themeToggle = document.getElementById('theme-toggle');
  const sunIcon = document.getElementById('sun-icon');
  const moonIcon = document.getElementById('moon-icon');
  const sidebar = document.getElementById('sidebar');
  const container = document.querySelector('.container');
  const hamburgerLines = document.querySelectorAll('.hamburger-line');
  const mainContent = document.getElementById('main-content');

  const savedTheme = localStorage.getItem('theme') || 'light';
  setTheme(savedTheme);

  themeToggle.addEventListener('click', () => {
    const newTheme = html.classList.contains('dark') ? 'light' : 'dark';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
  });

  function setTheme(theme) {
    if (theme === 'dark') {
      html.classList.add('dark');
      html.classList.remove('light');
    } else {
      html.classList.add('light');
      html.classList.remove('dark');
    }
    sunIcon.style.display = theme === 'dark' ? 'block' : 'none';
    moonIcon.style.display = theme === 'light' ? 'block' : 'none';

    const sidebarColor = theme === 'light' ? '#99BBC7' : '#0B2833';
    sidebar.style.backgroundColor = sidebarColor;
    if (container) {
      container.style.backgroundColor = sidebarColor;
    }

    const hamburgerColor = theme === 'light' ? 'black' : 'white';
    hamburgerLines.forEach(line => {
      line.style.backgroundColor = hamburgerColor;
    });
  }

  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebarState = localStorage.getItem('sidebarState') || 'open';
  if (sidebarState === 'closed') {
    sidebar.classList.add('closed');
    mainContent.classList.add('sidebar-closed');
  }

  sidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('closed');
    mainContent.classList.toggle('sidebar-closed');
    const newState = sidebar.classList.contains('closed') ? 'closed' : 'open';
    localStorage.setItem('sidebarState', newState);
  });

  const navLinks = document.querySelectorAll('.nav-link');
  const currentPath = window.location.pathname;

  navLinks.forEach(link => {
    if (link.getAttribute('href') === currentPath) {
      link.classList.add('active');
    }
  });
});