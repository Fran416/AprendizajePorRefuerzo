// =============================================================================
// app.js — Navegación, menú móvil e indicadores HTMX
// =============================================================================
// Gestiona el toggle del menú hamburguesa en móvil, cierra el sidebar al
// hacer clic en un enlace, y resalta el enlace activo según la URL actual.

document.addEventListener('DOMContentLoaded', function () {
    const menuToggle = document.getElementById('menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    // Menú hamburguesa: toggle sidebar en móvil
    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function () {
            sidebar.classList.toggle('open');
        });

        // Cerrar sidebar al hacer clic en un enlace (móvil)
        sidebar.querySelectorAll('nav a').forEach(function (link) {
            link.addEventListener('click', function () {
                if (window.innerWidth <= 768) {
                    sidebar.classList.remove('open');
                }
            });
        });
    }

    // Resaltar enlace activo según la URL actual
    if (sidebar) {
        const currentPath = window.location.pathname;
        sidebar.querySelectorAll('nav a').forEach(function (link) {
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }
});
