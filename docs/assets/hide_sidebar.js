document.addEventListener("DOMContentLoaded", function() {
    // Só aplica em desktop
    const isDesktop = window.matchMedia("(min-width: 1024px)").matches;

    if (isDesktop) {
        // Procura o link ativo do menu
        const activeLink = document.querySelector(".md-nav__link--active");
        if (activeLink && activeLink.textContent.trim() === "Início") {
            // Oculta a sidebar
            const sidebar = document.querySelector(".md-sidebar--primary[data-md-type='navigation']");
            if (sidebar) sidebar.style.display = "none";

            // Centraliza o conteúdo
            const main = document.querySelector(".md-main__inner");
            if (main) {
                main.style.marginLeft = "auto";
                main.style.marginRight = "auto";
                main.style.maxWidth = "1000px";
            }

            const content = document.querySelector(".md-content");
            if (content) content.style.margin = "0 auto";
        }
    }
});
