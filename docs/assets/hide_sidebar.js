document.addEventListener("DOMContentLoaded", function() {
    const isHome = window.location.pathname.endsWith("/") || window.location.pathname.endsWith("index.html");
    const isDesktop = window.matchMedia("(min-width: 1024px)").matches; // só desktop

    if (isHome && isDesktop) {
        const sidebar = document.querySelector(".md-sidebar--primary[data-md-type='navigation']");
        if (sidebar) sidebar.style.display = "none";

        const main = document.querySelector(".md-main__inner");
        if (main) {
            main.style.marginLeft = "auto";
            main.style.marginRight = "auto";
            main.style.maxWidth = "1000px";
        }

        const content = document.querySelector(".md-content");
        if (content) content.style.margin = "0 auto";
    }
});
