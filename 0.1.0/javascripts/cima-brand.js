document.addEventListener("DOMContentLoaded", () => {
  const logoLink = document.querySelector(".md-header__button.md-logo");
  if (logoLink) {
    logoLink.href = "https://www.cimafoundation.org";
    logoLink.target = "_blank";
    logoLink.rel = "noopener";
  }
});
