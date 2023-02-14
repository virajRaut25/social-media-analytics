const link = document.getElementsByClassName("nav-link");
for (let i = 0; i < link.length; i++) {
  link[i].addEventListener("click", function () {
    let current = document.getElementsByClassName("active");
    current[0].className = current[0].className.replace("active", "");
    this.className += " active";
  });
}
