const activePage = window.location.pathname;
const hashPage = window.location.hash;
const link = document.getElementsByClassName("nav-link");

if (activePage === "/") {
  if (hashPage === "") {
    let active = document.querySelectorAll(".nav-item a[href='/']");
    active[0].className += " active";
  }
  else{
    let active = document.querySelectorAll(`.nav-item a[href='/${hashPage}']`);
    active[0].className += " active";    
  }
}

for (let i = 0; i < link.length; i++) {
  link[i].addEventListener("click", function () {
    let current = document.getElementsByClassName("active");
    current[0].className = current[0].className.replace("active", "");
    this.className += " active";
  });
}

if (activePage === "/predict") {
  let active = document.querySelectorAll(".nav-item a[href='/predict']");
  active[0].className += " active";
}

if (activePage === "/analyze") {
  let active = document.querySelectorAll(".nav-item a[href='/analyze']");
  active[0].className += " active";
}
