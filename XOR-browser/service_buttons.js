let special_button_hide = document.querySelector(".css-omocl")
let reload_page = document.querySelector(".reload-page")
special_button_hide.addEventListener("click", () => {
    document.querySelector("html").style.marginRight = "0px";
    reload_page.style.right = "50px"
})

let special_button_maximize = document.querySelectorAll(".css-1x8bgru")[0]
special_button_maximize.style.display = "none";