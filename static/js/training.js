document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("trainingForm");
  const button = document.getElementById("trainButton");

  if (!form || !button) {
    return;
  }

  form.addEventListener("submit", () => {
    button.disabled = true;
    button.textContent = "Optimizing Pipeline...";
    button.classList.add("loading-state");
  });
});
