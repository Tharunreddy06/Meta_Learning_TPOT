document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("dataset");
  const fileName = document.getElementById("fileName");

  if (fileInput && fileName) {
    fileInput.addEventListener("change", () => {
      const name = fileInput.files && fileInput.files.length ? fileInput.files[0].name : "No file selected";
      fileName.textContent = name;
    });
  }

  const dropzone = document.getElementById("dropzone");
  if (dropzone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.add("loading-state");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (event) => {
        event.preventDefault();
        dropzone.classList.remove("loading-state");
      });
    });
  }
});
