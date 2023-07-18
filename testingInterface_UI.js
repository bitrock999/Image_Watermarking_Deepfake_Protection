$(document).ready(function() {
    // Set up the drop area as a droppable element
    $("#drop-area").droppable({
      drop: function(event, ui) {
        var file = ui.helper[0].files[0];
        displayInputImage(file);
      },
      accept: "image/*"
    });
  
    // Handle the file input change event
    $("#input-image").on("change", function(event) {
      var file = event.target.files[0];
      displayInputImage(file);
    });
  
    // Handle the select image button click event
    $("#select-image").on("click", function() {
      $("#input-image").click();
    });
  
    // Function to display the input image
    function displayInputImage(file) {
      if (file) {
        var reader = new FileReader();
        reader.onload = function(event) {
          var image = new Image();
          image.onload = function() {
            $("#output-image").attr("src", image.src);
          };
          image.src = event.target.result;
        };
        reader.readAsDataURL(file);
  
        $("#drop-text").hide();
        $("#input-image").show();
      }
    }
  });
  