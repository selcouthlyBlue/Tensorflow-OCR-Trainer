$(document).ready(function () {
   $('form[name="deletion"]').submit(function () {
       return confirm("Are you sure you want to delete this?");
   });
});