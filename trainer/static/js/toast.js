function toast(messages){
    $.each(messages, function(index, message){
        Materialize.toast(message, 3000);
    });
}