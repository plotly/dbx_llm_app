// assets/typewriter.js
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        typeWriterEffect: function(text) {
            const target = document.getElementById('model-output');
            target.innerHTML = ""; // Clear existing content
            let i = 0;
            function typing() {
                if (i < text.length) {
                    target.innerHTML += text.charAt(i);
                    i++;
                    setTimeout(typing, 10); // Typing speed
                }
            }
            if (text) {
                typing();
            }
            return ''; // Return empty string to dummy output
        }
    }
});
