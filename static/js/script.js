document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatOutput = document.getElementById('chat-box');

    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const userInput = chatInput.value;

        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': document.getElementById('csrf_token').value
            },
            body: JSON.stringify({ input: userInput })
        })
        .then(response => response.json())
        .then(data => {
            const userMessage = document.createElement('div');
            userMessage.classList.add('message', 'user-message');
            userMessage.textContent = 'You: ' + userInput;
            chatOutput.appendChild(userMessage);

            const botMessage = document.createElement('div');
            botMessage.classList.add('message', 'bot-message');
            botMessage.textContent = 'Bot: ' + data.response;
            chatOutput.appendChild(botMessage);

            chatInput.value = '';
        })
        .catch(error => console.error('Error:', error));
    });
});
