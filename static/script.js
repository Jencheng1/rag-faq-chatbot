// Main JavaScript for the chatbot interface
document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatButton = document.getElementById('chatButton');
    const chatContainer = document.getElementById('chatContainer');
    const closeChat = document.getElementById('closeChat');
    const userInput = document.getElementById('userInput');
    const sendMessage = document.getElementById('sendMessage');
    const chatMessages = document.getElementById('chatMessages');

    // Toggle chat window
    chatButton.addEventListener('click', function() {
        chatContainer.style.display = 'flex';
        chatButton.style.display = 'none';
        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
        // Focus on input
        userInput.focus();
    });

    // Close chat window
    closeChat.addEventListener('click', function() {
        chatContainer.style.display = 'none';
        chatButton.style.display = 'flex';
    });

    // Send message on button click
    sendMessage.addEventListener('click', sendUserMessage);

    // Send message on Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendUserMessage();
        }
    });

    // Function to send user message
    function sendUserMessage() {
        const message = userInput.value.trim();
        if (message === '') return;

        // Add user message to chat
        addMessage(message, 'user');

        // Clear input
        userInput.value = '';

        // Show typing indicator
        showTypingIndicator();

        // Send message to server
        fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: message }),
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();

            // Add bot response to chat
            if (data.error) {
                addMessage('Sorry, I encountered an error. Please try again later.', 'bot');
                console.error(data.error);
            } else {
                addMessage(data.answer, 'bot');
            }
        })
        .catch(error => {
            // Remove typing indicator
            removeTypingIndicator();

            // Add error message
            addMessage('Sorry, I encountered an error. Please try again later.', 'bot');
            console.error('Error:', error);
        });
    }

    // Function to add message to chat
    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        const paragraph = document.createElement('p');
        paragraph.textContent = text;

        contentDiv.appendChild(paragraph);
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);

        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to show typing indicator
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-indicator-container';
        typingDiv.id = 'typingIndicator';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content typing-indicator';

        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            contentDiv.appendChild(dot);
        }

        typingDiv.appendChild(contentDiv);
        chatMessages.appendChild(typingDiv);

        // Scroll to bottom of chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Function to remove typing indicator
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
});
