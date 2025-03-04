function handleInput(currentInput) {
    currentInput.value = currentInput.value.replace(/[^0-9]/g, '');
    if (currentInput.value.length === 1) {
        const nextInput = currentInput.nextElementSibling;
        if (nextInput && nextInput.classList.contains('digit-box')) {
            nextInput.focus();
        }
    }
    validateInputs();
}

function handleKeyDown(currentInput, event) {
    if (event.key === 'Backspace' && currentInput.value.length === 0) {
        const prevInput = currentInput.previousElementSibling;
        if (prevInput && prevInput.classList.contains('digit-box')) {
            prevInput.focus();
        }
    }
    if (event.key === 'Enter') {
        validateLogin();
    }
}

function validateInputs() {
    const username = document.getElementById('Username').value; // FIXED ID
    const digitBoxes = document.querySelectorAll('.digit-box');
    const loginBtn = document.getElementById('login-btn');
    const errorMessage = document.getElementById('error-message');

    const allFilled = username.length > 0 && Array.from(digitBoxes).every(box => box.value.length === 1 && /^\d$/.test(box.value));

    if (allFilled) {
        loginBtn.style.background = '#32cd32';
        errorMessage.textContent = "";
    } else {
        loginBtn.style.background = '#555';
        errorMessage.textContent = username.length === 0
            ? "Please enter your username!"
            : "Please fill all 6 boxes with digits!";
    }
}

function validateLogin() {
    const username = document.getElementById('Username').value; // FIXED ID
    const digitBoxes = document.querySelectorAll('.digit-box');
    const errorMessage = document.getElementById('error-message');
    const loginBtn = document.getElementById('login-btn');

    if (!username) {
        errorMessage.textContent = "Please enter your username!";
        return;
    }

    const id = Array.from(digitBoxes).map(box => box.value).join('');
    if (id.length !== 6 || !/^\d{6}$/.test(id)) {
        errorMessage.textContent = "Please fill all 6 boxes with digits!";
        return;
    }

    errorMessage.textContent = "";
    loginBtn.disabled = true;
    loginBtn.textContent = "Logging in...";

    fetch('http://localhost:3000/login', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, id }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert(data.message);
            window.location.href = '/dashboard.html';
        } else {
            errorMessage.textContent = "Invalid credentials!"; // SHOW ERROR MESSAGE FOR INCORRECT CREDENTIALS
        }
    })
    .catch(error => {
        errorMessage.textContent = "Failed to connect. Try again!";
    })
    .finally(() => {
        loginBtn.disabled = false;
        loginBtn.textContent = "Join the Fun!";
    });
}


// Fix paste event for all digit boxes
document.querySelectorAll('.digit-box').forEach((box, index, boxes) => {
    box.addEventListener('paste', (event) => {
        const pasteData = (event.clipboardData || window.clipboardData).getData('text');
        if (/^\d{6}$/.test(pasteData)) {
            const digits = pasteData.split('');
            boxes.forEach((box, i) => box.value = digits[i] || '');
            validateInputs();
        }
        event.preventDefault();
    });
});
