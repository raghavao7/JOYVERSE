document.getElementById('addChildBtn').addEventListener('click', async () => {
    const name = document.getElementById('childName').value.trim();
    const phone = document.getElementById('phoneNumber').value.trim();

    // Basic validation
    if (!name || !phone) {
        alert('Please enter both name and phone number.');
        return;
    }
    if (!/^\d{10}$/.test(phone)) {
        alert('Phone number must be 10 digits.');
        return;
    }

    // Generate 6-digit code
    const code = Math.floor(100000 + Math.random() * 900000).toString();

    // Display the code
    const codeDisplay = document.getElementById('codeValue');
    codeDisplay.textContent = code;
    document.getElementById('generatedCode').style.display = 'block';

    // Send data to backend
    try {
        const response = await fetch('http://localhost:3000/register-child', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, phone, code })
        });

        if (!response.ok) throw new Error('Failed to register child');
        console.log('Child registered successfully');
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to register child. Check console for details.');
    }
});