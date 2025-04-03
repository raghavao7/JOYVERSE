// document.getElementById('addChildBtn').addEventListener('click', async () => {
//     const name = document.getElementById('childName').value.trim();
//     const phone = document.getElementById('phoneNumber').value.trim();

//     // Basic validation
//     if (!name || !phone) {
//         alert('Please enter both name and phone number.');
//         return;
//     }
//     if (!/^\d{10}$/.test(phone)) {
//         alert('Phone number must be 10 digits.');
//         return;
//     }

//     // Generate 6-digit user ID and password
//     const code = Math.floor(100000 + Math.random() * 900000).toString();

//     // Display the generated code
//     document.getElementById('codeValue').textContent = code;
//     document.getElementById('generatedCode').style.display = 'block';

//     // Send data to backend
//     try {
//         const response = await fetch('http://localhost:3000/register', { // Fixed API endpoint
//             method: 'POST',
//             headers: { 'Content-Type': 'application/json' },
//             body: JSON.stringify({ name, phone, userId: code, password: code }) // Using code as userId and password
//         });

//         if (!response.ok) throw new Error('Failed to register child');
//         const result = await response.json();
//         alert(result.message);
//         console.log('Child registered successfully');
//     } catch (error) {
//         console.error('Error:', error);
//         alert('Failed to register child. Check console for details.');
//     }
// });



const socket = io("http://localhost:3000");

// 📌 Fetch current date
document.getElementById('currentDate').textContent = new Date().toLocaleDateString();

// 📌 Fetch recent children
async function fetchChildren() {
    const response = await fetch('http://localhost:3000/children');
    const children = await response.json();

    const tableBody = document.getElementById('childrenList');
    tableBody.innerHTML = ''; // Clear table

    children.forEach(child => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${child.name}</td>
            <td>${child.phone}</td>
            <td>${child.userId}</td>
            <td>${new Date(child.registeredAt).toLocaleString()}</td>
        `;
        tableBody.appendChild(row);
    });
}

// 📌 Listen for real-time updates
socket.on("newChild", fetchChildren);
socket.on("updateChildren", fetchChildren);

// 📌 Register child
document.getElementById('addChildBtn').addEventListener('click', async () => {
    const name = document.getElementById('childName').value.trim();
    const phone = document.getElementById('phoneNumber').value.trim();

    if (!name || !phone || !/^\d{10}$/.test(phone)) {
        alert('Please enter a valid name and 10-digit phone number.');
        return;
    }

    const code = Math.floor(100000 + Math.random() * 900000).toString();
    document.getElementById('codeValue').textContent = code;
    document.getElementById('generatedCode').style.display = 'block';

    try {
        const response = await fetch('http://localhost:3000/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, phone, userId: code, password: code })
        });

        if (!response.ok) throw new Error('Registration failed');
        alert('Child registered successfully!');
    } catch (error) {
        console.error(error);
        alert('Failed to register child.');
    }
});

// 📌 Initial Fetch
fetchChildren();
