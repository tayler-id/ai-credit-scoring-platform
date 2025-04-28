document.getElementById('basic-profile-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());

    // Convert years_in_business to integer, handle empty string
    data.years_in_business = data.years_in_business ? parseInt(data.years_in_business, 10) : null;

    // Set occupation to null if empty string
    data.occupation = data.occupation || null;

    const statusElement = document.getElementById('profile-status');
    statusElement.textContent = 'Submitting...';
    statusElement.style.color = 'orange';

    console.log('Sending Basic Profile data:', JSON.stringify(data)); // Add logging

    try {
        const response = await fetch('http://127.0.0.1:8000/api/v1/mvp/basic_profile', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (response.ok) {
            statusElement.textContent = 'Basic Profile submitted successfully!';
            statusElement.style.color = 'green';
        } else {
            const error = await response.json();
            console.error('Basic Profile submission error:', error); // Log the full error
            statusElement.textContent = `Error submitting Basic Profile: ${error.detail || JSON.stringify(error)}`;
            statusElement.style.color = 'red';
        }
    } catch (error) {
        console.error('Basic Profile submission fetch error:', error); // Log fetch error
        statusElement.textContent = `Error submitting Basic Profile: ${error}`;
        statusElement.style.color = 'red';
    }
});

document.getElementById('declared-income-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    // Convert monthly_income to float, handle empty string
    data.monthly_income = data.monthly_income ? parseFloat(data.monthly_income) : null;


    const statusElement = document.getElementById('income-status');
    statusElement.textContent = 'Submitting...';
    statusElement.style.color = 'orange';

    console.log('Sending Declared Income data:', JSON.stringify(data)); // Add logging

    try {
        const response = await fetch('http://127.0.0.1:8000/api/v1/mvp/declared_income', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        if (response.ok) {
            statusElement.textContent = 'Declared Income submitted successfully!';
            statusElement.style.color = 'green';
        } else {
            const error = await response.json();
            console.error('Declared Income submission error:', error); // Log the full error
            statusElement.textContent = `Error submitting Declared Income: ${error.detail || JSON.stringify(error)}`;
            statusElement.style.color = 'red';
        }
    } catch (error) {
        console.error('Declared Income submission fetch error:', error); // Log fetch error
        statusElement.textContent = `Error submitting Declared Income: ${error}`;
        statusElement.style.color = 'red';
    }
});

document.getElementById('score-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const applicantId = formData.get('applicant_id');

    const scoreElement = document.getElementById('basic-score');
    const loanOfferElement = document.getElementById('loan-offer');
    const statusElement = document.getElementById('score-status');

    scoreElement.textContent = 'Calculating...';
    loanOfferElement.textContent = 'Calculating...';
    statusElement.textContent = 'Requesting score...';
    statusElement.style.color = 'orange';


    try {
        const response = await fetch(`http://127.0.0.1:8000/api/v1/mvp/score?applicant_id=${applicantId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (response.ok) {
            const result = await response.json();
            scoreElement.textContent = result.basic_score; // Use basic_score from response
            loanOfferElement.textContent = result.loan_offer || 'N/A'; // Use loan_offer if available, otherwise N/A
            statusElement.textContent = 'Score retrieved successfully!';
            statusElement.style.color = 'green';
        } else {
             const error = await response.json();
             console.error('Get Score error:', error); // Log the full error
            scoreElement.textContent = 'N/A';
            loanOfferElement.textContent = 'N/A';
            statusElement.textContent = `Error getting score: ${error.detail || JSON.stringify(error)}`;
            statusElement.style.color = 'red';
        }
    } catch (error) {
        console.error('Get Score fetch error:', error); // Log fetch error
        scoreElement.textContent = 'N/A';
        loanOfferElement.textContent = 'N/A';
        statusElement.textContent = `Error getting score: ${error}`;
        statusElement.style.color = 'red';
    }
});
