let updateTimeout;
const loading = document.querySelector('.loading');

// Preset configurations
const presets = {
    'hardy-increase': {
        'bleaching-population': 17.25,
        'bleaching-colony': 32.25,
        'coral-damage': 2.50,
        'giant-clam-40-50': 0.00,
        'giant-clam-50-plus': 0.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.00,
        'grouper-40-50': 0.00,
        'grouper-total': 0.00,
        'total-giant-clam': 8.00,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.00,
        'tripneustes': 0.00
    },
    'low-isles-increase': {
        'bleaching-population': 0.00,
        'bleaching-colony': 0.00,
        'coral-damage': 0.50,
        'giant-clam-40-50': 0.50,
        'giant-clam-50-plus': 1.25,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.00,
        'grouper-40-50': 0.00,
        'grouper-total': 0.00,
        'total-giant-clam': 7.00,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.00,
        'tripneustes': 0.00
    },
    'hastings-increase': {
        'bleaching-population': 58.75,
        'bleaching-colony': 71.25,
        'coral-damage': 1.75,
        'giant-clam-40-50': 0.00,
        'giant-clam-50-plus': 0.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.00,
        'grouper-40-50': 0.00,
        'grouper-total': 0.00,
        'total-giant-clam': 0.50,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.00,
        'tripneustes': 0.00
    },
    'mandu-increase': {
        'bleaching-population': 5.00,
        'bleaching-colony': 20.00,
        'coral-damage': 1.25,
        'giant-clam-40-50': 0.00,
        'giant-clam-50-plus': 0.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.00,
        'grouper-40-50': 0.00,
        'grouper-total': 0.00,
        'total-giant-clam': 0.25,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.00,
        'tripneustes': 0.00
    },
    'john-brewer-decrease': {
        'bleaching-population': 25.00,
        'bleaching-colony': 34.50,
        'coral-damage': 1.50,
        'giant-clam-40-50': 0.25,
        'giant-clam-50-plus': 0.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.00,
        'grouper-40-50': 0.00,
        'grouper-total': 0.00,
        'total-giant-clam': 1.50,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.00,
        'tripneustes': 0.00
    },
    'palm-island-decrease': {
        'bleaching-population': 4.25,
        'bleaching-colony': 39.50,
        'coral-damage': 2.25,
        'giant-clam-40-50': 0.00,
        'giant-clam-50-plus': 1.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 1.00,
        'grouper-40-50': 0.00,
        'grouper-total': 1.00,
        'total-giant-clam': 47.00,
        'humphead-wrasse': 0.00,
        'parrotfish': 5.25,
        'tripneustes': 0.00
    },
    'shag-rock-decrease': {
        'bleaching-population': 1.75,
        'bleaching-colony': 19.50,
        'coral-damage': 2.25,
        'giant-clam-40-50': 0.00,
        'giant-clam-50-plus': 0.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.00,
        'grouper-40-50': 0.00,
        'grouper-total': 0.00,
        'total-giant-clam': 0.50,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.00,
        'tripneustes': 14.00
    },
    'heron-decrease': {
        'bleaching-population': 1.00,
        'bleaching-colony': 12.50,
        'coral-damage': 0.50,
        'giant-clam-40-50': 0.00,
        'giant-clam-50-plus': 0.00,
        'grouper-60-plus': 0.00,
        'grouper-30-40': 0.25,
        'grouper-40-50': 0.50,
        'grouper-total': 0.75,
        'total-giant-clam': 1.00,
        'humphead-wrasse': 0.00,
        'parrotfish': 0.25,
        'tripneustes': 0.00
    }
};

// Handle preset selection
document.getElementById('preset').addEventListener('change', function() {
    const selectedPreset = presets[this.value];
    if (selectedPreset) {
        Object.keys(selectedPreset).forEach(id => {
            const value = selectedPreset[id];
            document.getElementById(id).value = value;
            document.getElementById(id + '-num').value = value;
        });
        getPrediction();
    }
});

// Sync slider and number input values
document.querySelectorAll('input[type="range"]').forEach(slider => {
    const numInput = document.getElementById(slider.id + '-num');
    
    slider.addEventListener('input', () => {
        numInput.value = slider.value;
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(getPrediction, 500);
    });

    numInput.addEventListener('input', () => {
        slider.value = numInput.value;
        clearTimeout(updateTimeout);
        updateTimeout = setTimeout(getPrediction, 500);
    });
});

async function getPrediction() {
    loading.classList.add('active');
    
    try {
        const response = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                grouper_60_plus: document.getElementById('grouper-60-plus').value,
                grouper_30_40: document.getElementById('grouper-30-40').value,
                grouper_40_50: document.getElementById('grouper-40-50').value,
                grouper_total: document.getElementById('grouper-total').value,
                giant_clam_40_50: document.getElementById('giant-clam-40-50').value,
                giant_clam_50_plus: document.getElementById('giant-clam-50-plus').value,
                total_giant_clam: document.getElementById('total-giant-clam').value,
                bleaching_population: document.getElementById('bleaching-population').value,
                bleaching_colony: document.getElementById('bleaching-colony').value,
                coral_damage: document.getElementById('coral-damage').value,
                humphead_wrasse: document.getElementById('humphead-wrasse').value,
                parrotfish: document.getElementById('parrotfish').value,
                tripneustes: document.getElementById('tripneustes').value
            })
        });

        const data = await response.json();
        updateDisplay(data);
    } catch (error) {
        console.error('Prediction failed:', error);
        document.getElementById('prediction').textContent = 'Error';
        document.getElementById('confidence').textContent = 'Prediction service unavailable';
    } finally {
        loading.classList.remove('active');
    }
}

function updateDisplay(data) {
    // Update prediction text
    const predictionElem = document.getElementById('prediction');
    predictionElem.textContent = data.prediction.charAt(0).toUpperCase() + data.prediction.slice(1);
    
    // Find highest confidence
    const maxConfidence = Math.max(...Object.values(data.probabilities));
    document.getElementById('confidence').textContent = `${maxConfidence.toFixed(1)}% confident`;

    // Update gauge rotation
    const gaugeBackground = document.querySelector('.gauge-background');
    let rotation = 0;
    if (data.prediction === 'increase') rotation = 240;
    else if (data.prediction === 'decrease') rotation = 0;
    else rotation = 120;
    
    gaugeBackground.style.transform = `rotate(${rotation}deg)`;
}

// Initial prediction
getPrediction();
