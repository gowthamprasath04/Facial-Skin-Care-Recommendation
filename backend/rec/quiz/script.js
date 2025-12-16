let currentQuestion = 1;
let skinType = '';

function nextQuestion(questionNumber) {
  const selectedOptions = document.querySelectorAll(`input[name="q${questionNumber}"]:checked`);

  if (selectedOptions.length > 0) {
    // Determine skin type only for question 2
    if (questionNumber === 2) {
      determineSkinType(selectedOptions[0].value);
    }

    hideCurrentQuestion(); // Hide current question regardless of number
    showNextQuestion(questionNumber + 1); // Move to the next question
  } else {
    alert('Please select an option.');
  }
}

function determineSkinType(option) {
  switch (option) {
    case 'a':
      skinType = 'Dry Skin';
      break;
    case 'b':
      skinType = 'Normal Skin';
      break;
    case 'c':
      skinType = 'Combination Skin';
      break;
    case 'd':
      skinType = 'Oily Skin';
      break;
    default:
      break;
  }
}

function hideCurrentQuestion() {
  const currentQuestionElement = document.getElementById(`question${currentQuestion}`);
  currentQuestionElement.style.display = 'none';
}

function showNextQuestion(questionNumber) {
  const nextQuestionElement = document.getElementById(`question${questionNumber}`);
  nextQuestionElement.style.display = 'block';

  currentQuestion = questionNumber;
}

function continueToNextQuestion() {
  hideResult();
  showNextQuestion(currentQuestion + 1);
}

// Function to hide result (assuming you have it implemented in HTML)
function hideResult() {
  const resultElement = document.getElementById('result');
  resultElement.style.display = 'none';
}
