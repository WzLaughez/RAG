document.getElementById('sendBtn').addEventListener('click', async () => {
  const promptInput = document.getElementById('promptInput');
  const chatHistory = document.getElementById('chatHistory');
  const prompt = promptInput.value.trim();

  if (!prompt) return;

  // Tampilkan pesan user
  const userMessage = document.createElement('div');
  userMessage.className = 'message user';
  userMessage.textContent = prompt;
  chatHistory.appendChild(userMessage);

  // Tampilkan loading message
  const loadingMessage = document.createElement('div');
  loadingMessage.className = 'message assistant loading';
  loadingMessage.textContent = 'Generating...';
  chatHistory.appendChild(loadingMessage);

  promptInput.value = '';
  promptInput.disabled = true;
  document.getElementById('sendBtn').disabled = true;

  try {
    const res = await fetch(`/get_answer?prompt=${encodeURIComponent(prompt)}`);
    const data = await res.json();

    // Ganti loading dengan hasil jawaban
    loadingMessage.classList.remove('loading');
    loadingMessage.textContent = data.history[data.history.length - 1][1]; // bot answer
  } catch (err) {
    loadingMessage.textContent = 'Gagal mendapatkan jawaban.';
  }

  promptInput.disabled = false;
  document.getElementById('sendBtn').disabled = false;
  promptInput.focus();
  chatHistory.scrollTop = chatHistory.scrollHeight;
});
