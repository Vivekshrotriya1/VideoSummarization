// Temporary local development backend.
const API_BASE_URL = "http://localhost:8000";

const analyzeButton = document.querySelector("#analyze-button");
const askButton = document.querySelector("#ask-button");
const explainButton = document.querySelector("#explain-button");
const clearHistoryButton = document.querySelector("#clear-history-button");
const questionInput = document.querySelector("#question");
const videoStatus = document.querySelector("#video-status");
const statusMessage = document.querySelector("#status");
const questionSection = document.querySelector("#question-section");
const answerSection = document.querySelector("#answer-section");
const chatHistoryElement = document.querySelector("#chat-history");

let currentVideoUrl = null;
let chatHistory = [];
const MAX_CONVERSATION_CONTEXT_CHARACTERS = 1200;

function setStatus(message, isError = false) {
  statusMessage.textContent = typeof message === "string" ? message : JSON.stringify(message);
  statusMessage.classList.toggle("error", isError);
}

function getErrorMessage(error) {
  if (error instanceof Error) return getErrorMessage(error.message);
  if (typeof error === "string") return error;
  if (Array.isArray(error)) {
    return error.map((item) => getErrorMessage(item)).filter(Boolean).join(" ");
  }
  if (error && typeof error === "object") {
    if (error.detail) return getErrorMessage(error.detail);
    if (error.message) return getErrorMessage(error.message);
    if (error.msg) return error.msg;
  }
  return "Something went wrong. Please try again.";
}

function throwApiError(data, fallbackMessage) {
  throw data?.detail || data?.message || fallbackMessage;
}

function formatTime(seconds) {
  const totalSeconds = Math.floor(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

function getRecentConversationContext() {
  const recentMessages = [];
  let characterCount = 0;

  for (const item of [...chatHistory].reverse()) {
    const message = `${item.role}: ${item.text}`;
    if (characterCount + message.length > MAX_CONVERSATION_CONTEXT_CHARACTERS) {
      break;
    }
    recentMessages.unshift(message);
    characterCount += message.length;
  }

  return recentMessages.join("\n");
}

async function getCurrentYouTubeVideo() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab?.url || !/^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\//.test(tab.url)) {
    throw new Error("Please open a YouTube video first.");
  }

  const url = new URL(tab.url);
  const isWatchPage = url.hostname.includes("youtube.com") && url.pathname === "/watch" && url.searchParams.has("v");
  const isShortLink = url.hostname === "youtu.be" && url.pathname.length > 1;
  const isShortsPage = url.pathname.startsWith("/shorts/");
  if (!isWatchPage && !isShortLink && !isShortsPage) {
    throw new Error("Please open a YouTube video page first.");
  }

  return { url: tab.url, title: tab.title || "Current YouTube video" };
}

async function loadCurrentVideo() {
  try {
    const video = await getCurrentYouTubeVideo();
    currentVideoUrl = video.url;
    videoStatus.textContent = video.title;
    analyzeButton.disabled = false;
  } catch (error) {
    currentVideoUrl = null;
    videoStatus.textContent = error.message;
    analyzeButton.disabled = true;
  }
}

async function analyzeVideo() {
  try {
    const video = await getCurrentYouTubeVideo();
    currentVideoUrl = video.url;
    videoStatus.textContent = video.title;
  } catch (error) {
    setStatus(getErrorMessage(error), true);
    return;
  }

  analyzeButton.disabled = true;
  setStatus("Analyzing transcript. This can take a few minutes for a long video...");
  clearChatHistory(false);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url: currentVideoUrl })
    });
    const data = await response.json();
    if (!response.ok) throwApiError(data, "Could not analyze the video.");

    questionSection.hidden = false;
    setStatus(`Ready. ${data.chunk_count} transcript chunks analyzed.`);
    questionInput.focus();
  } catch (error) {
    setStatus(getErrorMessage(error), true);
  } finally {
    analyzeButton.disabled = false;
  }
}

function renderSources(sources) {
  const sourcesElement = document.createElement("div");
  sourcesElement.className = "sources";
  sources.forEach((source) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "source-link";
    button.textContent = `Watch at ${formatTime(source.start_time)}–${formatTime(source.end_time)}`;
    button.addEventListener("click", () => seekToTimestamp(source.start_time));
    sourcesElement.append(button);
  });
  return sourcesElement;
}

function addChatMessage(role, text, sources = []) {
  const message = document.createElement("article");
  message.className = `chat-message ${role}`;

  const label = document.createElement("strong");
  label.textContent = role === "user" ? "You" : "Assistant";
  const content = document.createElement("p");
  content.textContent = text;
  message.append(label, content);

  if (sources.length) {
    const sourceTitle = document.createElement("span");
    sourceTitle.className = "source-title";
    sourceTitle.textContent = "Sources";
    message.append(sourceTitle, renderSources(sources));
  }

  chatHistoryElement.append(message);
  answerSection.hidden = false;
  message.scrollIntoView({ behavior: "smooth", block: "end" });
}

function clearChatHistory(showStatus = true) {
  chatHistory = [];
  chatHistoryElement.replaceChildren();
  answerSection.hidden = true;
  if (showStatus) setStatus("Conversation cleared.");
}

async function seekToTimestamp(timestamp) {
  try {
    const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (!tab?.id) throw new Error("Please open the YouTube video tab first.");

    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      args: [timestamp],
      func: (time) => {
        const player = document.querySelector("video");
        if (!player) throw new Error("YouTube player was not found.");
        player.currentTime = time;
        player.play();
      }
    });
    setStatus(`Jumped to ${formatTime(timestamp)}.`);
  } catch (error) {
    setStatus(getErrorMessage(error) || "Could not jump to this timestamp.", true);
  }
}

async function askQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("Please enter a question.", true);
    return;
  }

  askButton.disabled = true;
  setStatus("Finding the answer...");
  try {
    const previousConversation = getRecentConversationContext();
    const questionWithHistory = previousConversation
      ? `Previous conversation:\n${previousConversation}\n\nCurrent question: ${question}`
      : question;
    const response = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: questionWithHistory })
    });
    const data = await response.json();
    if (!response.ok) throwApiError(data, "Could not answer the question.");

    addChatMessage("user", question);
    addChatMessage("assistant", data.answer, data.sources);
    chatHistory.push({ role: "User", text: question });
    chatHistory.push({ role: "Assistant", text: data.answer });
    questionInput.value = "";
    setStatus("");
  } catch (error) {
    setStatus(getErrorMessage(error), true);
  } finally {
    askButton.disabled = false;
  }
}

async function getCurrentPlaybackTime() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  if (!tab?.id) throw new Error("Please open a YouTube video first.");

  const [{ result }] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => document.querySelector("video")?.currentTime ?? null
  });
  if (result === null) throw new Error("Could not read the YouTube playback time.");
  return result;
}

async function explainCurrentMoment() {
  explainButton.disabled = true;
  setStatus("Getting an explanation for this moment...");
  try {
    const timestamp = await getCurrentPlaybackTime();
    const response = await fetch(`${API_BASE_URL}/explain-moment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timestamp })
    });
    const data = await response.json();
    if (!response.ok) throwApiError(data, "Could not explain this moment.");

    const momentQuestion = `Explain this moment (${formatTime(timestamp)})`;
    addChatMessage("user", momentQuestion);
    addChatMessage("assistant", data.answer, data.sources);
    chatHistory.push({ role: "User", text: momentQuestion });
    chatHistory.push({ role: "Assistant", text: data.answer });
    setStatus(`Explanation for ${formatTime(timestamp)}.`);
  } catch (error) {
    setStatus(getErrorMessage(error), true);
  } finally {
    explainButton.disabled = false;
  }
}

analyzeButton.addEventListener("click", analyzeVideo);
askButton.addEventListener("click", askQuestion);
explainButton.addEventListener("click", explainCurrentMoment);
clearHistoryButton.addEventListener("click", () => clearChatHistory());
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) askQuestion();
});

loadCurrentVideo();
