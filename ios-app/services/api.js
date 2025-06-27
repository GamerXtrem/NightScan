export const BASE_URL = 'http://localhost:8000';
export const PREDICT_API_URL =
  process.env.PREDICT_API_URL || 'http://localhost:8001/api/predict';

export async function login(username, password) {
  const resp = await fetch(`${BASE_URL}/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
    credentials: 'include',
  });
  if (!resp.ok) {
    throw new Error('Login failed');
  }
}

export async function register(username, password) {
  const resp = await fetch(`${BASE_URL}/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: `username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`,
    credentials: 'include',
  });
  if (!resp.ok) {
    throw new Error('Registration failed');
  }
}

export async function fetchDetections() {
  const resp = await fetch(`${BASE_URL}/api/detections`, {
    credentials: 'include',
  });
  if (!resp.ok) {
    throw new Error('Failed to fetch detections');
  }
  return resp.json();
}

export async function uploadMedia(uri, mimeType = 'application/octet-stream') {
  const form = new FormData();
  form.append('file', {
    uri,
    type: mimeType,
    name: uri.split('/').pop() || 'upload',
  });

  const resp = await fetch(PREDICT_API_URL, {
    method: 'POST',
    body: form,
    credentials: 'include',
  });

  if (!resp.ok) {
    throw new Error('Upload failed');
  }

  return resp.json();
}
