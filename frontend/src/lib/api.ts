const getApiBaseUrl = (): string => {
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  if (typeof window !== 'undefined') {
    // Vite local dev server runs on port 5173
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return window.location.port === '5173' ? 'http://localhost:8000' : '';
    }
    // GitHub Pages mirror connects to the live Render deployment
    if (window.location.hostname.endsWith('github.io')) {
      return 'https://supportiq-rndw.onrender.com';
    }
    // Render or unified production container: backend & frontend share origin
    return '';
  }
  return 'http://localhost:8000';
};

const API_BASE_URL = getApiBaseUrl();

export type Ticket = {
  id: number;
  subject: string;
  description: string;
  category: string;
  category_confidence: number;
  priority: string;
  priority_confidence: number;
  sentiment: string;
  sentiment_confidence: number;
  created_at: string;
};

export type CountItem = { name: string; count: number };

export type Analytics = {
  total_tickets: number;
  category_counts: CountItem[];
  priority_counts: CountItem[];
  sentiment_counts: CountItem[];
  recent_tickets: Ticket[];
};

export type ModelMetrics = {
  available: boolean;
  metrics: null | {
    dataset_rows_after_cleaning: number;
    training_rows: number;
    test_rows: number;
    category_model: { accuracy: number; macro_f1: number; weighted_f1: number };
    priority_model: { accuracy: number; macro_f1: number; weighted_f1: number };
    important_note: string;
  };
};

export type TicketInput = { subject: string; description: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {}),
    },
  });

  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const api = {
  getAnalytics: () => request<Analytics>('/api/analytics'),
  getMetrics: () => request<ModelMetrics>('/api/model-metrics'),
  createTicket: (ticket: TicketInput) =>
    request<Ticket>('/api/tickets', {
      method: 'POST',
      body: JSON.stringify(ticket),
    }),
};
