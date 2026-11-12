import { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { api, Analytics, ModelMetrics, Ticket } from './lib/api';

const PRIORITY_COLORS: Record<string, string> = {
  critical: '#dc5353',
  high: '#e49a43',
  medium: '#527fce',
  low: '#3b9d82',
};

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function StatCard({ label, value, hint }: { label: string; value: string | number; hint: string }) {
  return (
    <article className="stat-card">
      <span className="eyebrow">{label}</span>
      <strong>{value}</strong>
      <span className="stat-hint">{hint}</span>
    </article>
  );
}

function Tag({ value, kind }: { value: string; kind: 'priority' | 'sentiment' | 'category' }) {
  return <span className={`tag tag-${kind} ${kind === 'priority' ? `priority-${value}` : ''}`}>{value}</span>;
}

function TicketRow({ ticket }: { ticket: Ticket }) {
  return (
    <div className="ticket-row">
      <div className="ticket-main">
        <strong>{ticket.subject}</strong>
        <p>{ticket.description}</p>
        <span className="ticket-date">{new Date(ticket.created_at).toLocaleString()}</span>
      </div>
      <div className="ticket-tags">
        <Tag value={ticket.category} kind="category" />
        <Tag value={ticket.priority} kind="priority" />
        <Tag value={ticket.sentiment} kind="sentiment" />
      </div>
    </div>
  );
}

export default function App() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [subject, setSubject] = useState('');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [latestPrediction, setLatestPrediction] = useState<Ticket | null>(null);

  const loadDashboard = useCallback(async () => {
    const [analyticsResult, metricsResult] = await Promise.all([
      api.getAnalytics(),
      api.getMetrics(),
    ]);
    setAnalytics(analyticsResult);
    setMetrics(metricsResult);
  }, []);

  useEffect(() => {
    loadDashboard()
      .catch((err: unknown) => setError(err instanceof Error ? err.message : 'Could not load dashboard.'))
      .finally(() => setLoading(false));
  }, [loadDashboard]);

  const averageModelScore = useMemo(() => {
    if (!metrics?.available || !metrics.metrics) return '—';
    const category = metrics.metrics.category_model.macro_f1;
    const priority = metrics.metrics.priority_model.macro_f1;
    return `${((category + priority) / 2 * 100).toFixed(1)}%`;
  }, [metrics]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError('');
    setSubmitting(true);
    try {
      const created = await api.createTicket({ subject: subject.trim(), description: description.trim() });
      setLatestPrediction(created);
      setSubject('');
      setDescription('');
      await loadDashboard();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Ticket analysis failed.');
    } finally {
      setSubmitting(false);
    }
  }

  const priorityData = analytics?.priority_counts || [];
  const categoryData = analytics?.category_counts || [];

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="SupportIQ home">
          <span className="brand-mark">S</span>
          <span>Support<span className="brand-accent">IQ</span></span>
        </a>
        <div className="topbar-right">
          <span className="status-dot" />
          <span>ML workspace</span>
          <span className="topbar-pill">DEMO</span>
        </div>
      </header>

      <main id="top" className="content">
        <section className="hero">
          <div>
            <span className="eyebrow hero-eyebrow">CUSTOMER SUPPORT INTELLIGENCE</span>
            <h1>Understand every ticket.<br /><span>Prioritize what matters.</span></h1>
            <p>Classify incoming support requests, estimate urgency, and identify customer sentiment using machine learning.</p>
          </div>
          <div className="hero-chip"><span className="hero-chip-icon">✦</span><span>ML-assisted triage</span></div>
        </section>

        {error && <div className="error-banner" role="alert">{error}</div>}

        <section className="stats-grid" aria-label="Ticket statistics">
          <StatCard label="TOTAL ANALYZED TICKETS" value={loading ? '…' : analytics?.total_tickets ?? 0} hint="Tickets submitted in this workspace" />
          <StatCard label="CATEGORIES DETECTED" value={analytics?.category_counts.length ?? 0} hint="Distinct ticket categories" />
          <StatCard label="HIGH / CRITICAL" value={(analytics?.priority_counts || []).filter((x) => x.name === 'high' || x.name === 'critical').reduce((sum, x) => sum + x.count, 0)} hint="Tickets requiring closer attention" />
          <StatCard label="AVERAGE MODEL MACRO F1" value={averageModelScore} hint="Average of the two held-out test scores" />
        </section>

        <section className="workspace-grid">
          <article className="panel ticket-form-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">LIVE INFERENCE</span>
                <h2>Analyze a support ticket</h2>
              </div>
              <span className="step-badge">01 / INPUT</span>
            </div>
            <form onSubmit={handleSubmit} className="ticket-form">
              <label htmlFor="subject">Ticket subject</label>
              <input
                id="subject"
                value={subject}
                onChange={(event) => setSubject(event.target.value)}
                placeholder="e.g. Payment was charged twice"
                minLength={3}
                maxLength={180}
                required
              />
              <label htmlFor="description">Customer message</label>
              <textarea
                id="description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                placeholder="Describe the issue in the customer's own words..."
                minLength={8}
                maxLength={4000}
                rows={5}
                required
              />
              <button className="primary-button" type="submit" disabled={submitting}>
                {submitting ? 'Analyzing ticket…' : 'Analyze ticket'}
                <span aria-hidden="true">↗</span>
              </button>
              <p className="form-footnote">First analysis may take longer while the pretrained sentiment model downloads.</p>
            </form>
          </article>

          <article className="panel result-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">MODEL OUTPUT</span>
                <h2>Prediction summary</h2>
              </div>
              <span className="step-badge">02 / RESULT</span>
            </div>
            {latestPrediction ? (
              <div className="prediction-results">
                <p className="result-subject">{latestPrediction.subject}</p>
                <div className="prediction-item">
                  <div><span>Predicted category</span><strong><Tag value={latestPrediction.category} kind="category" /></strong></div>
                  <b>{percent(latestPrediction.category_confidence)}</b>
                </div>
                <div className="prediction-item">
                  <div><span>Predicted priority</span><strong><Tag value={latestPrediction.priority} kind="priority" /></strong></div>
                  <b>{percent(latestPrediction.priority_confidence)}</b>
                </div>
                <div className="prediction-item">
                  <div><span>Customer sentiment</span><strong><Tag value={latestPrediction.sentiment} kind="sentiment" /></strong></div>
                  <b>{percent(latestPrediction.sentiment_confidence)}</b>
                </div>
                <p className="result-note">Confidence values are model scores, not calibrated guarantees. Review predictions before taking action.</p>
              </div>
            ) : (
              <div className="empty-result">
                <div className="empty-icon">✳</div>
                <strong>Your analysis will appear here</strong>
                <p>Submit a customer ticket to see its predicted category, priority, and sentiment.</p>
              </div>
            )}
          </article>
        </section>

        <section className="charts-grid">
          <article className="panel chart-panel">
            <div className="panel-heading">
              <div><span className="eyebrow">DISTRIBUTION</span><h2>Ticket categories</h2></div>
            </div>
            {categoryData.length ? (
              <ResponsiveContainer width="100%" height={245}>
                <BarChart data={categoryData} margin={{ top: 8, right: 8, left: -18, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e8edf3" />
                  <XAxis dataKey="name" tickLine={false} axisLine={false} tick={{ fill: '#7b8797', fontSize: 11 }} />
                  <YAxis allowDecimals={false} tickLine={false} axisLine={false} tick={{ fill: '#7b8797', fontSize: 11 }} />
                  <Tooltip cursor={{ fill: '#f3f6fa' }} />
                  <Bar dataKey="count" fill="#527fce" radius={[6, 6, 0, 0]} maxBarSize={44} />
                </BarChart>
              </ResponsiveContainer>
            ) : <div className="chart-empty">No analyzed tickets yet.</div>}
          </article>

          <article className="panel chart-panel">
            <div className="panel-heading">
              <div><span className="eyebrow">URGENCY MIX</span><h2>Priority breakdown</h2></div>
            </div>
            {priorityData.length ? (
              <div className="priority-chart-wrap">
                <ResponsiveContainer width="58%" height={245}>
                  <PieChart>
                    <Pie data={priorityData} dataKey="count" nameKey="name" innerRadius={58} outerRadius={88} paddingAngle={3}>
                      {priorityData.map((entry) => <Cell key={entry.name} fill={PRIORITY_COLORS[entry.name] || '#8c9bad'} />)}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="legend-list">
                  {priorityData.map((entry) => (
                    <div key={entry.name}><span className="legend-dot" style={{ background: PRIORITY_COLORS[entry.name] || '#8c9bad' }} /><span>{entry.name}</span><strong>{entry.count}</strong></div>
                  ))}
                </div>
              </div>
            ) : <div className="chart-empty">No analyzed tickets yet.</div>}
          </article>
        </section>

        <section className="panel model-panel">
          <div className="panel-heading model-heading">
            <div><span className="eyebrow">MODEL EVALUATION</span><h2>Held-out test performance</h2></div>
            <span className="evaluation-caption">Metrics loaded from the training report</span>
          </div>
          {metrics?.available && metrics.metrics ? (
            <div className="model-metrics-grid">
              <div className="model-metric"><span>Category · Logistic Regression</span><strong>{percent(metrics.metrics.category_model.macro_f1)}</strong><small>Macro F1</small></div>
              <div className="model-metric"><span>Priority · XGBoost</span><strong>{percent(metrics.metrics.priority_model.macro_f1)}</strong><small>Macro F1</small></div>
              <div className="model-metric"><span>Dataset rows</span><strong>{metrics.metrics.dataset_rows_after_cleaning}</strong><small>After cleaning</small></div>
              <div className="model-metric"><span>Test examples</span><strong>{metrics.metrics.test_rows}</strong><small>Held-out split</small></div>
            </div>
          ) : (
            <div className="metrics-placeholder">Train the models with <code>python backend/scripts/train.py</code> to display evaluation metrics.</div>
          )}
          <p className="disclaimer">Evaluation scores depend on the dataset. The included generated dataset is only a pipeline demo; do not present its scores as real-world performance.</p>
        </section>

        <section className="panel recent-panel">
          <div className="panel-heading">
            <div><span className="eyebrow">WORKSPACE ACTIVITY</span><h2>Recent tickets</h2></div>
            <span className="record-count">{analytics?.recent_tickets.length ?? 0} latest records</span>
          </div>
          {analytics?.recent_tickets.length ? (
            <div className="ticket-list">{analytics.recent_tickets.map((ticket) => <TicketRow key={ticket.id} ticket={ticket} />)}</div>
          ) : <div className="chart-empty">Tickets you analyze will be listed here.</div>}
        </section>

        <footer className="footer"><span>SupportIQ</span><span>ML predictions are decision support, not automatic decisions.</span></footer>
      </main>
    </div>
  );
}
