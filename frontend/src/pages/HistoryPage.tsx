import { useEffect, useState } from 'react';
import { RefreshCw, Search } from 'lucide-react';
import { Link } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { AppShell, Button, Card, EmptyState, ErrorState, PageHero, SeverityBadge, Skeleton, StatusPill } from '../components/DesignSystem';
import { readable, type ResultRow } from '../components/pageShared';

const PAGE_SIZE = 10;
const missingMessage = 'The classification database has not been migrated. Apply the Supabase migrations before using history.';

export function HistoryPage() {
  const [rows, setRows] = useState<ResultRow[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [severity, setSeverity] = useState('');
  const [reviewStatus, setReviewStatus] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [sort, setSort] = useState('created_at-desc');
  const [page, setPage] = useState(0);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState('');
  const [databaseMissing, setDatabaseMissing] = useState(false);

  const load = async () => {
    setBusy(true);
    setError('');
    setDatabaseMissing(false);
    if (!supabase) {
      setDatabaseMissing(true);
      setBusy(false);
      return;
    }
    const [sortColumn, direction] = sort.split('-') as ['created_at' | 'predicted_label' | 'review_status' | 'patient_ref', 'asc' | 'desc'];
    let query = supabase.from('classification_results')
      .select('id,patient_ref,predicted_label,probabilities,review_status,model_version,created_at,assessments(filename,decision,created_at,metrics)', { count: 'exact' })
      .order(sortColumn, { ascending: direction === 'asc' });
    if (search.trim()) {
      const term = search.trim().replace(/[%(),]/g, '');
      query = query.or(`patient_ref.ilike.%${term}%,assessments.filename.ilike.%${term}%`);
    }
    if (severity) query = query.eq('predicted_label', severity);
    if (reviewStatus) query = query.eq('review_status', reviewStatus);
    if (dateFrom) query = query.gte('created_at', `${dateFrom}T00:00:00.000Z`);
    if (dateTo) query = query.lte('created_at', `${dateTo}T23:59:59.999Z`);
    const response = await query.range(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE - 1);
    if (response.error) {
      if (/relation|does not exist|schema cache/i.test(response.error.message)) setDatabaseMissing(true);
      else setError(response.error.message);
      setRows([]);
      setTotal(0);
    } else {
      setRows((response.data ?? []) as unknown as ResultRow[]);
      setTotal(response.count ?? 0);
    }
    setBusy(false);
  };

  useEffect(() => { void load(); }, [page, sort, severity, reviewStatus, dateFrom, dateTo]);
  const submitSearch = (event: React.FormEvent) => { event.preventDefault(); setPage(0); void load(); };
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return <AppShell><PageHero eyebrow="Workspace" title="Patient history" description="Search and review saved classification results. References are pseudonymous only." />
    <Card>
      <form className="toolbar history-toolbar" onSubmit={submitSearch}>
        <div className="search-input"><Search size={16} /><input value={search} onChange={event => setSearch(event.target.value)} placeholder="Search patient reference or filename" aria-label="Search patient reference or filename" /></div>
        <select className="filter-select" value={severity} onChange={event => { setSeverity(event.target.value); setPage(0); }} aria-label="Severity"><option value="">All grades</option>{['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR'].map(value => <option key={value}>{value}</option>)}</select>
        <select className="filter-select" value={reviewStatus} onChange={event => { setReviewStatus(event.target.value); setPage(0); }} aria-label="Review status"><option value="">All review statuses</option><option value="pending">Pending</option><option value="reviewed">Reviewed</option><option value="flagged">Flagged</option></select>
        <input className="date-filter" type="date" value={dateFrom} onChange={event => { setDateFrom(event.target.value); setPage(0); }} aria-label="From date" />
        <input className="date-filter" type="date" value={dateTo} onChange={event => { setDateTo(event.target.value); setPage(0); }} aria-label="To date" />
        <select className="filter-select" value={sort} onChange={event => { setSort(event.target.value); setPage(0); }} aria-label="Sort results"><option value="created_at-desc">Newest first</option><option value="created_at-asc">Oldest first</option><option value="predicted_label-asc">Grade A–Z</option><option value="review_status-asc">Review status</option></select>
        <Button variant="ghost" type="button" onClick={() => void load()}><RefreshCw size={15} /> Refresh</Button>
      </form>
      {busy ? <div className="history-loading">{[1, 2, 3].map(value => <Skeleton key={value} />)}</div> : databaseMissing ? <EmptyState title="Database not migrated" text={missingMessage} /> : error ? <ErrorState text={error} retry={() => void load()} /> : rows.length ? <>
        <div className="table-wrap history-table"><table><thead><tr><th>Patient reference</th><th>Filename</th><th>Date</th><th>Severity</th><th>Review</th><th>Result</th></tr></thead><tbody>{rows.map(row => <tr key={row.id}><td>{row.patient_ref || '—'}</td><td>{row.assessments?.filename || '—'}</td><td>{new Date(row.created_at).toLocaleString()}</td><td><SeverityBadge label={row.predicted_label} /></td><td><StatusPill tone={row.review_status === 'flagged' ? 'danger' : row.review_status === 'reviewed' ? 'success' : 'neutral'}>{readable(row.review_status)}</StatusPill></td><td><Link className="result-link" to={`/results/${encodeURIComponent(row.id)}`}>View result</Link></td></tr>)}</tbody></table></div>
        <div className="history-cards">{rows.map(row => <Link className="history-card" to={`/results/${encodeURIComponent(row.id)}`} key={row.id}><span><strong>{row.patient_ref || 'Unspecified reference'}</strong><small>{row.assessments?.filename || '—'} · {new Date(row.created_at).toLocaleDateString()}</small></span><span><SeverityBadge label={row.predicted_label} /><StatusPill tone={row.review_status === 'flagged' ? 'danger' : row.review_status === 'reviewed' ? 'success' : 'neutral'}>{readable(row.review_status)}</StatusPill></span></Link>)}</div>
        <div className="pagination"><span>{total} exact result{total === 1 ? '' : 's'} · page {page + 1} of {totalPages}</span><span><Button variant="secondary" disabled={page === 0} onClick={() => setPage(value => value - 1)}>Previous</Button><Button variant="secondary" disabled={page + 1 >= totalPages} onClick={() => setPage(value => value + 1)}>Next</Button></span></div>
      </> : <EmptyState title="No matching results" text="Saved classifications will appear here once available." />}
    </Card></AppShell>;
}
