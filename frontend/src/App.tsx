import { useEffect, useState } from 'react';
import axios from 'axios';
const api = axios.create({ baseURL: 'http://localhost:8001' });

export default function App() {
  const [headline, setHeadline] = useState(''); const [body, setBody] = useState(''); const [model, setModel] = useState('distilbert');
  const [result, setResult] = useState<any>(); const [error, setError] = useState(''); const [history, setHistory] = useState<any[]>([]);
  const refresh = async () => { try { setHistory((await api.get('/history')).data); } catch {} };
  useEffect(() => { void refresh(); }, []);
  const analyze = async () => { setError(''); try { const { data } = await api.post('/predict', { headline, body, model }); setResult(data); void refresh(); } catch (e: any) { setError(e.response?.data?.detail || 'Analysis failed.'); } };
  return <main><header><span>VERITAS</span><h1>Fake News Detection Lab</h1><p>Evidence-oriented article classification with reproducible trained local models.</p></header>
    <section className="grid"><article className="panel"><h2>Analyze article</h2><label>Model<select value={model} onChange={e => setModel(e.target.value)}><option value="distilbert">DistilBERT baseline</option><option value="bert">BERT baseline</option><option value="roberta">RoBERTa baseline</option></select></label><label>Headline<input value={headline} onChange={e => setHeadline(e.target.value)} placeholder="Optional headline" /></label><label>Article body<textarea value={body} onChange={e => setBody(e.target.value)} placeholder="Paste at least 20 characters of article text" /></label><button onClick={analyze} disabled={body.length < 20}>Analyze article</button>{error && <p className="error">{error}</p>}</article>
      <article className="panel"><h2>Result</h2>{result ? <><strong className={result.prediction === 'FAKE' ? 'fake' : 'real'}>{result.prediction}</strong><p>{(result.confidence * 100).toFixed(1)}% confidence · {result.inference_ms} ms · {result.tokens} words</p><h3>Why this baseline predicted it</h3><p className="muted">{result.explanation_notice}</p><div className="tokens">{result.explanation.map((x: any) => <span className={x.contribution > 0 ? 'positive' : 'negative'} key={x.token}>{x.token} ({x.contribution.toFixed(2)})</span>)}</div></> : <p className="muted">Train or install a versioned model asset, then submit an article.</p>}</article></section>
    <section className="panel"><h2>Recent predictions</h2><table><thead><tr><th>Headline</th><th>Result</th><th>Confidence</th><th>Model</th></tr></thead><tbody>{history.map(x => <tr key={x.id}><td>{x.headline || 'Untitled article'}</td><td>{x.prediction}</td><td>{(x.confidence * 100).toFixed(1)}%</td><td>{x.model}</td></tr>)}</tbody></table></section></main>;
}
