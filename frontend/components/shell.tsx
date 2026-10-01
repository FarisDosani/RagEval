'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';
import { api } from '../lib/api';
const links = [['/', 'Overview'], ['/query','Query'], ['/experiments','Experiments'], ['/comparisons','Comparisons'], ['/sweeps','Sweeps'], ['/failure-analysis','Failure Analysis']];
export function Shell({children}: {children: React.ReactNode}) {
 const path = usePathname(); const [status,setStatus] = useState('Checking API');
 useEffect(() => { let active = true; api.health().then(() => {if(active) setStatus('API reachable');}).catch(() => {if(active) setStatus('API unavailable');}); return () => {active=false;}; }, []);
 return <div className="shell"><aside><Link className="brand" href="/">RAG<span>Eval</span><small>RESEARCH WORKSPACE</small></Link><div className="nav-label">WORKSPACE</div><nav aria-label="Main navigation">{links.map(([href,label],i) => <Link key={href} href={href} aria-current={path===href?'page':undefined}><span className="nav-index">0{i+1}</span>{label}</Link>)}</nav><div className="sidebar-note">Controlled experiments.<br/>Traceable evidence.<br/>Comparable results.</div><div className="api-status"><i className={status==='API reachable'?'online':''}/>{status}</div></aside><main><div className="topbar"><span>RAGEval / Research</span><span className="tag">LOCAL WORKSPACE</span></div>{children}<footer>RAGEval · Results reflect recorded evaluations. Missing values are never estimated.</footer></main></div>;
}
