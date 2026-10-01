import type { Metadata } from 'next';
import { Shell } from '../components/shell';
import './globals.css';
export const metadata: Metadata = { title: 'RAGEval · Research workspace', description: 'Inspect retrieval, generation and evaluation results.' };
export default function Layout({children}: Readonly<{children: React.ReactNode}>) { return <html lang="en"><body><Shell>{children}</Shell></body></html>; }
