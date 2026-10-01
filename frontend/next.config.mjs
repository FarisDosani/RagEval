const base = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
const config = {
  async rewrites() { return [{ source: '/backend/:path*', destination: `${base}/:path*` }]; },
};
export default config;
