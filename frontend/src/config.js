/**
 * Backend API origin.
 * - Local dev (Vite): http://127.0.0.1:8000
 * - Docker (nginx proxy): /api
 */
export const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";
