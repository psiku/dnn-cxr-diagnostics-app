import { useEffect, useState } from "react";
import axios from "axios";
import { API_BASE } from "../config";

export function usePathologies() {
  const [pathologies, setPathologies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");

    axios
      .get(`${API_BASE}/pathologies`)
      .then(({ data }) => {
        if (!cancelled) {
          setPathologies(Array.isArray(data?.pathologies) ? data.pathologies : []);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setPathologies([]);
          setError(e?.message || "Failed to load pathologies");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { pathologies, loading, error };
}
