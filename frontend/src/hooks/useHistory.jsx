import { useEffect, useState } from "react";
import { api } from "../lib/axios";

const useHistory = ( limit = null ) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const res = await api.get(`/history${limit == null ? "" : "?limit=" + limit}`);
        // setHistory(res.data);
        const newHistory = res.data.slice()
        newHistory.sort((a, b) => {
          return a.date > b.date;
        })
        setHistory(newHistory);
      } catch (err) {
        setError(err);
        setLoading(false);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  return [ history, loading, error ];
};

export default useHistory;
