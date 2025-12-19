import { createContext, useContext } from "react";
import { useState, useEffect } from "react";
import { api } from "../lib/axios";

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(null);
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem("token");
    if (stored) {
      setToken(stored);
      api.defaults.headers.common.Authorization = `Bearer ${stored}`;
    }
    setLoading(false);
  }, []);

  const setAuthToken = (tk) => {
    setToken(tk);
    localStorage.setItem("token", tk);
    api.defaults.headers.common.Authorization = `Bearer ${tk}`;
  };

  return (
    <AuthContext.Provider
      value={{ token, loading, setAuthToken, isAuth: !!token }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
