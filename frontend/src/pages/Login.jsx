import { Link, useNavigate } from "react-router-dom";
import Button from "../components/Button";
import Input from "../components/Input";
import { api } from "../lib/axios";
import { useState } from "react";
import { loginSchema } from "../schemas/login";
import { useAuth } from "../context/AuthContext";

const Login = () => {
  const { setAuthToken } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();

    const result = loginSchema.safeParse({
      email,
      password,
    });

    if (!result.success) {
      const fieldErrors = {};
      for (const [key, message] of Object.entries(
        result.error.flatten().fieldErrors,
      )) {
        fieldErrors[key] = message;
      }
      setErrors(fieldErrors);
      return;
    }

    try {
      setIsSubmitting(true);
      const res = await api.post("/auth/login", result.data);
      setAuthToken(res.data.token);
      navigate("/");
    } catch (err) {
      const detail = err.response?.data?.detail;

      if (Array.isArray(detail)) {
        detail.forEach((e) => {
          setErrors((prev) => ({ ...prev, [e.loc.at(-1)]: e.msg }));
        });
      } else if (typeof detail === "string") {
        setErrors({ _form: detail });
      } else {
        setErrors({ _form: "Unexpected error" });
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="flex flex-col justify-center items-center text-center h-screen bg-linear-to-br to-primary/30 from-white">
      <div className="px-16 py-8 flex flex-col w-lg bg-neutral-50 rounded-xl shadow-[0_8px_32px] shadow-neutral-100">
        <h1 className="mb-4">Login</h1>
        <form className="grid grid-rows-3 gap-4" onSubmit={submit}>
          <Input
            name="email"
            type="text"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            error={errors.email}
          />
          <Input
            name="password"
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            error={errors.password}
          />
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Logging in..." : "Log In"}
          </Button>
        </form>
        <div className="m-2">Dont have an account? <Link className="text-primary hover:text-primary-dark" to="/signup">Create one</Link></div>
      </div>
    </div>
  );
};

export default Login;
