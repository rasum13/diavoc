import { useState, useEffect } from "react";
import Button from "../components/Button";
import Input from "../components/Input";
import { useNavigate } from "react-router-dom";
import { userEmailUpdateSchema } from "../schemas/updateUser";
import { api } from "../lib/axios";
import Container from "../components/Container";

const SettingsChangeEmail = () => {
  const [email, setEmail] = useState("");

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    const getUser = async () => {
      const userData = await api.get("/user/me");
      const data = userData.data;
      setEmail(data.email);
    };

    getUser();
  }, []);

  const submit = async (e) => {
    e.preventDefault();

    const result = userEmailUpdateSchema.safeParse({
      email,
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
      await api.post("/user/update/email", result.data);
      navigate("/settings");
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
    <Container>
      <form className="grid grid-rows-3 gap-4" onSubmit={submit}>
        <h2>Change your email</h2>
        <Input
          name="email"
          type="text"
          placeholder="Update Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          error={errors.email}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Updating..." : "Update"}
        </Button>
      </form>
    </Container>
  );
};

export default SettingsChangeEmail;
