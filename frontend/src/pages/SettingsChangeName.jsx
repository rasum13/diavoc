import { useState, useEffect } from "react";
import Button from "../components/Button";
import Input from "../components/Input";
import { useNavigate } from "react-router-dom";
import { userNameUpdateSchema } from "../schemas/updateUser";
import { api } from "../lib/axios";
import Container from "../components/Container";

const SettingsChangeName = () => {
  const [fullName, setFullName] = useState("");

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const navigate = useNavigate();

  useEffect(() => {
    const getUser = async () => {
      const userData = await api.get("/user/me");
      const data = userData.data;
      setFullName(data.full_name);
    };

    getUser();
  }, []);

  const submit = async (e) => {
    e.preventDefault();

    const result = userNameUpdateSchema.safeParse({
      full_name: fullName,
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
      await api.post("/user/update/name", result.data);
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
        <h2>Change your name</h2>
        <Input
          name="full_name"
          type="text"
          placeholder="Full Name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
          error={errors.full_name}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Updating..." : "Update"}
        </Button>
      </form>
    </Container>
  );
};

export default SettingsChangeName;
