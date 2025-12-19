import { useState } from "react";
import Button from "../components/Button";
import Input from "../components/Input";
import { useNavigate } from "react-router-dom";
import { userPasswordUpdateSchema } from "../schemas/updateUser";
import { api } from "../lib/axios";
import Container from "../components/Container";

const SettingsChangePassword = () => {
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();

    const result = userPasswordUpdateSchema.safeParse({
      old_password: oldPassword,
      new_password: newPassword,
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
      await api.post("/user/update/password", result.data);
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
      <form className="grid grid-rows-5 gap-4" onSubmit={submit}>
        <h2>Change your password</h2>
        <p>Password must be of at least 8 characters</p>
        <Input
          name="old_password"
          type="password"
          placeholder="Old Password"
          value={oldPassword}
          onChange={(e) => setOldPassword(e.target.value)}
          error={errors.new_password}
        />
        <Input
          name="new_password"
          type="password"
          placeholder="New Password"
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
          error={errors.old_password}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Updating..." : "Update"}
        </Button>
      </form>
    </Container>
  );
};

export default SettingsChangePassword;
