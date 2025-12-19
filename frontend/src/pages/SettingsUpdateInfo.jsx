import { useState, useEffect } from "react";
import Button from "../components/Button";
import Input from "../components/Input";
import { useNavigate } from "react-router-dom";
import { userUpdateInfoSchema } from "../schemas/updateUser";
import { api } from "../lib/axios";
import ToggleInput from "../components/ToggleInput";
import Container from "../components/Container";

const SettingsUpdateInfo = () => {
  const [gender, setGender] = useState(false);
  const [age, setAge] = useState(0);
  const [heightFeet, setHeightFeet] = useState(0.0);
  const [heightInches, setHeightInches] = useState(0.0);
  const [heightMeter, setHeightMeter] = useState(0.0);
  const [weight, setWeight] = useState(0.0);

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const getUser = async () => {
      const userData = await api.get("/user/me");
      const data = userData.data;
      setGender(data.gender);
      setAge(data.age);

      const heightFt = data.height_m / 0.3048;
      setHeightFeet(Math.floor(heightFt));
      setHeightInches(
        ((heightFt - Math.floor(heightFt)) / (1 / 12)).toFixed(1),
      );

      setWeight(data.weight_kg);
    };

    getUser();
  }, []);

  useEffect(() => {
    setHeightMeter(heightFeet * 0.3048 + heightInches * 0.0254);
  }, [heightFeet, heightInches]);

  const ageOnChange = (e) => {
    setAge(Math.max(0, e.target.value));
  };

  const heightFtOnChange = (e) => {
    setHeightFeet(Math.max(0, e.target.value));
  };

  const heightInchOnChange = (e) => {
    setHeightInches(Math.max(0, e.target.value));
  };

  const weightOnChange = (e) => {
    setWeight(Math.max(0, e.target.value));
  };

  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();

    const result = userUpdateInfoSchema.safeParse({
      gender,
      age,
      height_m: heightMeter,
      weight_kg: weight,
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
      await api.post("/user/update/info", result.data);
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
      <h2 className="mb-4">Update your info</h2>
      <form className="grid grid-rows-3 gap-4" onSubmit={submit}>
        <ToggleInput
          label="Gender"
          trueValue="Male"
          currentValue={gender}
          falseValue="Female"
          onChange={(value) => {
            setGender(value);
          }}
          error={errors.gender}
        />
        <Input
          name="age"
          type="number"
          value={age}
          onChange={ageOnChange}
          error={errors.age}
          label="Age"
        />
        <Input
          name="height_ft"
          type="number"
          value={heightFeet}
          onChange={heightFtOnChange}
          error={errors.height_m}
          label="Height (feet)"
        />
        <Input
          name="height_in"
          type="number"
          value={heightInches}
          onChange={heightInchOnChange}
          error={errors.height_m}
          label="Height (inches)"
          step={0.5}
        />
        <Input
          name="weight_kg"
          type="number"
          value={weight}
          onChange={weightOnChange}
          error={errors.weight_kg}
          label="Weight (kg)"
          step={0.01}
        />
        <Button type="submit" disabled={isSubmitting}>
          {isSubmitting ? "Updating..." : "Update"}
        </Button>
      </form>
    </Container>
  );
};

export default SettingsUpdateInfo;
