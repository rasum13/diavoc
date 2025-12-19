import Input from "../components/Input";
import Button from "../components/Button";
import { signupSchema } from "../schemas/signup";
import { useEffect, useState } from "react";
import { api } from "../lib/axios";
import { useNavigate } from "react-router-dom";
import ToggleInput from "../components/ToggleInput";

const Signup = () => {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [gender, setGender] = useState(false);
  const [age, setAge] = useState(0);
  const [heightFeet, setHeightFeet] = useState(0.0);
  const [heightInches, setHeightInches] = useState(0.0);
  const [heightMeter, setHeightMeter] = useState(0.0);
  const [weight, setWeight] = useState(0.0);
  const [isAsian, setIsAsian] = useState(false);

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    setHeightMeter(heightFeet * 0.3048 + heightInches * 0.025400);
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

    const result = signupSchema.safeParse({
      full_name: fullName,
      email,
      password,
      gender,
      age,
      height_m: heightMeter,
      weight_kg: weight,
      is_asian: isAsian,
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
      await api.post("/auth/signup", result.data);
      navigate("/login");
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
    <div className="flex flex-col justify-center items-center text-center h-screen bg-primary-light">
      <div className="px-16 py-8 flex flex-col w-lg bg-neutral-50 rounded-md shadow-[0_8px_32px] shadow-neutral-200">
        <h1 className="mb-4">Register</h1>
        <form className="grid grid-rows-3 gap-4" onSubmit={submit}>
          <Input
            name="full_name"
            type="text"
            placeholder="Full Name"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            error={errors.full_name}
          />
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
          <ToggleInput
            label="Gender"
            trueValue="Male"
            falseValue="Female"
            currentValue={gender}
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
          <ToggleInput
            label="Are you Asian?"
            trueValue="Yes"
            currentValue={isAsian}
            falseValue="No"
            onChange={(value) => {
              setIsAsian(value);
            }}
            error={errors.gender}
          />
          <Button type="submit" disabled={isSubmitting}>
            {isSubmitting ? "Signing up..." : "Register"}
          </Button>
        </form>
      </div>
    </div>
  );
};

export default Signup;
