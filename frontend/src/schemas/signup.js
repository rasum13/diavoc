import { z } from "zod";

export const signupSchema = z.object({
  full_name: z.string().min(1, "Full name is required"),
  email: z.email("Invalid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
  gender: z.boolean(),
  age: z.number().int().positive("Age must be a positive number"),
  height_m: z.number().positive("Height must be a positive number"),
  weight_kg: z.number().positive("Age must be a positive number"),
  is_asian: z.boolean()
});
