import { z } from "zod";

export const userNameUpdateSchema = z.object({
  full_name: z.string(),
});

export const userEmailUpdateSchema = z.object({
  email: z.email("Invalid email address"),
});

export const userPasswordUpdateSchema = z.object({
  old_password: z.string(),
  new_password: z.string().min(8, "Password must be at least 8 characters"),
});

export const userUpdateInfoSchema = z.object({
  gender: z.boolean(),
  age: z.number().int().positive("Age must be a positive number"),
  height_m: z.number().positive("Height must be a positive number"),
  weight_kg: z.number().positive("Age must be a positive number"),
})
