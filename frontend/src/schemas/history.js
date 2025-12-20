import { z } from "zod";

export const historyItemCreateSchema = z.object({
  score: z.number(),
})
