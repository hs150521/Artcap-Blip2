import { Message } from "@/types";
import type { NextApiRequest, NextApiResponse } from "next";

export const config = {
	runtime: "nodejs"
};

const handler = async (
	req: NextApiRequest,
	res: NextApiResponse
): Promise<void> => {
	try {
		if (req.method !== "POST") {
			res.status(405).json({ error: "Method not allowed" });
			return;
		}

		const { messages } = req.body as {
			messages: Message[];
		};

		console.log("Received request with", messages.length, "messages");

		// Get the last user message (should contain image and prompt)
		const lastMessage = messages[messages.length - 1];

		if (!lastMessage || lastMessage.role !== "user") {
			console.error("No user message found");
			res.status(400).json({ error: "No user message found" });
			return;
		}

		if (!lastMessage.image) {
			console.error("No image provided");
			res.status(400).json({ error: "No image provided" });
			return;
		}

		// The content is already formatted as "Question: ... Short answer:" by the frontend
		const prompt = lastMessage.content || "Question: Describe this image. Short answer:";
		console.log("Calling FastAPI with prompt:", prompt);

		// Call FastAPI service
		const apiUrl = process.env.BLIP2_API_URL || "http://localhost:8001";
		const response = await fetch(`${apiUrl}/api/generate`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({
				image: lastMessage.image,
				prompt: prompt,
			}),
		});

		console.log("FastAPI response status:", response.status);

		if (!response.ok) {
			const errorText = await response.text();
			console.error("FastAPI error:", errorText);
			res.status(response.status).json({ error: errorText });
			return;
		}

		const data = await response.json();
		const caption = data.caption || "";
		console.log("Generated caption length:", caption.length);

		// Return as stream for consistency with original API
		res.setHeader("Content-Type", "text/plain; charset=utf-8");
		res.status(200).send(caption);
	} catch (error) {
		console.error("Error in API handler:", error);
		res.status(500).json({ error: String(error) });
	}
};

export default handler;
