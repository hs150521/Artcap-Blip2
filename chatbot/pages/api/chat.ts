import { Message } from "@/types";
import type { NextApiRequest, NextApiResponse } from "next";

export const config = {
	runtime: "nodejs",
	api: {
		bodyParser: {
			sizeLimit: '50mb',
		},
	},
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

		// Call both FastAPI services in parallel
		const apiUrl = process.env.BLIP2_API_URL || "http://localhost:8001";

		const [blip2Response, kvResponse] = await Promise.allSettled([
			fetch(`${apiUrl}/api/generate`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					image: lastMessage.image,
					prompt: prompt,
				}),
			}),
			fetch(`${apiUrl}/api/generate-kv`, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify({
					image: lastMessage.image,
					prompt: prompt,
				}),
			}),
		]);

		// Process BLIP-2 response
		let blip2Caption = "";
		if (blip2Response.status === "fulfilled" && blip2Response.value.ok) {
			const blip2Data = await blip2Response.value.json();
			blip2Caption = blip2Data.caption || "";
		} else {
			const error = blip2Response.status === "fulfilled"
				? await blip2Response.value.text().catch(() => "Unknown error")
				: blip2Response.reason?.message || "Failed to call BLIP-2";
			blip2Caption = `Error: ${error}`;
		}

		// Process KV response
		let kvCaption = "";
		if (kvResponse.status === "fulfilled" && kvResponse.value.ok) {
			const kvData = await kvResponse.value.json();
			kvCaption = kvData.caption || "";
		} else {
			const error = kvResponse.status === "fulfilled"
				? await kvResponse.value.text().catch(() => "Unknown error")
				: kvResponse.reason?.message || "Failed to call KV model";
			kvCaption = `Error: ${error}`;
		}

		console.log("BLIP-2 caption length:", blip2Caption.length);
		console.log("KV caption length:", kvCaption.length);

		// Return both captions as JSON
		res.setHeader("Content-Type", "application/json");
		res.status(200).json({
			blip2: blip2Caption,
			kv: kvCaption
		});
	} catch (error) {
		console.error("Error in API handler:", error);
		res.status(500).json({ error: String(error) });
	}
};

export default handler;
