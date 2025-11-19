import { Chat } from "@/components/Chat/Chat";
import { Footer } from "@/components/Layout/Footer";
import { Navbar } from "@/components/Layout/Navbar";
import { Message } from "@/types";
import Head from "next/head";
import { useEffect, useRef, useState } from "react";

export default function Home() {
	const [messages, setMessages] = useState<Message[]>([]);
	const [loading, setLoading] = useState<boolean>(false);
	const [lastUploadedImage, setLastUploadedImage] = useState<string | null>(null);

	const messagesEndRef = useRef<HTMLDivElement>(null);

	const scrollToBottom = () => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	};

	const handleSend = async (message: Message) => {
		// Find the most recent image from message history if current message doesn't have one
		let imageToUse = message.image;
		if (!imageToUse) {
			// Look for the most recent user message with an image
			for (let i = messages.length - 1; i >= 0; i--) {
				if (messages[i].role === "user" && messages[i].image) {
					imageToUse = messages[i].image;
					break;
				}
			}
		}

		// Update last uploaded image if message contains an image
		if (message.image) {
			setLastUploadedImage(message.image);
		} else if (imageToUse) {
			// Also update if we're using a previous image
			setLastUploadedImage(imageToUse);
		}

		// Create message with image (either new or from history)
		const messageToSend: Message = {
			...message,
			image: imageToUse
		};

		const updatedMessages = [...messages, messageToSend];

		setMessages(updatedMessages);
		setLoading(true);

		try {
			const response = await fetch("/api/chat", {
				method: "POST",
				headers: {
					"Content-Type": "application/json"
				},
				body: JSON.stringify({
					messages: updatedMessages
				})
			});

			if (!response.ok) {
				setLoading(false);
				let errorMessage = response.statusText;
				try {
					const errorData = await response.json();
					errorMessage = errorData.error || errorMessage;
				} catch (e) {
					const errorText = await response.text();
					errorMessage = errorText || errorMessage;
				}

				setMessages((messages) => [
					...messages,
					{
						role: "assistant",
						content: `Error: ${errorMessage}`
					}
				]);
				return;
			}

			const caption = await response.text();
			setLoading(false);

			setMessages((messages) => [
				...messages,
				{
					role: "assistant",
					content: caption
				}
			]);
		} catch (error) {
			setLoading(false);
			const errorMessage = error instanceof Error ? error.message : String(error);
			setMessages((messages) => [
				...messages,
				{
					role: "assistant",
					content: `Error: ${errorMessage}`
				}
			]);
		}
	};

	const handleReset = () => {
		setMessages([
			{
				role: "assistant",
				content: `Hi there! I'm an AI assistant powered by BLIP-2. Upload an image and provide a prompt, and I'll generate a description for you.`
			}
		]);
		setLastUploadedImage(null);
	};

	useEffect(() => {
		scrollToBottom();
	}, [messages]);

	useEffect(() => {
		setMessages([
			{
				role: "assistant",
				content: `Hi there! I'm an AI assistant powered by BLIP-2. Upload an image and provide a prompt, and I'll generate a description for you.`
			}
		]);
	}, []);

	return (
		<>
			<Head>
				<title>BLIP-2 Image Captioning</title>
				<meta
					name="description"
					content="BLIP-2 Image Captioning - Upload images and generate descriptions using AI."
				/>
				<meta
					name="viewport"
					content="width=device-width, initial-scale=1"
				/>
				<link
					rel="icon"
					href="/favicon.ico"
				/>
			</Head>

			<div className="flex flex-col h-screen">
				<Navbar />

				<div className="flex-1 overflow-auto sm:px-10 pb-4 sm:pb-10">
					<div className="max-w-[800px] mx-auto mt-4 sm:mt-12">
						<Chat
							messages={messages}
							loading={loading}
							onSend={handleSend}
							onReset={handleReset}
							lastUploadedImage={lastUploadedImage}
						/>
						<div ref={messagesEndRef} />
					</div>
				</div>
				<Footer />
			</div>
		</>
	);
}
