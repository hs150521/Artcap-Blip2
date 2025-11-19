import { Message } from "@/types";
import { IconArrowUp } from "@tabler/icons-react";
import { FC, KeyboardEvent, useEffect, useRef, useState } from "react";
import { ImageUpload } from "./ImageUpload";

interface Props {
	onSend: (message: Message) => void;
	lastUploadedImage: string | null;
}

export const ChatInput: FC<Props> = ({ onSend, lastUploadedImage }) => {
	const [content, setContent] = useState<string>();
	const [image, setImage] = useState<string | null>(null);

	const textareaRef = useRef<HTMLTextAreaElement>(null);

	const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
		const value = e.target.value;
		if (value.length > 4000) {
			alert("Message limit is 4000 characters");
			return;
		}

		setContent(value);
	};

	// Simple English detection: check if text contains mostly English characters
	const isEnglish = (text: string): boolean => {
		if (!text || text.trim().length === 0) return true; // Empty text is allowed

		// Remove whitespace and common punctuation
		const cleaned = text.replace(/[\s\.,!?;:'"()\[\]{}\-_=+*&^%$#@~`]/g, '');
		if (cleaned.length === 0) return true; // Only punctuation/whitespace

		// Check if text contains mostly ASCII characters (English)
		// Allow common English characters: a-z, A-Z, 0-9, and some common symbols
		const englishPattern = /^[a-zA-Z0-9\s\.,!?;:'"()\[\]{}\-_=+*&^%$#@~`]+$/;

		// Check if at least 80% of non-whitespace characters are English
		const nonWhitespace = cleaned.length;
		const englishChars = cleaned.match(/[a-zA-Z0-9]/g)?.length || 0;

		return englishPattern.test(text) && (nonWhitespace === 0 || englishChars / nonWhitespace >= 0.5);
	};

	const handleSend = () => {
		// Allow sending if there's content, or image, or lastUploadedImage exists
		if (!content && !image && !lastUploadedImage) {
			alert("Please enter a message or upload an image");
			return;
		}

		// Check if input is English (only check if there's text content)
		if (content && content.trim().length > 0 && !isEnglish(content)) {
			alert("Please enter your question in English.");
			return;
		}

		// Format the prompt: "Question: {content} Short answer:"
		let formattedContent = content || "";
		if (formattedContent.trim().length > 0) {
			formattedContent = `Question: ${formattedContent.trim()} Short answer:`;
		}

		// Use new image if provided, otherwise use lastUploadedImage
		const imageToSend = image || lastUploadedImage || undefined;

		onSend({
			role: "user",
			content: formattedContent,
			image: imageToSend
		});
		setContent("");
		setImage(null); // Clear new image, but lastUploadedImage persists
	};

	const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	};

	useEffect(() => {
		if (textareaRef && textareaRef.current) {
			textareaRef.current.style.height = "inherit";
			textareaRef.current.style.height = `${textareaRef.current?.scrollHeight}px`;
		}
	}, [content]);

	return (
		<div>
			<ImageUpload image={image} onImageChange={setImage} />
			{lastUploadedImage && !image && (
				<div className="mb-2 px-2 py-1 text-xs text-neutral-600 bg-neutral-100 rounded border border-neutral-200">
					<span className="font-medium">ℹ️</span> Will use previously uploaded image. Upload a new image to change it.
				</div>
			)}
			<div className="relative">
				<textarea
					ref={textareaRef}
					className="min-h-[44px] rounded-lg pl-4 pr-12 py-2 w-full focus:outline-none focus:ring-1 focus:ring-neutral-300 border-2 border-neutral-200"
					style={{ resize: "none" }}
					placeholder={lastUploadedImage ? "Type a prompt (using previous image)..." : "Type a prompt..."}
					value={content}
					rows={1}
					onChange={handleChange}
					onKeyDown={handleKeyDown}
				/>

				<button onClick={() => handleSend()}>
					<IconArrowUp className="absolute right-2 bottom-3 h-8 w-8 hover:cursor-pointer rounded-full p-1 bg-blue-500 text-white hover:opacity-80" />
				</button>
			</div>
		</div>
	);
};
