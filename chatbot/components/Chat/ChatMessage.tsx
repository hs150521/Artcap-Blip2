import { Message } from "@/types";
import { FC } from "react";

interface Props {
	message: Message;
}

export const ChatMessage: FC<Props> = ({ message }) => {
	const isBlip2 = message.blip2Response !== undefined && !message.kvResponse;
	const isKV = message.kvResponse !== undefined && !message.blip2Response;
	const isBoth = message.blip2Response !== undefined && message.kvResponse !== undefined;

	// Determine label and color
	let label = "";
	let bgColor = message.role === "assistant" ? "bg-neutral-200 text-neutral-900" : "bg-blue-500 text-white";

	if (message.role === "assistant") {
		if (isBlip2) {
			label = "BLIP-2";
			bgColor = "bg-blue-100 text-blue-900 border border-blue-300";
		} else if (isKV) {
			label = "KV Model";
			bgColor = "bg-green-100 text-green-900 border border-green-300";
		}
	}

	return (
		<div className={`flex flex-col ${message.role === "assistant" ? "items-start" : "items-end"}`}>
			{message.image && (
				<div className="mb-2">
					<img
						src={message.image}
						alt="User uploaded"
						className="max-h-64 max-w-md rounded-lg object-contain border-2 border-neutral-200"
					/>
				</div>
			)}
			{message.content && (
				<div className="w-full max-w-[67%]">
					{label && (
						<div className="text-xs font-semibold mb-1 text-neutral-600">
							{label}
						</div>
					)}
					<div
						className={`flex items-center ${bgColor} rounded-2xl px-3 py-2 whitespace-pre-wrap`}
						style={{ overflowWrap: "anywhere" }}
					>
						{message.content}
					</div>
				</div>
			)}
		</div>
	);
};
