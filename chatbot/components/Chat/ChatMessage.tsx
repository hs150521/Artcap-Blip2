import { Message } from "@/types";
import { FC } from "react";

interface Props {
	message: Message;
}

export const ChatMessage: FC<Props> = ({ message }) => {
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
				<div
					className={`flex items-center ${message.role === "assistant" ? "bg-neutral-200 text-neutral-900" : "bg-blue-500 text-white"} rounded-2xl px-3 py-2 max-w-[67%] whitespace-pre-wrap`}
					style={{ overflowWrap: "anywhere" }}
				>
					{message.content}
				</div>
			)}
		</div>
	);
};
