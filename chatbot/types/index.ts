export enum OpenAIModel {
	DAVINCI_TURBO = "gpt-3.5-turbo"
}

export interface Message {
	role: Role;
	content: string;
	image?: string; // base64 encoded image
}

export type Role = "assistant" | "user";
