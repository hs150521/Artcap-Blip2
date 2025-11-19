import { IconPhoto, IconX } from "@tabler/icons-react";
import { FC, useRef, useState, DragEvent } from "react";

interface Props {
	image: string | null;
	onImageChange: (image: string | null) => void;
}

export const ImageUpload: FC<Props> = ({ image, onImageChange }) => {
	const fileInputRef = useRef<HTMLInputElement>(null);
	const [isDragging, setIsDragging] = useState(false);

	const handleFileSelect = (file: File) => {
		if (!file.type.startsWith("image/")) {
			alert("Please select an image file");
			return;
		}

		const reader = new FileReader();
		reader.onload = (e) => {
			const result = e.target?.result as string;
			onImageChange(result);
		};
		reader.readAsDataURL(file);
	};

	const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (file) {
			handleFileSelect(file);
		}
	};

	const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		setIsDragging(true);
	};

	const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		setIsDragging(false);
	};

	const handleDrop = (e: DragEvent<HTMLDivElement>) => {
		e.preventDefault();
		setIsDragging(false);

		const file = e.dataTransfer.files?.[0];
		if (file) {
			handleFileSelect(file);
		}
	};

	const handleRemoveImage = () => {
		onImageChange(null);
		if (fileInputRef.current) {
			fileInputRef.current.value = "";
		}
	};

	return (
		<div className="mb-2">
			{image ? (
				<div className="relative inline-block">
					<img
						src={image}
						alt="Uploaded"
						className="max-h-32 max-w-xs rounded-lg object-contain border-2 border-neutral-200"
					/>
					<button
						onClick={handleRemoveImage}
						className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600"
					>
						<IconX size={16} />
					</button>
				</div>
			) : (
				<div
					onDragOver={handleDragOver}
					onDragLeave={handleDragLeave}
					onDrop={handleDrop}
					className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-colors ${isDragging
							? "border-blue-500 bg-blue-50"
							: "border-neutral-300 hover:border-neutral-400"
						}`}
					onClick={() => fileInputRef.current?.click()}
				>
					<input
						ref={fileInputRef}
						type="file"
						accept="image/*"
						onChange={handleFileInputChange}
						className="hidden"
					/>
					<IconPhoto className="mx-auto mb-2 text-neutral-400" size={32} />
					<p className="text-sm text-neutral-600">
						Click to upload or drag and drop
					</p>
					<p className="text-xs text-neutral-400 mt-1">
						PNG, JPG, WEBP up to 10MB
					</p>
				</div>
			)}
		</div>
	);
};



