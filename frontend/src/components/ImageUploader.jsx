import React, { useState, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, X, AlertCircle, ArrowRight, Loader2 } from 'lucide-react';

export default function ImageUploader({ onPredict, isLoading, selectedFile, setSelectedFile, previewUrl, setPreviewUrl }) {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const validateFile = (file) => {
    setError(null);
    if (!file) return false;

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setError('Invalid format. Please upload JPG, PNG, or WEBP images.');
      return false;
    }

    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      setError('File size exceeds 10MB limit.');
      return false;
    }

    return true;
  };

  const handleFileChange = (file) => {
    if (validateFile(file)) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleRemove = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  return (
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200">
      <h3 className="text-lg font-bold text-slate-800 mb-4 flex items-center gap-2">
        <ImageIcon className="w-5 h-5 text-emerald-600" />
        Upload Animal Image
      </h3>

      {error && (
        <div className="mb-4 p-3 rounded-xl bg-red-50 border border-red-200 text-red-700 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {!previewUrl ? (
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 ${
            dragActive
              ? 'border-emerald-500 bg-emerald-50/50 scale-[0.99]'
              : 'border-slate-300 hover:border-emerald-400 hover:bg-slate-50'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/jpg,image/png,image/webp"
            className="hidden"
            onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
          />
          
          <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-emerald-100 text-emerald-700 flex items-center justify-center">
            <UploadCloud className="w-8 h-8" />
          </div>
          
          <p className="text-base font-semibold text-slate-700">
            Drag and drop your image here, or <span className="text-emerald-600 underline">browse</span>
          </p>
          <p className="text-xs text-slate-400 mt-2">
            Supports JPG, PNG, WEBP up to 10MB
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative rounded-2xl overflow-hidden border border-slate-200 bg-slate-900 group max-h-96 flex items-center justify-center">
            <img
              src={previewUrl}
              alt="Uploaded Cattle or Buffalo"
              className="max-h-96 w-auto object-contain rounded-xl"
            />
            <button
              onClick={handleRemove}
              disabled={isLoading}
              className="absolute top-3 right-3 p-2 bg-slate-900/80 hover:bg-red-600 text-white rounded-full transition-colors duration-150 backdrop-blur-sm"
              title="Remove image"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-500 px-1">
            <span className="font-medium truncate max-w-xs">{selectedFile?.name}</span>
            <span>{(selectedFile?.size / (1024 * 1024)).toFixed(2)} MB</span>
          </div>

          <button
            onClick={() => onPredict(selectedFile)}
            disabled={isLoading}
            className="w-full py-3.5 px-6 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-700 hover:from-emerald-700 hover:to-teal-800 text-white font-semibold shadow-md shadow-emerald-600/20 flex items-center justify-center gap-2 transition-all duration-200 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Running EfficientNet-B0 CNN Inference...</span>
              </>
            ) : (
              <>
                <span>Identify Breed</span>
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
