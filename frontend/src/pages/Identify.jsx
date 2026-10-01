import React, { useState } from 'react';
import ImageUploader from '../components/ImageUploader';
import PredictionResult from '../components/PredictionResult';
import { predictBreed } from '../services/api';
import { AlertCircle, HelpCircle } from 'lucide-react';

export default function Identify() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handlePredict = async (file) => {
    if (!file) return;
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await predictBreed(file);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to backend service.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8 py-6">
      
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          Cattle & Buffalo Breed Identification
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Upload an image of a cattle or buffalo to identify its breed using CABBI BreedVision.
        </p>
      </div>

      {/* Global Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-sm flex items-start gap-3 shadow-xs">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold">Prediction Request Failed</h4>
            <p className="text-xs text-red-700 mt-0.5">{error}</p>
          </div>
        </div>
      )}

      {/* Main Grid: Upload Area & Result Card */}
      <div className="grid lg:grid-cols-2 gap-8 items-start">
        
        {/* Left Column: Image Uploader */}
        <div>
          <ImageUploader
            onPredict={handlePredict}
            isLoading={isLoading}
            selectedFile={selectedFile}
            setSelectedFile={setSelectedFile}
            previewUrl={previewUrl}
            setPreviewUrl={setPreviewUrl}
          />
        </div>

        {/* Right Column: Prediction Result or Placeholder */}
        <div>
          {result ? (
            <PredictionResult result={result} previewUrl={previewUrl} />
          ) : (
            <div className="bg-white rounded-2xl p-8 border border-slate-200 text-center text-slate-400 space-y-4 min-h-[350px] flex flex-col items-center justify-center">
              <div className="w-16 h-16 rounded-full bg-blue-50 flex items-center justify-center text-blue-500">
                <HelpCircle className="w-8 h-8" />
              </div>
              <div>
                <h3 className="font-bold text-slate-700 text-base">No Prediction Yet</h3>
                <p className="text-xs text-slate-400 max-w-xs mt-1">
                  Upload an image on the left and click "Identify Breed" to view classification results.
                </p>
              </div>
            </div>
          )}
        </div>

      </div>

    </div>
  );
}
