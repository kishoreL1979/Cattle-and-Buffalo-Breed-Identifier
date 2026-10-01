import React from 'react';
import { CheckCircle2, ListOrdered, Award } from 'lucide-react';

export default function PredictionResult({ result, previewUrl }) {
  if (!result) return null;

  const {
    final_breed,
    final_confidence,
    top_predictions,
    ai_breed,
    ai_confidence
  } = result;

  const finalPct = final_confidence ? (final_confidence * 100).toFixed(1) : '0.0';

  // Gather valid alternative suggestions (excluding final_breed)
  const alternativeSuggestions = [];

  if (ai_breed && ai_breed.toLowerCase() !== final_breed.toLowerCase()) {
    alternativeSuggestions.push({
      breed: ai_breed,
      confidence: ai_confidence
    });
  }

  if (top_predictions && Array.isArray(top_predictions)) {
    top_predictions.forEach((item) => {
      if (
        item.breed &&
        item.breed.toLowerCase() !== final_breed.toLowerCase() &&
        !alternativeSuggestions.some(s => s.breed.toLowerCase() === item.breed.toLowerCase())
      ) {
        alternativeSuggestions.push({
          breed: item.breed,
          confidence: item.confidence
        });
      }
    });
  }

  const validSuggestions = alternativeSuggestions.slice(0, 2);

  return (
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
      
      {/* Top Status Header */}
      <div className="flex items-center gap-2 px-3 py-1 bg-blue-100/80 text-blue-800 border border-blue-200/80 rounded-full text-xs font-bold w-fit">
        <CheckCircle2 className="w-4 h-4 text-blue-600 fill-blue-600/20" />
        <span>Identification Complete</span>
      </div>

      {/* 1. Final Recommendation Card */}
      <div className="bg-gradient-to-br from-blue-50/80 to-indigo-50/50 border border-blue-100 rounded-2xl p-5 flex flex-col sm:flex-row items-center sm:items-stretch gap-5 shadow-xs">
        {/* Thumbnail Image */}
        {previewUrl ? (
          <img
            src={previewUrl}
            alt={final_breed}
            className="w-28 h-28 sm:w-32 sm:h-32 rounded-xl object-cover border border-blue-200 shadow-sm shrink-0"
          />
        ) : (
          <div className="w-28 h-28 sm:w-32 sm:h-32 rounded-xl bg-blue-100 flex items-center justify-center text-blue-700 font-bold shrink-0">
            <Award className="w-10 h-10 text-blue-600" />
          </div>
        )}

        {/* Recommendation Details */}
        <div className="flex flex-col justify-between py-0.5 text-center sm:text-left flex-1">
          <div>
            <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800 border border-blue-200/60 mb-2">
              FINAL RECOMMENDATION
            </span>
            <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight">
              {final_breed}
            </h2>
            <div className="flex items-center justify-center sm:justify-start gap-2 mt-2">
              <span className="text-2xl font-bold text-blue-600">{finalPct}%</span>
              <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider bg-white/80 px-2.5 py-1 rounded border border-slate-200">
                Match Confidence
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Alternative Suggestions Section (ONLY IF VALID ALTERNATIVE PREDICTIONS EXIST) */}
      {validSuggestions.length > 0 && (
        <div className="border-t border-slate-100 pt-5 space-y-3">
          <h4 className="text-sm font-bold text-slate-700 flex items-center gap-2">
            <ListOrdered className="w-4 h-4 text-blue-600" />
            Alternative Suggestions
          </h4>

          <div className="space-y-2.5">
            {validSuggestions.map((item, index) => (
              <div
                key={index}
                className="p-3.5 rounded-xl flex items-center justify-between border bg-slate-50 border-slate-200/80 text-slate-700 font-medium"
              >
                <div className="flex items-center gap-3">
                  <span className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-extrabold bg-slate-200 text-slate-600">
                    {index + 1}
                  </span>
                  <span className="text-base font-semibold text-slate-800">{item.breed}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
