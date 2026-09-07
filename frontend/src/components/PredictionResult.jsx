import React from 'react';
import { Award, ListOrdered, CheckCircle2 } from 'lucide-react';

export default function PredictionResult({ result }) {
  if (!result) return null;

  const {
    animal_type,
    final_breed,
    top_predictions
  } = result;

  const isCattle = animal_type === 'Cattle';

  // Build top recommendations list starting with final_breed as #1
  const recommendations = [];
  if (final_breed) {
    recommendations.push(final_breed);
  }
  
  if (top_predictions && Array.isArray(top_predictions)) {
    top_predictions.forEach((item) => {
      if (item.breed && !recommendations.includes(item.breed)) {
        recommendations.push(item.breed);
      }
    });
  }

  return (
    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-200 space-y-6">
      
      {/* Header & Recommended Breed */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
        <div>
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span
              className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider ${
                isCattle
                  ? 'bg-amber-100 text-amber-800 border border-amber-200'
                  : 'bg-teal-100 text-teal-800 border border-teal-200'
              }`}
            >
              <Award className="w-3.5 h-3.5" />
              {animal_type}
            </span>

            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
              Primary Recommendation
            </span>
          </div>

          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400 mt-1">Recommended Breed</p>
          <h2 className="text-3xl font-extrabold text-slate-900 tracking-tight mt-0.5">
            {final_breed}
          </h2>
        </div>
      </div>

      {/* Top Recommendations Section */}
      <div className="border-t border-slate-100 pt-5">
        <h4 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
          <ListOrdered className="w-4 h-4 text-emerald-600" />
          Top Recommendations
        </h4>

        <div className="space-y-2.5">
          {recommendations.map((breedName, index) => {
            const isTop = index === 0;

            return (
              <div
                key={index}
                className={`p-3.5 rounded-xl flex items-center justify-between border transition-colors ${
                  isTop
                    ? 'bg-emerald-50/70 border-emerald-200 text-slate-900 font-bold'
                    : 'bg-slate-50 border-slate-200/80 text-slate-700 font-medium'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span
                    className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-extrabold ${
                      isTop
                        ? 'bg-emerald-600 text-white shadow-xs'
                        : 'bg-slate-200 text-slate-600'
                    }`}
                  >
                    {index + 1}
                  </span>
                  <span className="text-base">{breedName}</span>
                </div>

                {isTop && (
                  <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800">
                    Best Match
                  </span>
                )}
              </div>
            );
          })}
        </div>
      </div>

    </div>
  );
}
