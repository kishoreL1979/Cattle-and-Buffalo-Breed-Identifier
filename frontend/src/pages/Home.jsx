import React from 'react';
import { Cpu, ArrowRight, ShieldCheck, Zap, Database, Layers } from 'lucide-react';

export default function Home({ onNavigateToIdentify }) {
  const breeds = [
    { name: 'Gir', type: 'Cattle', origin: 'Gujarat', desc: 'Indigenous dairy cattle breed famous for heat tolerance.' },
    { name: 'Jaffrabadi', type: 'Buffalo', origin: 'Gujarat', desc: 'Heavy buffalo breed known for high butterfat milk production.' },
    { name: 'Kankrej', type: 'Cattle', origin: 'Gujarat/Rajasthan', desc: 'Dual-purpose breed valued for draught and milk capability.' },
    { name: 'Mehsana', type: 'Buffalo', origin: 'Gujarat', desc: 'Cross-breed of Murrah and Surti, high milk yield.' },
    { name: 'Murrah', type: 'Buffalo', origin: 'Haryana/Punjab', desc: 'Premier dairy buffalo breed worldwide.' },
    { name: 'Red Sindhi', type: 'Cattle', origin: 'Sindh region', desc: 'Deep red dairy cattle known for disease resistance.' },
    { name: 'Sahiwal', type: 'Cattle', origin: 'Punjab region', desc: 'Top zebu dairy breed producing rich A2 milk.' },
    { name: 'Surti', type: 'Buffalo', origin: 'Gujarat', desc: 'Medium-sized docile buffalo breed with sickle-shaped horns.' },
    { name: 'Tharparkar', type: 'Cattle', origin: 'Thar Desert', desc: 'Resilient white dual-purpose desert cattle.' },
  ];

  return (
    <div className="space-y-16 py-8">
      
      {/* Hero Banner */}
      <section className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-slate-900 via-slate-800 to-emerald-950 text-white p-8 sm:p-12 shadow-xl border border-slate-800">
        <div className="max-w-3xl relative z-10 space-y-6">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
            <Cpu className="w-4 h-4" />
            <span>EfficientNet-B0 CNN Transfer Learning</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-black tracking-tight leading-tight">
            AI-Powered Cattle & Buffalo Breed Identification
          </h1>

          <p className="text-slate-300 text-base sm:text-lg leading-relaxed">
            Automated image classification system for identifying 9 major indigenous cattle and buffalo breeds using deep convolutional neural networks.
          </p>

          <div className="pt-2 flex flex-wrap gap-4">
            <button
              onClick={onNavigateToIdentify}
              className="px-6 py-3.5 rounded-xl bg-emerald-500 hover:bg-emerald-600 text-white font-bold text-base flex items-center gap-2 transition-all shadow-lg shadow-emerald-500/25"
            >
              <span>Identify Breed Now</span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      </section>

      {/* Key System Features */}
      <section className="grid md:grid-cols-3 gap-6">
        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
          <div className="w-12 h-12 rounded-xl bg-emerald-100 text-emerald-700 flex items-center justify-center">
            <Layers className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-slate-800">9 Breed Classes</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Classifies 5 major cattle breeds (Gir, Kankrej, Red Sindhi, Sahiwal, Tharparkar) and 4 buffalo breeds (Jaffrabadi, Mehsana, Murrah, Surti).
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
          <div className="w-12 h-12 rounded-xl bg-teal-100 text-teal-700 flex items-center justify-center">
            <Zap className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-slate-800">Fast CNN Prediction</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Instant probability calculation and top-3 prediction breakdown powered by PyTorch and FastAPI.
          </p>
        </div>

        <div className="bg-white rounded-2xl p-6 border border-slate-200 shadow-sm space-y-3">
          <div className="w-12 h-12 rounded-xl bg-amber-100 text-amber-800 flex items-center justify-center">
            <Database className="w-6 h-6" />
          </div>
          <h3 className="text-lg font-bold text-slate-800">Imbalance Handling</h3>
          <p className="text-sm text-slate-600 leading-relaxed">
            Trained with class-weighted cross-entropy loss to ensure robust recognition across rare and common breeds.
          </p>
        </div>
      </section>

      {/* Supported Breeds Section */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-slate-900">Supported Breeds</h2>
            <p className="text-sm text-slate-500">List of 9 indigenous cattle and buffalo breeds recognized by the CNN model</p>
          </div>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {breeds.map((breed, idx) => (
            <div key={idx} className="bg-white rounded-xl p-4 border border-slate-200 shadow-xs hover:border-emerald-300 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <h4 className="font-bold text-slate-800 text-base">{breed.name}</h4>
                <span className={`text-xs px-2.5 py-0.5 rounded-full font-semibold ${
                  breed.type === 'Cattle' ? 'bg-amber-100 text-amber-800' : 'bg-teal-100 text-teal-800'
                }`}>
                  {breed.type}
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium mb-1">Origin: {breed.origin}</p>
              <p className="text-xs text-slate-600 leading-normal">{breed.desc}</p>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}
