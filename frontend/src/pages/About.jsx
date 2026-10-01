import React from 'react';
import { Cpu, ShieldCheck, Database, Layers, CheckCircle2, GitBranch } from 'lucide-react';

export default function About() {
  return (
    <div className="space-y-10 py-6 max-w-5xl mx-auto">
      
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-slate-900 tracking-tight">
          About the System
        </h1>
        <p className="text-sm text-slate-500 mt-1">
          Deep learning architecture, dataset overview, and phase roadmap.
        </p>
      </div>

      {/* Model Overview Section */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800">CNN Model Architecture</h2>
            <p className="text-xs text-slate-400">EfficientNetV2 Transfer Learning with PyTorch</p>
          </div>
        </div>

        <p className="text-sm text-slate-600 leading-relaxed">
          The core classification model is based on <strong>EfficientNetV2</strong> pre-trained on ImageNet. It utilizes compound scaling of depth, width, and resolution to achieve high classification accuracy while maintaining fast inference speed suitable for real-time agricultural applications.
        </p>

        <div className="grid sm:grid-cols-2 gap-4 pt-2">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-slate-800 text-sm mb-1">Stage 1: Head Training</h4>
            <p className="text-xs text-slate-600">Backbone features are frozen. The custom fully-connected classifier head (Dropout 0.3 + Linear 9-class) is trained with AdamW ($lr=10^{-3}$).</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200">
            <h4 className="font-bold text-slate-800 text-sm mb-1">Stage 2: Fine-Tuning</h4>
            <p className="text-xs text-slate-600">Upper MBConv feature blocks are unfrozen and fine-tuned with a low learning rate ($lr=10^{-4}$) and Cosine Annealing scheduler.</p>
          </div>
        </div>
      </section>

      {/* Dataset & Class Breakdown */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800">Dataset Overview</h2>
            <p className="text-xs text-slate-400">843 total images across 9 indigenous classes</p>
          </div>
        </div>

        <div className="grid sm:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-blue-50 border border-blue-200 text-center">
            <span className="text-2xl font-black text-blue-700">488</span>
            <p className="text-xs text-blue-800 font-semibold mt-0.5">Train Images (~58%)</p>
          </div>
          <div className="p-4 rounded-xl bg-indigo-50 border border-indigo-200 text-center">
            <span className="text-2xl font-black text-indigo-700">248</span>
            <p className="text-xs text-indigo-800 font-semibold mt-0.5">Validation Images (~29%)</p>
          </div>
          <div className="p-4 rounded-xl bg-amber-50 border border-amber-200 text-center">
            <span className="text-2xl font-black text-amber-700">107</span>
            <p className="text-xs text-amber-800 font-semibold mt-0.5">Test Images (~13%)</p>
          </div>
        </div>
      </section>

      {/* Multi-Modal Research System Pipeline */}
      <section className="bg-white rounded-2xl p-6 sm:p-8 border border-slate-200 shadow-sm space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-blue-100 text-blue-700 flex items-center justify-center">
            <GitBranch className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800">Proposed Multi-Modal System Architecture</h2>
            <p className="text-xs text-slate-400">YOLO ROI Detection + EfficientNetV2 + Gemini Vision API + Adaptive Fusion</p>
          </div>
        </div>

        <div className="space-y-4">
          <div className="p-4 rounded-xl border border-blue-200 bg-blue-50/50">
            <div className="flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-5 h-5 text-blue-600" />
              <h4 className="font-bold text-blue-900 text-sm">CABBI BreedVision Research Pipeline</h4>
            </div>
            <p className="text-xs text-slate-700 ml-7">
              Input Image $\rightarrow$ YOLO Animal Detection & ROI Cropping $\rightarrow$ EfficientNetV2 Classifier $\rightarrow$ Gemini Vision Analysis $\rightarrow$ Highest Match Arbitration $\rightarrow$ Final Recommendation.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}
