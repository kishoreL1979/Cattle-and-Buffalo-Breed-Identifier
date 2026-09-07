import React from 'react';
import { Cpu } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="bg-slate-900 text-slate-400 py-8 border-t border-slate-800 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-600 flex items-center justify-center text-white">
            <Cpu className="w-5 h-5" />
          </div>
          <span className="text-slate-200 font-semibold text-sm">
            AI Cattle & Buffalo Breed Identification System
          </span>
        </div>

        <div className="text-xs text-slate-400 text-center md:text-right">
          <p>Powered by Transfer Learning (EfficientNet-B0 CNN) & PyTorch</p>
          <p className="mt-1 text-slate-500">Research & Academic Project • Phase 1 CNN Model</p>
        </div>

      </div>
    </footer>
  );
}
