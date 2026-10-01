import React from 'react';
import { Cpu, ShieldCheck } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'home', label: 'Home' },
    { id: 'identify', label: 'Breed Identification' },
    { id: 'about', label: 'About System' },
  ];

  return (
    <header className="sticky top-0 z-50 bg-white/95 backdrop-blur-md border-b border-slate-200 shadow-xs">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo / Brand */}
          <div 
            className="flex items-center space-x-3 cursor-pointer"
            onClick={() => setActiveTab('home')}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-700 flex items-center justify-center text-white shadow-md shadow-blue-500/20">
              <Cpu className="w-6 h-6" />
            </div>
            <div>
              <span className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-blue-900 via-blue-700 to-indigo-800 bg-clip-text text-transparent">
                CABBI BreedVision
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`px-3 py-2 sm:px-4 rounded-lg text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-blue-50 text-blue-700 font-semibold shadow-xs border border-blue-200'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* System Status Pill */}
          <div className="hidden lg:flex items-center space-x-2 px-3 py-1 bg-slate-100 border border-slate-200 rounded-full text-xs font-medium text-slate-700">
            <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
            <ShieldCheck className="w-3.5 h-3.5 text-blue-600" />
            <span>YOLO + EfficientNetV2 + Gemini Vision</span>
          </div>

        </div>
      </div>
    </header>
  );
}
