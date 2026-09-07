import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Identify from './pages/Identify';
import About from './pages/About';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8">
        {activeTab === 'home' && <Home onNavigateToIdentify={() => setActiveTab('identify')} />}
        {activeTab === 'identify' && <Identify />}
        {activeTab === 'about' && <About />}
      </main>

      <Footer />
    </div>
  );
}
