import { useState } from 'react';
import backgroundImage from './assets/background.png';
import mattergenLogo from './assets/mattergen-crop.png';
import mattersimLogo from './assets/mattersim-crop.png';
import mongologo from './assets/mongo.svg';

// --- NavLink Component Definition ---
function NavLink({ href, children, isActive = false }) {
  const baseClasses = "block transition duration-200 ease-in-out";
  const activeClasses = "text-emerald-950 font-bold text-2xl";
  const inactiveClasses = "text-xl font-semibold text-zinc-800 hover:underline";
  const combinedClasses = `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`;

  return (
    <a href={href} className={combinedClasses}>
      {children}
    </a>
  );
}

function App() {
  const [count, setCount] = useState(0);

  return (
    <div className="w-screen h-screen flex bg-zinc-100 text-zinc-900">
      {/* Sidebar */}
      <aside className="w-88 bg-green-100 text-zinc-800 shadow-[0_4px_20px_rgba(0,0,0,0.5)] p-10 flex flex-col justify-between">
        <div>
          {/* Sidebar Header */}
          <a 
            href="/"
            className="flex items-center space-x-5 mb-10 group"
          >
            <span className="flex items-center justify-center w-12 h-12 bg-teal-700 rounded-xl text-teal-50 font-bold font-['Poppins'] text-2xl group-hover:bg-teal-900 transition duration-200 ease-in-out">
              MB
            </span>
            <span className="text-3xl font-extrabold tracking-wider uppercase text-teal-800 group-hover:text-teal-900 transition duration-200 ease-in-out font-['Poppins'] drop-shadow-sm">
            MatBuddy
            </span>
          </a>

          {/* Sidebar Navigation */}
          <nav className="space-y-4">
            <NavLink href="#">Generate</NavLink>
            <NavLink href="#bottom">Retrieve</NavLink>
            <NavLink href="#">Download</NavLink>
          </nav>
        </div>
        {/* Sidebar Footer */}
        <div className="text-md text-zinc-500">
          &copy; 2025 SRG
        </div>
      </aside>

      {/* Main Content */}
      <main
        className="relative  flex-1 flex flex-col items-center justify-center p-4 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${backgroundImage})` }}
        >
        <div className="absolute inset-0 bg-black opacity-85 z-0"></div>
        
        {/* Logos section */}
        <div className="relative z-10 flex items-center justify-center gap-8 mb-8">
          <img src={mattergenLogo} className="h-40 hover:drop-shadow-lg" alt="MatterGen Logo" />
          <img src={mattersimLogo} className="h-40 hover:drop-shadow-lg" alt="MatterSim Logo" />
          <img src={mongologo} className="h-40 hover:drop-shadow-lg" alt="MongoDB Logo" />
        </div>
        {/* Main Heading */}
        <h1 className="relative z-10 font-['Poppins'] text-6xl mb-8 mt-8 text-center tracking-wide leading-tight text-white drop-shadow-[0_2px_4px_rgba(0,0,0,0.6)]">
          <span className="drop-shadow-lg text-7xl bg-gradient-to-r from-teal-300 via-emerald-400 to-cyan-400 bg-clip-text text-transparent">
            Artificial Intelligence for Materials Discovery
          </span>
        </h1>
        {/* Subheading */}
        <div className="relative z-10 text-center mb-6 max-w-3xl">
          <h2 className="text-3xl sm:text-4xl font-semibold text-teal-50 leading-snug">
            Welcome to <span className="text-emerald-200 font-bold font-['Inter']">MatBuddy</span>
          </h2>
          <p className="mt-2 text-lg sm:text-xl text-teal-100 font-light">
            Your AI assistant for exploring and generating material structures.
          </p>
        </div>

        {/* Action Card */}
        <div className="relative z-10 bg-teal-50 p-8 rounded-xl shadow-xl text-center mt-6 max-w-md text-zinc-800"> 
          <button
            onClick={() => setCount((count) => count + 1)}
            className="bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-6 rounded transition"
          >
            {count} {count === 1 ? "like" : "likes"}
          </button>
          <p className="mt-4 text-base font-medium">
            Start generating materials or retrieve existing structures using the sidebar!
          </p>
        </div>
      </main>
    </div>
  );
}

export default App;