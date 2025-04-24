import backgroundImage from './assets/background.png';
import { Routes, Route, Link, useLocation } from 'react-router-dom';

// import components
import Home from './components/Home';
import GenerateLattice from './components/GenerateLattice';
import DownloadLattice from './components/DownloadLattice';

// --- NavLink Component Definition ---
function NavLink({ to, children }) {
  const location = useLocation();
  const isActive = location.pathname === to;
  const baseClasses = "block transition duration-200 ease-in-out";
  const activeClasses = "text-emerald-950 font-black text-2xl";
  const inactiveClasses = "text-xl font-semibold text-zinc-800 hover:underline";
  const combinedClasses = `${baseClasses} ${isActive ? activeClasses : inactiveClasses}`;

  return (
    <Link to={to} className={combinedClasses}>
      {children}
    </Link>
  );
}

function App() {

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
            <NavLink to="/generate">Generate</NavLink>
            <NavLink to="/retrieve">Retrieve</NavLink>
            <NavLink to="/download">Download</NavLink>
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
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/generate" element={<GenerateLattice />} />
          <Route path="/download" element={<DownloadLattice />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;