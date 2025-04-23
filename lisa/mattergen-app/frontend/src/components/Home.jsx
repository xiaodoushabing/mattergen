import { useState } from 'react';
import mattergenLogo from '../assets/mattergen-crop.png';
import mattersimLogo from '../assets/mattersim-crop.png';
import mongologo from '../assets/mongo.svg';

function Home() {
    const [count, setCount] = useState(0);
    return (
        <>
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
            {/* <h2 className="text-3xl sm:text-4xl font-semibold text-teal-50 leading-snug">
                Welcome to <span className="text-emerald-200 font-bold font-['Inter']">MatBuddy</span>
            </h2> */}
            <p className="mt-2 text-lg sm:text-xl text-teal-100 font-light">
                Your AI assistant for exploring and generating material structures
                <br></br>
                Powered by{' '}
                <a
                href="https://github.com/microsoft/mattergen"
                target="_blank" //opens in a new tab
                rel="noopener noreferrer" // security best practice
                className='font-semibold hover:font-bold hover:text-emerald-100 transition duration-150 ease-in-out'
                >
                MatterGen
                </a>
                {' '}and{' '}
                <a
                href="https://github.com/microsoft/mattersim"
                target='_blank'
                rel="noopener noreferrer"
                className='font-semibold hover:font-bold hover:text-emerald-100 transition duration-150 ease-in-out'
                >
                MatterSim
                </a>
                {' '}open-source models
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
        </>
    )
}

export default Home;