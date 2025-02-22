import React from "react";

const LoginCard = () => {
  return (
    <div className="h-screen flex items-center justify-center bg-[url('./src/assets/background-pattern.png')] bg-cover">
      <div className="relative w-96 p-6 bg-blue-300 rounded-lg shadow-lg text-center">
        {/* Character Image */}
        <div className="absolute -top-24 left-1/2 transform -translate-x-1/2">
          <img
            src="./src/assets/trail2.png"
            alt="Character"
            className="w-32 h-32 object-contain"
          />
        </div>

        {/* Login Form */}
        <h2 className="text-xl font-bold text-gray-800 mt-10">LOGIN</h2>
        <div className="mt-4">
          <input
            type="text"
            placeholder="Username"
            className="w-full p-2 rounded border border-gray-400 mb-3"
          />
          <input
            type="password"
            placeholder="Password"
            className="w-full p-2 rounded border border-gray-400 mb-3"
          />
          <button className="w-full bg-gray-700 text-white py-2 rounded font-bold">
            SIGN IN
          </button>
        </div>
        <div className="flex justify-between text-sm mt-2 text-gray-600">
          <label className="flex items-center">
            <input type="checkbox" className="mr-1" /> Remember me
          </label>
          <a href="#" className="text-blue-700">Forgot your password?</a>
        </div>
      </div>
    </div>
  );
};

export default LoginCard;
