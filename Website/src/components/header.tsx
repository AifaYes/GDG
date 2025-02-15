import React from "react";
import "./header.css";
import logo from "./logo.png";
const Header: React.FC = () => {
  return (
    <div className="header">
      {logo && <img src={logo} alt="Logo" className="logo" />}
      <h1>AI TikTok Agent</h1>
    </div>
  );
};

export default Header;
