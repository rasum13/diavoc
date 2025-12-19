import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import Button from "../components/Button";
import { Link, useNavigate } from "react-router-dom";

const SettingsButton = ({ className, children, to, ...props }) => {
  const navigate = useNavigate();

  return (
    <>
      {to ? (
        <Link
          to={to}
          className={`cursor-pointer hover:bg-neutral-50 text-left px-4 py-4 border-y-[0.5px] border-neutral-200 ${className}`}
          {...props}
        >
          {children}
        </Link>
      ) : (
        <button
          className={`cursor-pointer hover:bg-neutral-50 text-left px-4 py-4 border-y-[0.5px] border-neutral-200 ${className}`}
          {...props}
        >
          {children}
        </button>
      )}
    </>
  );
};

const ProfileSettings = () => {
  return (
    <>
      <h1 className="mb-4">Profile Settings</h1>
      <div className="py-0 rounded-md shadow-[2px_2px_12px] shadow-neutral-100 border-[0.5px] border-neutral-300 flex flex-col overflow-hidden">
        <SettingsButton to="/settings/name">Change Name</SettingsButton>
        <SettingsButton to="/settings/email">Change Email</SettingsButton>
        <SettingsButton to="/settings/password">Change Password</SettingsButton>
        <SettingsButton to="/settings/info">Update Info</SettingsButton>
      </div>
    </>
  );
};

export default ProfileSettings;
