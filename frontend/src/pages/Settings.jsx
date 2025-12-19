import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import Button from "../components/Button";
import { Link, useNavigate } from "react-router-dom";

const LogoutButton = () => {
  const [loggingOut, setLoggingOut] = useState(false);

  const { setAuthToken } = useAuth();

  const logOut = () => {
    setAuthToken("");
  };

  return (
    <>
      {loggingOut ? (
        <SettingsButton className="cursor-default! hover:bg-white flex flex-row items-center justify-between">
          <div>Are you sure?</div>
          <div className="grid grid-cols-2 gap-2 w-100">
            <Button
              className="hover:bg-red-light hover:text-red bg-red text-fg-dark"
              onClick={logOut}
            >
              Yes
            </Button>
            <Button onClick={() => setLoggingOut(false)}>Cancel</Button>
          </div>
        </SettingsButton>
      ) : (
        <SettingsButton
          className="text-red"
          onClick={() => setLoggingOut(true)}
        >
          Logout
        </SettingsButton>
      )}
    </>
  );
};

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

const Settings = () => {
  return (
    <>
      <h1 className="mb-4">Settings</h1>
      <div className="py-0 rounded-md shadow-[2px_2px_12px] shadow-neutral-200 border-[0.5px] border-neutral-300 flex flex-col overflow-hidden">
        <SettingsButton to="/settings/name">Change Name</SettingsButton>
        <SettingsButton to="/settings/email">Change Email</SettingsButton>
        <SettingsButton to="/settings/password">Change Password</SettingsButton>
        <SettingsButton to="/settings/info">Update Info</SettingsButton>
        <LogoutButton />
      </div>
    </>
  );
};

export default Settings;
