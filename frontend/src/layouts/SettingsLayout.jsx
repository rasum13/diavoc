import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

const SettingsLayout = () => {
  const { isAuth } = useAuth();

  return (
    <>
      <div className="flex h-full w-full justify-center items-center">
        <Outlet />
      </div>
    </>
  );
};

export default SettingsLayout;
