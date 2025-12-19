import { Navigate, Outlet } from "react-router-dom";
import Navbar from "../components/Navbar";
import { useAuth } from "../context/AuthContext";

const MainLayout = () => {
  const { isAuth } = useAuth();

  return (
    <>
      <Navbar />
      {
        isAuth
        ? <div className="px-8 py-8 mb-28 lg:ml-80 dark:bg-bg-dark h-200">
            <Outlet />
          </div>
        : <Navigate to="/login" />
      }
    </>
  );
};

export default MainLayout;
