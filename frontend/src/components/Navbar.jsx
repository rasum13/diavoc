import { Link, NavLink } from "react-router-dom";
import { History, LayoutDashboard, Mic, Settings } from "lucide-react";

const Navbar = () => {
  const linkClass = ({ isActive }) => {
    return (
      (isActive ? "bg-primary text-white" : "hover:bg-primary hover:text-fg-dark") +
      " px-6 py-3 my-1 rounded-md transition ease-in-out lg:grid lg:grid-cols-[2rem_auto] lg:gap-1 grow-1 flex justify-center items-center lg:justify-start"
    );
  };

  return (
    <nav className="p-4 lg:w-80 w-full h-28 fixed bottom-0 lg:h-screen dark:text-fg-dark">
      <div className="px-4 h-full lg:w-full flex flex-col py-2 rounded-md bg-bg-secondary dark:bg-neutral-800/50 shadow-[2px_1px_20px] shadow-neutral-200 hover:shadow-[4px_2px_24px] transition ease-in-out">
        <Link className="my-8 font-bold lg:flex items-center justify-center text-3xl text-primary hidden" to="/">
          DiaVoc
        </Link>
        <div className="w-auto h-full grid grid-cols-4 gap-4 justify-between lg:h-auto lg:flex lg:flex-col lg:gap-0">
          <NavLink className={linkClass} to="/">
            <LayoutDashboard /> <span className="hidden lg:inline">Dashboard</span>
          </NavLink>
          <NavLink className={linkClass} to="/record">
            <Mic /> <span className="hidden lg:inline">Record</span>
          </NavLink>
          <NavLink className={linkClass} to="/history">
            <History /> <span className="hidden lg:inline">History</span>
          </NavLink>
          <NavLink className={linkClass} to="/settings">
            <Settings /> <span className="hidden lg:inline">Settings</span>
          </NavLink>
          {/* <NavLink className={linkClass} to="/messages"> */}
          {/*   Messages */}
          {/* </NavLink> */}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
