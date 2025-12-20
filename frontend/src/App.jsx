import { Navigate, Route, Routes } from "react-router-dom";

import Login from "./pages/Login";
import Survey from "./pages/Survey";
import Dashboard from "./pages/Dashboard";
import Messages from "./pages/Messages";
import Splash from "./pages/Splash";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";
import MainLayout from "./layouts/MainLayout";
import Record from "./pages/Record";
import History from "./pages/History";
import Signup from "./pages/Signup";
import { useAuth } from "./context/AuthContext";
import Loading from "./components/Loading";
import SettingsChangeName from "./pages/SettingsChangeName";
import SettingsChangeEmail from "./pages/SettingsChangeEmail";
import SettingsChangePassword from "./pages/SettingsChangePassword";
import SettingsLayout from "./layouts/SettingsLayout";
import SettingsUpdateInfo from "./pages/SettingsUpdateInfo";
import ProfileSettings from "./pages/ProfileSettings";

function App() {
  const { loading } = useAuth();
  const { isAuth } = useAuth();

  if (loading) {
    return <Loading>Loading...</Loading>;
  }

  return (
    <Routes>
      <Route path="/" element={<MainLayout />}>
        <Route index path="/" element={isAuth ? <Dashboard /> : <Navigate to="/splash" />} />
        <Route path="/record" element={<Record />} />
        <Route path="/survey" element={<Survey />} />
        <Route path="/messages" element={<Messages />} />
        <Route path="/history" element={<History />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/settings/profile" element={<ProfileSettings />} />
        <Route path="/" element={<SettingsLayout />}>
          <Route path="/settings/name" element={<SettingsChangeName />} />
          <Route path="/settings/email" element={<SettingsChangeEmail />} />
          <Route path="/settings/info" element={<SettingsUpdateInfo />} />
          <Route
            path="/settings/password"
            element={<SettingsChangePassword />}
          />
        </Route>
      </Route>
      <Route path="/login" element={isAuth ? <Navigate to="/" /> : <Login />} />
      <Route path="/signup" element={isAuth ? <Navigate to="/" /> : <Signup />} />
      <Route path="*" element={<NotFound />} />
      <Route path="/splash" element={<Splash />} />
    </Routes>
  );
}

export default App;
