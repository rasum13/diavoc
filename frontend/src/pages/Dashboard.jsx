import Card from "../components/Card";
import { Mic, History, Settings } from "lucide-react";
import HistoryItem from "../components/HistoryItem";
import HistoryList from "../components/HistoryList";
import HistoryChart from "../components/HistoryChart";
import { Link } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../lib/axios";
import Loading from "../components/Loading";

const Dashboard = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const getUser = async () => {
      const userData = await api.get("/user/me");
      setUser(userData.data);
    }

    getUser();
  }, [])

  if (!user) {
    return <Loading>"Loading your profile..."</Loading>
  }

  return (
    <>
      <h1 className="text-2xl font-medium mb-8">Hi, {user?.full_name}</h1>
      <div className="flex flex-col justify-center w-full">
        <div className="grid lg:grid-cols-2 gap-8">
          <Card
            className="border-4! border-primary"
            title="Record"
            desc="Record your voice"
            link="/record"
            icon={Mic}
          />
          <Card
            title="Settings"
            desc="Your settings"
            link="/settings"
            icon={Settings}
          />
        </div>
      </div>
      <h2 className="my-8 font-medium text-2xl">Your past screenings</h2>
      <div className="lg:grid lg:grid-cols-[4fr_3fr]">
        <Link to="/history">
          <HistoryList limit="4" />
        </Link>
        <HistoryChart />
      </div>
    </>
  );
};

export default Dashboard;
