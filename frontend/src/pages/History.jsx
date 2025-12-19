import Container from "../components/Container";
import HistoryChart from "../components/HistoryChart";
import HistoryItem from "../components/HistoryItem";
import HistoryList from "../components/HistoryList";

const History = () => {
  return (
    <>
      <h1>History</h1>
      <p className="mb-4">Your past screenings</p>
      <div className="lg:grid lg:grid-cols-2">
        <HistoryList />
        <HistoryChart />
      </div>
    </>
  );
};

export default History;
