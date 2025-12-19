import HistoryItem from "./HistoryItem";
import useHistory from "../hooks/useHistory";

const HistoryList = ({ limit }) => {
  const [history, loading, error] = useHistory(limit);

  if (history.length == 0) {
    return <p className="border-[0.5px] border-neutral-300 p-2 m-2 rounded-md">No scans yet</p>;
  }

  return (
    <div className="w-full">
      {!loading
        ?
        history.map((value, index) => (
          <HistoryItem
            key={index}
            date={value.date}
            score={value.score}
            accuracy={value.accuracy}
          />
        )).reverse()
        : <p>Loading...</p>
      }
    </div>
  );
};

export default HistoryList;
