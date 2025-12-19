import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import useHistory from "../hooks/useHistory";
import { parse, format } from "date-fns";

const HistoryChart = () => {
  const [history, loading, error] = useHistory(5);

  const data = history.slice();
  const formattedData = data.map((item) => {
    const dateString = item.date;
    const dateObject = parse(dateString, "yyyy-MM-dd", 0);
    const formattedDate = format(dateObject, "MMM d");
    return {
      ...item,
      score: item.score * 100,
      date: formattedDate,
    };
  });

  if (history.length == 0) {
    return <p className="border-[0.5px] border-neutral-300 p-2 m-2 rounded-md">No scans yet</p>;
  }

  return (
    <div>
      {loading ? (
        <p>Loading...</p>
      ) : (
        <LineChart
          className="w-[90%] p-4 aspect-[1.6] text-sm rounded-md"
          style={{
            zIndex: -1,
          }}
          data={formattedData}
          responsive
        >
          <CartesianGrid vertical={false} />
          <Line
            dataKey="score"
            stroke="var(--color-primary)"
            strokeWidth={3}
            name="Risk"
            animationDuration={500}
            animationEasing="ease-in-out"
            dot={{ stroke: "var(--color-primary)", strokeWidth: 6, r: 3 }}
          />
          <Line strokeWidth={3} />
          <XAxis dataKey="date" axisLine={false} tickLine={false} />
          <YAxis
            dataKey="score"
            axisLine={false}
            tickLine={false}
            domain={[0, 100]}
            unit="%"
          />
          <ReferenceLine
            y={66}
            stroke="var(--color-red)"
            strokeDasharray="5 5"
          />
          <ReferenceLine
            y={33}
            stroke="var(--color-yellow)"
            strokeDasharray="5 5"
          />
          <ReferenceArea y1={66} y2={100} fill="var(--color-red-light)" />
          <Tooltip />
        </LineChart>
      )}
    </div>
  );
};

export default HistoryChart;
