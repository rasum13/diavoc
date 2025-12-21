import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  LabelList,
  ResponsiveContainer,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
} from "recharts";
import { api } from "../lib/axios";
import Loading from "./Loading";

const patients = [
  {
    ada_score: 5,
    pitch: 86.201,
    jitter: 0.405,
    shimmer: 0.073,
    gender: 1,
    age: 53,
    bmi: 20.9,
    ethnicity: 1,
    prediction: 0.141,
    shap: {
      ada_score: -0.35,
      pitch: 0.08,
      jitter: 0.08,
      shimmer: 0.07,
      gender: -0.05,
      age: -0.04,
      bmi: 0.03,
      ethnicity: 0.01,
    },
  },
  {
    ada_score: 15,
    pitch: 120.5,
    jitter: 0.012,
    shimmer: 0.025,
    gender: 0,
    age: 67,
    bmi: 24.3,
    ethnicity: 2,
    prediction: 0.452,
    shap: {
      ada_score: 0.15,
      pitch: 0.05,
      jitter: -0.08,
      shimmer: -0.06,
      gender: 0.02,
      age: 0.03,
      bmi: 0.04,
      ethnicity: 0.02,
    },
  },
  {
    ada_score: 8,
    pitch: 95.8,
    jitter: 0.289,
    shimmer: 0.156,
    gender: 1,
    age: 45,
    bmi: 28.1,
    ethnicity: 1,
    prediction: 0.298,
    shap: {
      ada_score: -0.1,
      pitch: -0.02,
      jitter: 0.05,
      shimmer: 0.08,
      gender: -0.03,
      age: -0.05,
      bmi: 0.06,
      ethnicity: 0.01,
    },
  },
];

const buildContributions = (patient) => {
  let cumulative = baseValue;
  const contributions = [
    {
      feature: "Base",
      value: baseValue,
      cumulative: cumulative,
      contribution: 0,
    },
  ];

  // Add each feature contribution
  Object.keys(patient.shap).forEach((key) => {
    cumulative += patient.shap[key];
    contributions.push({
      feature: `${key} = ${patient[key]}`,
      value: patient.shap[key],
      cumulative: cumulative,
      contribution: patient.shap[key],
    });
  });

  contributions.push({
    feature: "Final",
    value: patient.prediction,
    cumulative: patient.prediction,
    contribution: 0,
  });

  return contributions;
};

const baseValue = 0.318;

const WaterfallPlot = () => {
  const [user, setUser] = useState(null);

  useEffect(() => {
    const getUser = async () => {
      const userData = await api.get("/user/me");
      setUser(userData.data);
    };

    getUser();
  }, []);

  if (!user) {
    return <Loading>"Loading your profile..."</Loading>;
  }

  const data = patients[user.weight_kg % 3];
  const contributions = buildContributions(data);

  const chartData = contributions.map((item, index) => {
    if (index === 0 || index === contributions.length - 1) {
      // Base and Final are absolute values
      return {
        ...item,
        start: 0,
        end: item.value,
        displayValue: item.value,
      };
    } else {
      // Calculate start position from previous cumulative
      const prevCumulative = contributions[index - 1].cumulative;
      return {
        ...item,
        start: Math.min(prevCumulative, item.cumulative),
        end: Math.max(prevCumulative, item.cumulative),
        displayValue: item.contribution,
      };
    }
  });

  return (
    <ResponsiveContainer width="100%" aspect={1.618} maxHeight={500}>
      <BarChart
        width={1000}
        height={500}
        data={chartData}
        margin={{ top: 20, right: 30, left: 100, bottom: 60 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="feature"
          angle={-45}
          textAnchor="end"
          height={100}
          interval={0}
        />
        <YAxis
          label={{
            value: "Prediction Value",
            angle: -90,
            position: "insideLeft",
          }}
          domain={[0, 0.6]}
          tickFormatter={(value) => value.toFixed(2)}
        />
        <Tooltip />
        <ReferenceLine
          y={baseValue}
          stroke="#666"
          strokeDasharray="3 3"
          label="Base"
        />
        <ReferenceLine
          y={data.prediction}
          stroke="#666"
          strokeDasharray="3 3"
          label="Final"
        />

        {/* Invisible bar to set the start position */}
        <Bar dataKey="start" stackId="a" fill="transparent" />

        {/* Visible bar showing the contribution */}
        <Bar dataKey={(entry) => entry.end - entry.start} stackId="a">
          {chartData.map((entry, index) => {
            let color = "#94a3b8"; // gray for base/final
            if (index !== 0 && index !== chartData.length - 1) {
              color =
                entry.contribution < 0
                  ? "var(--color-primary)"
                  : "var(--color-red)"; // blue for negative, pink for positive
            }
            return <Cell key={`cell-${index}`} fill={color} />;
          })}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default WaterfallPlot;
