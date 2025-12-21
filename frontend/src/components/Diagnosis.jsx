import WaterfallPlot from "./WaterfallPlot";

const Diagnosis = ({ score }) => {
  return (
    <>
      <div
        className={`text-center m-4 py-4 px-8 rounded-md ${score > 0.66 ? "bg-red-light" : score > 0.33 ? "bg-yellow-light" : "bg-primary-light"}`}
      >
        <p className="text-xl text-fg-light">
          <span className="font-semibold ">Score:</span>{" "}
          {(score * 100).toFixed(2)}%
        </p>
        <p
          className={`text-xl font-semibold ${score > 0.66 ? "text-red" : score > 0.33 ? "text-yellow" : "text-primary"}`}
        >
          {score > 0.66
            ? "High Risk"
            : score > 0.33
              ? "Medium Risk"
              : "Low Risk"}
        </p>
      </div>
      <WaterfallPlot />
    </>
  );
};

export default Diagnosis;
