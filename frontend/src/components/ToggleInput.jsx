import { useEffect, useState } from "react";

const ToggleInput = ({ label, trueValue, falseValue, onChange, currentValue, error }) => {
  const [yes, setYes] = useState(false);
  const classList = "p-2 rounded-md cursor-pointer";
  const selectedClassList = "bg-primary text-fg-dark";
  const unSelectedClassList = "hover:bg-primary-light";

  useEffect(() => {
    onChange(yes);
  }, [yes])

  useEffect(() => {
    setYes(currentValue);
  }, [currentValue])

  return (
    <div className="grid grid-cols-2 items-center">
      <div className="text-left">{label}</div>
      <div className="grid grid-cols-2 gap-2">
        <div
          onClick={() => setYes(true)}
          className={`${classList} p-2 rounded-md ${yes ? selectedClassList : unSelectedClassList}`}
        >
          {trueValue}
        </div>
        <div
          onClick={() => setYes(false)}
          className={`${classList} ${!yes ? selectedClassList : unSelectedClassList}`}
        >
          {falseValue}
        </div>
      </div>
      {error && <p className="text-red">{error}</p>}
    </div>
  );
};

export default ToggleInput;
