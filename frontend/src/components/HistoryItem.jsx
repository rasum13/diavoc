import { parse, format } from "date-fns";

const HistoryItem = ({ date, score, accuracy }) => {
  const dateObject = parse(date, "yyyy-MM-dd", 0);
  const formattedDate = format(dateObject, "MMMM d");

  return (
    <div className="transition py-4 px-6 border-[0.5px] border-neutral-300 my-2 rounded-md shadow-[2px_2px_8px] shadow-neutral-100 hover:scale-101 hover:shadow-neutral-200 flex justify-between items-center">
      <div>{formattedDate}</div>
      <div className="text-left flex items-center">
        <div className={"bg-primary-light w-18 py-1 rounded-full text-center " + (score > .66 ? "bg-red-light text-red" : (score > .33 ? "bg-yellow-light text-yellow" : "bg-primary-light text-primary"))}>{score * 100}/100</div>
      </div>
    </div>
  );
};

export default HistoryItem;
