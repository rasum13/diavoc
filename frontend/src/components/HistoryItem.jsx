import { parse, format, parseISO } from "date-fns";

const HistoryItem = ({ date, score, accuracy }) => {
  const dateObject = parseISO(date);
  const formattedDate = format(dateObject, "MMM d, h:mm a");

  return (
    <div className="transition py-4 px-6 border-[0.5px] border-neutral-300 my-2 rounded-md shadow-[2px_2px_8px] shadow-neutral-100 hover:scale-101 hover:shadow-neutral-200 flex justify-between items-center">
      <div>{formattedDate}</div>
      <div className="text-left flex items-center">
        <div className={"bg-primary-light py-1 px-2 rounded-full text-center " + (score > .66 ? "bg-red-light text-red" : (score > .33 ? "bg-yellow-light text-yellow" : "bg-primary-light text-primary"))}>{(score * 100).toFixed(2)}% Risk</div>
      </div>
    </div>
  );
};

export default HistoryItem;
