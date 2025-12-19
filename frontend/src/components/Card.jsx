import { Link } from "react-router-dom";

const Card = ({ title, link, desc, icon: Icon, className }) => {
  return (
    <Link
      to={link}
      className={
        "group flex flex-row content-around items-center transition ease-in-out hover:scale-102 hover:bg-primary hover:text-fg-dark hover:shadow-[2px_4px_12px] hover:shadow-neutral-300 rounded-xl border-[0.5px] border-neutral-300 shadow-[1px_3px_8px] shadow-neutral-100 p-8 " +
        className
      }
    >
      <div className="text-6xl mr-8">
        <Icon className="w-10 h-10" />
      </div>
      <div>
        <h3>{title}</h3>
        <p className="transition ease-in-out group-hover:text-fg-dark">
          {desc}
        </p>
      </div>
    </Link>
  );
};

export default Card;
