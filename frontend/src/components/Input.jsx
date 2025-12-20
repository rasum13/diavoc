const Input = ({ name, type, placeholder, className, error, value, onChange, label, ...props }) => {
  return (
    <div className={`${label && "grid grid-cols-2 align-middle"} text-left`}>
      {label && <label for={name}>{label}</label>}
      <input
        id={name}
        className={`bg-neutral-100 border-[0.5px] border-neutral-400 px-4 py-2 w-full rounded-md placeholder-gray-400 ${className}`}
        name={name}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        {...props}
      />
      {error && <p className="text-red">{error}</p>}
    </div>
  );
};

export default Input;
