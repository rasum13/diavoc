const Button = ({ className, onClick, disabled, children, ...props }) => {
  return (
    <button
      disabled={disabled}
      className={`transition ease-in-out cursor-pointer rounded-md shadow-[1px_2px_4px] shadow-neutral-200 hover:shadow-[2px_2px_8px] disabled:hover:shadow-none text-fg-dark bg-primary disabled:bg-bg-secondary disabled:text-neutral-400 disabled:cursor-not-allowed hover:bg-primary-light hover:text-primary px-4 py-2 ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;
