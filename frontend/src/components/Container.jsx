const Container = ({ className, children }) => {
  return (
    <div className="px-16 py-8 flex flex-col w-lg bg-neutral-50 rounded-md shadow-[0_8px_32px] shadow-neutral-200 text-center">
      {children}
    </div>
  );
};

export default Container;
