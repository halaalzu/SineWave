interface BarcodeProps {
  variant?: "top" | "bottom";
  count?: number;
}

const Barcode = ({ variant = "top", count = 20 }: BarcodeProps) => {
  const heights = variant === "top" ? "h-20" : "h-12";
  
  return (
    <div className={`flex gap-[2px] ${heights}`}>
      {Array.from({ length: count }).map((_, i) => {
        let width = "w-[3px]";
        if (i % 11 === 0) width = "w-[6px]";
        else if (i % 7 === 0) width = "w-[4px]";
        else if (i % 5 === 0) width = "w-[8px]";
        else if (i % 3 === 0) width = "w-[5px]";
        else if (i % 2 === 0) width = "w-[2px]";
        
        return (
          <span
            key={i}
            className={`bg-foreground ${width} h-full block`}
          />
        );
      })}
    </div>
  );
};

export default Barcode;
