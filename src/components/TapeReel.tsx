const TapeReel = ({ size = 100 }: { size?: number }) => {
  return (
    <div
      className="animate-spin-slow"
      style={{ width: size, height: size }}
    >
      <div
        className="w-full h-full rounded-full border-[8px] border-foreground"
        style={{
          background: `
            radial-gradient(circle, transparent 30%, hsl(var(--tape-black)) 30%, hsl(var(--tape-black)) 35%, transparent 35%),
            repeating-conic-gradient(from 0deg, hsl(var(--tape-black)) 0deg, hsl(var(--tape-black)) 10deg, transparent 10deg, transparent 20deg)
          `,
        }}
      />
    </div>
  );
};

export default TapeReel;
