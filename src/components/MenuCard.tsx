import { ReactNode } from "react";

interface MenuCardProps {
  icon: string;
  title: string;
  description: string;
  onClick?: () => void;
  children?: ReactNode;
  disabled?: boolean;
}

const MenuCard = ({ icon, title, description, onClick, children, disabled = false }: MenuCardProps) => {
  return (
    <div
      className={`
        bg-primary border-[6px] border-foreground p-8 relative
        shadow-card-retro transition-all duration-300 cursor-pointer
        hover:-translate-x-[3px] hover:-translate-y-[3px] hover:shadow-card-hover
        active:translate-x-[3px] active:translate-y-[3px] active:shadow-card-active
        ${disabled ? "opacity-50 pointer-events-none" : ""}
      `}
      onClick={onClick}
    >
      {/* Inner border */}
      <div className="absolute top-3 left-3 right-3 bottom-3 border-2 border-foreground/30 pointer-events-none" />
      
      <div className="flex items-center gap-6">
        <div className="text-5xl flex-shrink-0">{icon}</div>
        <div className="flex-1">
          <div className="font-display text-4xl tracking-[0.2em] text-primary-foreground drop-shadow-[2px_2px_0_rgba(0,0,0,0.3)] mb-1">
            {title}
          </div>
          <div className="text-primary-foreground font-bold text-sm tracking-wider opacity-90">
            {description}
          </div>
          {children}
        </div>
      </div>
    </div>
  );
};

export default MenuCard;
