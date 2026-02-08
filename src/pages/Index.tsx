import { useNavigate } from "react-router-dom";
import CassetteLayout from "@/components/CassetteLayout";
import MenuCard from "@/components/MenuCard";

const Index = () => {
  const navigate = useNavigate();

  return (
    <CassetteLayout
      verticalText="AGAIN"
      rightVerticalText="PLAY IT AGAIN"
      bottomRight={
        <button
          onClick={() => navigate("/analytics")}
          className="bg-secondary border-4 border-foreground px-6 py-3 font-display text-xl tracking-widest text-secondary-foreground cursor-pointer shadow-btn hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-btn-hover active:translate-x-0.5 active:translate-y-0.5 active:shadow-btn-active transition-all"
        >
          📊 STATS
        </button>
      }
    >
      {/* Title */}
      <div className="mb-12">
        <h1 className="font-display text-6xl md:text-7xl tracking-[0.25em] text-foreground leading-[0.9]">
          PLAY IT<br />AGAIN
        </h1>
      </div>

      {/* Menu Cards */}
      <div className="flex flex-col gap-6 pr-0 md:pr-12">
        <MenuCard
          icon="🎨"
          title="FREE STYLE"
          description="Practice at your own pace, no limits"
          onClick={() => navigate("/freestyle")}
        />

        <MenuCard
          icon="🎮"
          title="LEVELS"
          description="Progress through structured therapy stages"
          onClick={() => navigate("/levels")}
        />
      </div>
    </CassetteLayout>
  );
};

export default Index;
