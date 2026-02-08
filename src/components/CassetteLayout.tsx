import { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import Barcode from "./Barcode";
import TapeReel from "./TapeReel";

interface CassetteLayoutProps {
  children: ReactNode;
  verticalText?: string;
  showRedPanel?: boolean;
  showBackButton?: boolean;
  rightVerticalText?: string;
  bottomRight?: ReactNode;
}

const CassetteLayout = ({
  children,
  verticalText = "AGAIN",
  showRedPanel = true,
  showBackButton = false,
  rightVerticalText,
  bottomRight,
}: CassetteLayoutProps) => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4 md:p-8 relative z-[1]">
      <div className="max-w-[900px] w-full relative">
        {/* Vertical text on left side outside cassette */}
        {verticalText && (
          <div className="hidden md:block absolute -left-10 top-1/2 -translate-y-1/2 -rotate-90 font-display text-6xl tracking-[0.3em] text-foreground whitespace-nowrap font-black z-10">
            {verticalText}
          </div>
        )}

        <div className="bg-card border-[8px] border-foreground relative shadow-cassette">
          {/* Top border lines */}
          <div className="absolute top-0 left-0 right-0 h-20 border-b-[3px] border-foreground">
            <div className="absolute left-0 right-0 top-5 h-[3px] bg-foreground" />
            <div className="absolute left-0 right-0 top-10 h-[3px] bg-foreground" />
          </div>

          {/* Top barcode */}
          <div className="absolute top-8 right-8 md:right-12 z-10">
            <Barcode variant="top" />
          </div>

          {/* Red panel on left */}
          {showRedPanel && (
            <div className="hidden md:block absolute left-0 top-[100px] w-[35%] bottom-20 bg-primary border-r-4 border-foreground">
              <div className="absolute top-1/2 left-[15%] -translate-x-1/2 -translate-y-1/2 opacity-[0.15] pointer-events-none">
                <TapeReel />
              </div>
            </div>
          )}

          {/* Content area */}
          <div className="p-6 pt-24 md:p-12 md:pt-24 relative min-h-[500px]">
            {/* Back button */}
            {showBackButton && (
              <button
                onClick={() => navigate("/")}
                className="absolute top-[100px] left-4 md:left-[38%] z-10 bg-secondary border-4 border-foreground px-4 py-2 font-display text-lg tracking-widest text-secondary-foreground cursor-pointer shadow-btn hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-btn-hover active:translate-x-0.5 active:translate-y-0.5 active:shadow-btn-active transition-all"
              >
                ◀ BACK
              </button>
            )}

            {/* Main content with left margin for red panel */}
            <div className={showRedPanel ? "md:ml-[38%]" : ""}>
              {children}
            </div>

            {/* Right vertical text */}
            {rightVerticalText && (
              <div className="hidden md:block font-display text-2xl tracking-[0.3em] text-foreground absolute right-8 top-1/2 -translate-y-1/2 rotate-180" style={{ writingMode: "vertical-rl" }}>
                {rightVerticalText}
              </div>
            )}

            {/* Bottom right element */}
            {bottomRight && (
              <div className="absolute bottom-6 right-6">
                {bottomRight}
              </div>
            )}
          </div>

          {/* Bottom barcode */}
          <div className="absolute bottom-6 left-8">
            <Barcode variant="bottom" count={14} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default CassetteLayout;
